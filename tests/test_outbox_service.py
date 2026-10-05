import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from financial_assistant.publish_outbox import _main, run_publisher

_MODULE = "financial_assistant.publish_outbox"


async def test_loop_recovers_from_database_error_and_sanitizes_logs(caplog):
    stop = asyncio.Event()
    attempts = 0

    async def fake_publish(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("secret connection string")
        stop.set()
        return 1

    with patch(
        f"{_MODULE}.publish_pending_documents", side_effect=fake_publish
    ) as mock_publish:
        await asyncio.wait_for(
            run_publisher(limit=3, poll_interval=0.001, stop=stop), timeout=1
        )
    assert mock_publish.await_count == 2
    mock_publish.assert_awaited_with(limit=3, stop=stop)
    assert "ConnectionError" in caplog.text
    assert "secret connection string" not in caplog.text


async def test_stop_wakes_idle_publisher_without_waiting_for_poll_interval():
    stop = asyncio.Event()
    called = asyncio.Event()

    async def fake_publish(**kwargs):
        called.set()
        return 0

    with patch(
        f"{_MODULE}.publish_pending_documents", side_effect=fake_publish
    ) as mock_publish:
        async with asyncio.TaskGroup() as group:
            task = group.create_task(
                run_publisher(limit=3, poll_interval=3600, stop=stop)
            )
            await asyncio.wait_for(called.wait(), timeout=1)
            # Let the publisher enter its idle wait before requesting shutdown.
            await asyncio.sleep(0)
            stop.set()
            await asyncio.wait_for(task, timeout=1)
    mock_publish.assert_awaited_once_with(limit=3, stop=stop)


async def test_stop_does_not_cancel_in_flight_publication():
    stop = asyncio.Event()
    started = asyncio.Event()
    finish = asyncio.Event()

    async def fake_publish(**kwargs):
        started.set()
        await finish.wait()
        return 1

    with patch(
        f"{_MODULE}.publish_pending_documents", side_effect=fake_publish
    ) as mock_publish:
        async with asyncio.TaskGroup() as group:
            task = group.create_task(run_publisher(limit=3, poll_interval=2, stop=stop))
            await asyncio.wait_for(started.wait(), timeout=1)
            stop.set()
            await asyncio.sleep(0)
            assert not task.done()
            finish.set()
            await asyncio.wait_for(task, timeout=1)
    assert mock_publish.await_count == 1


async def test_preexisting_stop_does_not_open_database():
    stop = asyncio.Event()
    stop.set()
    with patch(f"{_MODULE}.publish_pending_documents") as mock_publish:
        await run_publisher(limit=1, poll_interval=2, stop=stop)
    mock_publish.assert_not_called()


@pytest.mark.parametrize(
    ("limit", "interval"),
    [(0, 2), (-1, 2), (True, 2), (1, 0), (1, -1), (1, float("inf")), (1, float("nan"))],
)
async def test_invalid_loop_settings_fail_before_opening_database(limit, interval):
    with (
        patch(f"{_MODULE}.publish_pending_documents") as mock_publish,
        pytest.raises(ValueError),
    ):
        await run_publisher(limit=limit, poll_interval=interval, stop=asyncio.Event())
    mock_publish.assert_not_called()


@pytest.mark.parametrize("continuous", [False, True])
async def test_cli_installs_shutdown_handlers_and_disposes_engine(continuous):
    loop = asyncio.get_running_loop()
    with (
        patch.object(loop, "add_signal_handler") as add,
        patch.object(loop, "remove_signal_handler") as remove,
        patch(f"{_MODULE}.engine", dispose=AsyncMock()) as engine,
        patch(f"{_MODULE}.publish_pending_documents", return_value=1) as once,
        patch(f"{_MODULE}.run_publisher") as forever,
    ):
        await _main(3, continuous=continuous, poll_interval=2)
    stop = add.call_args_list[0].args[1].__self__
    if continuous:
        forever.assert_awaited_once_with(limit=3, poll_interval=2, stop=stop)
        once.assert_not_awaited()
    else:
        once.assert_awaited_once_with(limit=3, stop=stop)
        forever.assert_not_awaited()
    assert remove.call_count == 2
    engine.dispose.assert_awaited_once_with()


async def test_one_shot_failure_still_disposes_engine():
    loop = asyncio.get_running_loop()
    with (
        patch.object(loop, "add_signal_handler"),
        patch.object(loop, "remove_signal_handler"),
        patch(f"{_MODULE}.engine", dispose=AsyncMock()) as engine,
        patch(f"{_MODULE}.publish_pending_documents", side_effect=ConnectionError),
        pytest.raises(ConnectionError),
    ):
        await _main(3, continuous=False, poll_interval=2)
    engine.dispose.assert_awaited_once_with()
