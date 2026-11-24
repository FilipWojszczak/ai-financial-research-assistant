"""The full pipeline as a fixture: API, outbox publisher, RabbitMQ and a worker."""

import asyncio
import os
import signal
import sys
import uuid
from contextlib import suppress
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote, urlsplit

import httpx
import pytest
from kombu import Connection
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from financial_assistant.api.server import app
from financial_assistant.core import document_storage
from financial_assistant.core.config import get_settings
from financial_assistant.core.db import get_session
from financial_assistant.core.messaging import DEAD_LETTER_QUEUE
from financial_assistant.core.security import create_access_token
from financial_assistant.models import User

ROOT = Path(__file__).resolve().parents[2]


def pdf_bytes():
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
    )
    stream = DecodedStreamObject()
    stream.set_data(
        b"BT /F1 12 Tf 50 700 Td (Alice leads Acme. Acme reports revenue.) Tj ET"
    )
    page[NameObject("/Contents")] = writer._add_object(stream)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


async def wait_until(check, *, description, processes):
    async with asyncio.timeout(45):
        while True:
            for process, _ in processes:
                if process.returncode is not None:
                    raise AssertionError(
                        f"Ingestion subprocess exited ({process.returncode}); "
                        "inspect the test's temporary log files"
                    )
            if await check():
                return
            await asyncio.sleep(0.1)


@pytest.fixture
async def flow(private_database, tmp_path, monkeypatch):
    settings = get_settings()
    broker = Connection(settings.broker_url)
    assert broker.hostname and broker.userid and broker.password
    host = broker.hostname
    if ":" in host:
        host = f"[{host}]"
    scheme = "https" if broker.ssl else "http"
    management = httpx.AsyncClient(
        base_url=f"{scheme}://{host}:{settings.rabbitmq_management_port}",
        auth=(broker.userid, broker.password),
        timeout=5,
    )
    vhost = "ingestion_test_" + uuid.uuid4().hex
    response = await management.put(f"/api/vhosts/{vhost}", json={})
    response.raise_for_status()
    processes = []
    logs = []
    try:
        response = await management.put(
            f"/api/permissions/{vhost}/{quote(broker.userid, safe='')}",
            json={"configure": ".*", "write": ".*", "read": ".*"},
        )
        response.raise_for_status()
        broker_url = urlsplit(settings.broker_url)._replace(path="/" + vhost).geturl()
        storage = tmp_path / "sources"
        monkeypatch.setattr(
            document_storage,
            "get_settings",
            lambda: SimpleNamespace(document_storage_path=storage),
        )
        environment = {
            **os.environ,
            "DATABASE_URL_OVERRIDE": private_database.url.get_secret_value(),
            "RABBITMQ_URL_OVERRIDE": broker_url,
            "DOCUMENT_STORAGE_PATH": str(storage),
            "GOOGLE_API_KEY": "mock-google-api-key-for-testing",
            "AI_REQUEST_TIMEOUT_SECONDS": "0.1",
            "AI_REQUEST_ATTEMPTS": "3",
            "INGESTION_TEST_DIRECTORY": str(tmp_path),
        }

        async def start(module, *args, failure=False):
            log = (tmp_path / f"{module}.log").open("wb")
            logs.append(log)
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                module,
                *args,
                cwd=ROOT,
                env={**environment, "INGESTION_TEST_FAILURE": "1" if failure else "0"},
                stdout=log,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append((process, module))
            return process

        initialize = await start("financial_assistant.services.configure_broker")
        assert await asyncio.wait_for(initialize.wait(), 20) == 0
        processes.remove((initialize, "financial_assistant.services.configure_broker"))

        async def retention_ready():
            response = await management.get(f"/api/queues/{vhost}/{DEAD_LETTER_QUEUE}")
            response.raise_for_status()
            arguments = response.json().get("arguments", {})
            return arguments.get("x-delivery-limit") == -1

        # RabbitMQ publishes queue statistics asynchronously after declaration.
        await wait_until(
            retention_ready, description="failure queue retention", processes=processes
        )

        async def get_test_session():
            async with private_database.factory() as session:
                yield session

        app.dependency_overrides[get_session] = get_test_session
        async with private_database.factory() as session, session.begin():
            user = User(email=f"{vhost}@example.com", hashed_password="unused")
            session.add(user)
            await session.flush()
            token = create_access_token(user.id)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            headers={"Authorization": f"Bearer {token}"},
        ) as client:

            async def upload(failure=False):
                await start("tests.e2e.ingestion_worker", failure=failure)

                async def ready():
                    return (tmp_path / "ready").exists()

                await wait_until(
                    ready, description="worker readiness", processes=processes
                )
                response = await client.post(
                    "/documents/",
                    data={
                        "company_ticker": "ACME",
                        "document_type": "10-K",
                        "year": "2026",
                        "is_public": "false",
                    },
                    files={"file": ("report.pdf", pdf_bytes(), "application/pdf")},
                )
                assert response.status_code == 202, response.text
                assert response.json()["status"] == "processing"
                document_id = response.json()["id"]
                await start(
                    "financial_assistant.services.publish_outbox",
                    "--loop",
                    "--poll-interval",
                    "0.1",
                )
                return document_id

            async def wait_status(document_id, status):
                async def check():
                    response = await client.get(f"/documents/{document_id}")
                    assert response.status_code == 200
                    return response.json()["status"] == status

                await wait_until(check, description=status, processes=processes)

            async def queue_count():
                response = await management.get(
                    f"/api/queues/{vhost}/{DEAD_LETTER_QUEUE}"
                )
                response.raise_for_status()
                return response.json().get("messages", 0)

            yield SimpleNamespace(
                client=client,
                upload=upload,
                wait_status=wait_status,
                start=start,
                queue_count=queue_count,
                processes=processes,
                factory=private_database.factory,
                storage=storage,
                directory=tmp_path,
            )
    finally:
        app.dependency_overrides.pop(get_session, None)
        for process, _ in reversed(processes):
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), 10)
                except TimeoutError:
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    await process.wait()
        for log in logs:
            log.close()
        response = await management.delete(f"/api/vhosts/{vhost}")
        await management.aclose()
        assert response.status_code in {204, 404}
