import pytest_asyncio

from tests.integration.retrieval.seeding import Seeder


@pytest_asyncio.fixture
async def seed(private_database) -> Seeder:
    return Seeder(private_database.factory)
