import pytest

from src.db import initialize_database


@pytest.fixture(scope="session", autouse=True)
def initialized_database() -> None:
    initialize_database()
