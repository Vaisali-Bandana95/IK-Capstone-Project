import os
import shutil
import threading
from pathlib import Path
from urllib.request import urlopen

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import StaticPool
from dotenv import load_dotenv

load_dotenv()

engine: Engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

EXPECTED_TABLES = {
    "Album",
    "Artist",
    "Customer",
    "Employee",
    "Genre",
    "Invoice",
    "InvoiceLine",
    "MediaType",
    "Playlist",
    "PlaylistTrack",
    "Track",
}

_initialization_lock = threading.Lock()
_initialized = False


def _default_sql_path() -> Path:
    repository_root = Path(__file__).resolve().parents[2]
    configured_path = os.getenv("CHINOOK_CACHE_PATH")
    if configured_path:
        cache_path = Path(configured_path)
        return cache_path if cache_path.is_absolute() else repository_root / cache_path
    return repository_root / "app" / "cache" / "Chinook_Sqlite.sql"


def _ensure_sql_script(cache_path: Path) -> Path:
    if cache_path.exists():
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    supplied_path = Path(__file__).resolve().parents[2] / "Chinook_Sqlite.sql"
    if supplied_path.exists():
        shutil.copyfile(supplied_path, cache_path)
        return cache_path

    sql_url = os.getenv("CHINOOK_SQL_URL")
    if not sql_url:
        raise FileNotFoundError(
            "Chinook SQL is not cached and CHINOOK_SQL_URL is not configured"
        )

    with urlopen(sql_url, timeout=30) as response:
        sql_script = response.read()
    if not sql_script.strip():
        raise ValueError("Downloaded Chinook SQL script is empty")

    temporary_path = cache_path.with_name(f"{cache_path.name}.tmp")
    temporary_path.write_bytes(sql_script)
    temporary_path.replace(cache_path)
    return cache_path


def initialize_database(sql_path: str | Path | None = None) -> None:
    global _initialized

    with _initialization_lock:
        if _initialized:
            return

        script_path = (
            Path(sql_path)
            if sql_path is not None
            else _ensure_sql_script(_default_sql_path())
        )
        sql_script = script_path.read_text(encoding="utf-8")
        connection = engine.raw_connection()
        try:
            connection.driver_connection.executescript(sql_script)
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

        _initialized = True


def verify_database() -> dict[str, bool | int]:
    table_count = 0
    try:
        with engine.connect() as connection:
            table_names = set(
                connection.execute(
                    text("SELECT name FROM sqlite_master WHERE type = :table_type"),
                    {"table_type": "table"},
                ).scalars()
            )
            table_count = len(table_names)
            if not EXPECTED_TABLES.issubset(table_names):
                return {"healthy": False, "table_count": table_count}

            connection.execute(text("SELECT COUNT(*) FROM Artist")).scalar_one()
        return {"healthy": True, "table_count": table_count}
    except SQLAlchemyError:
        return {"healthy": False, "table_count": table_count}


def health_check() -> bool:
    return bool(verify_database()["healthy"])
