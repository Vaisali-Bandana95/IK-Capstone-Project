import os
import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver


def create_persistent_checkpointer(path: str | Path | None = None) -> SqliteSaver:
    configured_path = path or os.getenv(
        "CHECKPOINT_DB_PATH",
        str(Path(__file__).resolve().parents[2] / "app" / "data" / "checkpoints.sqlite3"),
    )
    checkpoint_path = Path(configured_path).expanduser().resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(
        str(checkpoint_path),
        check_same_thread=False,
        timeout=30,
    )
    connection.execute("PRAGMA busy_timeout = 30000")
    connection.execute("PRAGMA journal_mode = WAL")
    checkpointer = SqliteSaver(connection)
    checkpointer.setup()
    return checkpointer
