import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "movies.db"
SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    """Открыть БД и создать таблицы, если их ещё нет."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    _migrate(conn)
    return conn


# столбцы, добавленные после создания таблиц: CREATE TABLE IF NOT EXISTS их в старую БД не добавит
NEW_COLUMNS = [("recommendations", "mode", "TEXT")]  # recommend | today | query — для онлайн-статистики


def _migrate(conn: sqlite3.Connection) -> None:
    for table, column, kind in NEW_COLUMNS:
        if column not in {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")
    conn.commit()
