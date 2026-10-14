"""Резервное копирование БД (ТЗ п. 4.4).

    python -m db.backup             # сделать копию и проверить её
    python -m db.backup --restore data/backups/movies-20261015-040000-000000.db.gz   # восстановить (бот должен быть остановлен)

Копия снимается «на ходу» встроенным в SQLite механизмом backup — бот можно не останавливать.
Каждая копия сразу проверяется: открывается, проходит integrity_check, число пользователей и оценок
совпадает с рабочей БД. Хранятся KEEP последних копий, сжатых gzip.
"""
import argparse
import gzip
import shutil
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path

from db.database import DB_PATH, ROOT

BACKUP_DIR = ROOT / "data" / "backups"
KEEP = 14
CHECK_TABLES = ["users", "ratings", "reactions", "user_lists", "favorites", "movies"]


def counts(conn: sqlite3.Connection) -> dict[str, int]:
    return {t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in CHECK_TABLES}


def backup(db_path: Path = DB_PATH, backup_dir: Path = BACKUP_DIR, keep: int = KEEP) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")  # доли секунды — чтобы копии подряд не затирали друг друга
    target = backup_dir / f"movies-{stamp}.db.gz"
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "copy.db"
        src, dst = sqlite3.connect(db_path), sqlite3.connect(raw)
        with dst:
            src.backup(dst)
        expected = counts(src)
        src.close()
        dst.close()
        check(raw, expected)
        with open(raw, "rb") as f, gzip.open(target, "wb") as g:
            shutil.copyfileobj(f, g)
    for old in sorted(backup_dir.glob("movies-*.db.gz"))[:-keep]:
        old.unlink()
    return target


def check(path: Path, expected: dict[str, int] | None = None) -> dict[str, int]:
    """Копия читается, цела и (если передано) содержит столько же строк, сколько рабочая БД."""
    conn = sqlite3.connect(path)
    try:
        status = conn.execute("PRAGMA integrity_check").fetchone()[0]
        if status != "ok":
            raise RuntimeError(f"копия повреждена: {status}")
        got = counts(conn)
    finally:
        conn.close()
    if expected is not None and got != expected:
        raise RuntimeError(f"в копии другие данные: {got} вместо {expected}")
    return got


def restore(archive: Path, db_path: Path = DB_PATH) -> dict[str, int]:
    """Развернуть копию на место рабочей БД; прежняя БД сохраняется рядом с суффиксом .before-restore."""
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "restored.db"
        with gzip.open(archive, "rb") as g, open(raw, "wb") as f:
            shutil.copyfileobj(g, f)
        got = check(raw)
        if db_path.exists():
            shutil.copy2(db_path, db_path.with_suffix(".db.before-restore"))
        shutil.copy2(raw, db_path)
    return got


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore", type=Path)
    args = parser.parse_args()
    if args.restore:
        print("Восстановлено:", restore(args.restore))
    else:
        path = backup()
        print(f"Копия: {path.relative_to(ROOT)} ({path.stat().st_size / 1e6:.1f} МБ), проверка пройдена")
