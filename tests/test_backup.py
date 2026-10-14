"""Резервная копия создаётся, проверяется, старые удаляются, восстановление возвращает данные."""
from db.backup import backup, restore
from db.database import connect
from recsys.profile import record_rating


def test_backup_and_restore(tmp_path):
    db = tmp_path / "movies.db"
    conn = connect(db)
    conn.execute("INSERT INTO movies (tconst, title) VALUES ('tt01', 'X')")
    record_rating(conn, 1, "tt01", 5)
    conn.commit()

    archives = [backup(db, tmp_path / "b", keep=2) for _ in range(3)]
    kept = sorted((tmp_path / "b").glob("*.db.gz"))
    assert len(kept) == 2 and archives[-1] in kept

    conn.execute("DELETE FROM ratings")  # «авария»: оценки потеряны
    conn.commit()
    conn.close()
    got = restore(archives[-1], db)
    assert got["ratings"] == 1
    assert connect(db).execute("SELECT rating FROM ratings").fetchone()[0] == 5
    assert (tmp_path / "movies.db.before-restore").exists()
