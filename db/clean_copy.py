"""Копия БД без данных пользователей — для первого развёртывания на сервер «с нуля».

    python -m db.clean_copy   # → data/movies-clean.db

Остаются фильмы, люди, жанры и кэш TMDB; удаляются пользователи, оценки, реакции, профили, списки,
любимые, компании, журнал рекомендаций и опросы. Рабочая БД не меняется.
"""
import sqlite3

from db.database import DB_PATH, ROOT

CLEAN_PATH = ROOT / "data" / "movies-clean.db"
USER_TABLES = ["events", "survey", "recommendations", "group_members", "groups", "user_names", "favorites", "user_lists",
               "daily_session", "user_profile", "reactions", "ratings", "users"]  # порядок — из-за внешних ключей


def main() -> None:
    CLEAN_PATH.unlink(missing_ok=True)
    src, dst = sqlite3.connect(DB_PATH), sqlite3.connect(CLEAN_PATH)
    with dst:
        src.backup(dst)
    src.close()
    with dst:
        for table in USER_TABLES:
            dst.execute(f"DELETE FROM {table}")
    dst.execute("VACUUM")
    left = {t: dst.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in USER_TABLES}
    movies = dst.execute("SELECT count(*) FROM movies").fetchone()[0]
    dst.close()
    assert not any(left.values()), left
    print(f"{CLEAN_PATH.relative_to(ROOT)}: фильмов {movies}, данных пользователей нет "
          f"({CLEAN_PATH.stat().st_size / 1e6:.0f} МБ)")


if __name__ == "__main__":
    main()
