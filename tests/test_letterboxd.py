"""Импорт выгрузки Letterboxd на маленькой тестовой БД (без обращения к TMDB)."""
import asyncio
import zipfile

import pandas as pd
import pytest

from data_pipeline.letterboxd import Matcher, import_export, letterboxd_rating, normalize
from db.database import connect
from recsys.profile import record_rating

MOVIES = pd.DataFrame([
    ("tt01", "Spirited Away", "Sen to Chihiro no kamikakushi", 2001, 900_000, 129),
    ("tt02", "Léon: The Professional", "Léon", 1994, 1_000_000, 101),
    ("tt03", "Brother", "Brat", 1997, 40_000, 20_992),
    ("tt04", "Brother", "Brother", 2000, 30_000, 555),  # тот же заголовок, другой год
], columns=["tconst", "title", "original_title", "year", "votes", "tmdb_id"])


@pytest.mark.parametrize("stars, ours", [(0.5, 1), (1, 1), (2.5, 3), (3.5, 4), (4.5, 5), (5, 5)])
def test_rating_scale(stars, ours):
    assert letterboxd_rating(stars) == ours


def test_matcher_uses_year_and_original_title():
    m = Matcher(MOVIES)
    assert m.find("Brother", 1997) == "tt03" and m.find("Brother", 2000) == "tt04"
    assert m.find("Brother", 1998) == "tt03"  # допуск ±1 год
    assert m.find("Sen to Chihiro no Kamikakushi", 2001) == "tt01"
    assert m.find("Leon the professional", 1994) is None  # без «é» — другая строка
    assert normalize("Léon: The Professional") == "léon the professional"


def test_import_zip(tmp_path):
    zpath = tmp_path / "letterboxd.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("ratings.csv", "Date,Name,Year,Letterboxd URI,Rating\n"
                                  "2024-01-01,Spirited Away,2001,x,4.5\n2024-01-01,Brother,1997,x,2.5\n"
                                  "2024-01-01,Unknown Film,2019,x,3\n")
        z.writestr("watched.csv", "Date,Name,Year,Letterboxd URI\n2024-01-01,Brother,2000,x\n")
        z.writestr("watchlist.csv", "Date,Name,Year,Letterboxd URI\n2024-01-01,Léon: The Professional,1994,x\n")
        z.writestr("deleted/ratings.csv", "Date,Name,Year,Letterboxd URI,Rating\n2024-01-01,Brother,2000,x,1\n")
    conn = connect(tmp_path / "t.db")
    conn.executemany("INSERT INTO movies (tconst, title, year) VALUES (?, ?, ?)",
                     MOVIES[["tconst", "title", "year"]].itertuples(index=False))
    record_rating(conn, 7, "tt01", 2)  # оценка в боте важнее импортированной
    res = asyncio.run(import_export(conn, 7, zpath, MOVIES))
    ratings = dict(conn.execute("SELECT tconst, rating FROM ratings WHERE user_id = 7").fetchall())
    lists = {tuple(r) for r in conn.execute("SELECT tconst, list FROM user_lists WHERE user_id = 7")}
    assert ratings == {"tt01": 2, "tt03": 3}
    assert ("tt04", "seen") in lists and ("tt02", "planned") in lists
    assert res.rated == 1 and res.skipped_existing == 1 and res.unmatched == ["Unknown Film (2019)"]
    conn.close()
