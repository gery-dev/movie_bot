"""Импорт выгрузки Letterboxd на маленькой тестовой БД (без обращения к TMDB)."""
import asyncio
import zipfile

import pandas as pd
import pytest

from data_pipeline.letterboxd import ExportError, Matcher, import_export, letterboxd_rating, normalize
from db.database import connect
from recsys.profile import record_rating

MOVIES = pd.DataFrame([
    ("tt01", "Spirited Away", "Sen to Chihiro no kamikakushi", 2001, 900_000, 129),
    ("tt02", "Léon: The Professional", "Léon", 1994, 1_000_000, 101),
    ("tt03", "Brother", "Brat", 1997, 40_000, 20_992),
    ("tt04", "Brother", "Brother", 2000, 30_000, 555),  # тот же заголовок, другой год
    ("tt05", "Брат 2", "Брат 2", 2000, 60_000, 21_000),  # кириллица — для проверки cp1251
], columns=["tconst", "title", "original_title", "year", "votes", "tmdb_id"])
RATINGS_CSV = ("Date,Name,Year,Letterboxd URI,Rating\n2024-01-01,Брат 2,2000,x,4\n"
               "2024-01-01,Spirited Away,2001,x,5\n")


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


def _db(tmp_path):
    conn = connect(tmp_path / "t.db")
    conn.executemany("INSERT INTO movies (tconst, title, year) VALUES (?, ?, ?)",
                     MOVIES[["tconst", "title", "year"]].itertuples(index=False))
    return conn


@pytest.mark.parametrize("encoding, label", [("utf-8", "utf-8"), ("utf-8-sig", "utf-8"), ("cp1251", "cp1251")])
def test_import_csv_in_any_encoding(tmp_path, encoding, label):
    """UTF-8, UTF-8 с BOM и cp1251 (Excel) читаются одинаково."""
    path = tmp_path / "ratings.csv"
    path.write_bytes(RATINGS_CSV.encode(encoding))
    conn = _db(tmp_path)
    res = asyncio.run(import_export(conn, 7, path, MOVIES))
    assert res.rated == 2 and res.unmatched == [] and res.reason is None
    assert res.encoding == label and res.rows == {"ratings": 2}
    conn.close()


def test_import_zip_in_cp1251(tmp_path):
    zpath = tmp_path / "letterboxd.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("ratings.csv", RATINGS_CSV.encode("cp1251"))
        z.writestr("watched.csv", "Date,Name,Year,Letterboxd URI\n2024-01-01,Brother,2000,x\n".encode("utf-8-sig"))
    conn = _db(tmp_path)
    res = asyncio.run(import_export(conn, 7, zpath, MOVIES))
    assert res.rated == 2 and res.watched == 1
    assert res.encoding == "cp1251,utf-8" and res.rows == {"ratings": 2, "watched": 1}
    conn.close()


@pytest.mark.parametrize("content, reason", [
    ("Date,Name,Year,Letterboxd URI,Rating\n", "empty"),        # заголовок без строк
    ("Title,Score\nSpirited Away,5\n", "no_name_column"),       # чужая таблица
])
def test_import_nothing_to_import_gives_reason(tmp_path, content, reason):
    path = tmp_path / "ratings.csv"
    path.write_text(content, encoding="utf-8")
    conn = _db(tmp_path)
    res = asyncio.run(import_export(conn, 7, path, MOVIES))
    assert res.total == 0 and res.rated == 0 and res.reason == reason
    assert conn.execute("SELECT count(*) FROM ratings").fetchone()[0] == 0
    conn.close()


def test_import_zip_without_known_files_gives_reason(tmp_path):
    zpath = tmp_path / "other.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("diary.csv", "Date,Name,Year\n2024-01-01,Brother,1997\n")
        z.writestr("letterboxd/ratings.csv", RATINGS_CSV)  # внутри папки — не верхний уровень выгрузки
    conn = _db(tmp_path)
    res = asyncio.run(import_export(conn, 7, zpath, MOVIES))
    assert res.total == 0 and res.reason == "no_known_files"
    conn.close()


@pytest.mark.parametrize("raw, reason", [
    (b"", "empty"),
    (b"a,b\n1,2\n1,2,3,4,5\n", "not_csv"),  # строки разной длины
    (bytes([0x98, 0x81]), "encoding"),  # не UTF-8, а в cp1251 такого байта нет
])
def test_unreadable_file_raises_export_error(tmp_path, raw, reason):
    path = tmp_path / "ratings.csv"
    path.write_bytes(raw)
    conn = _db(tmp_path)
    with pytest.raises(ExportError) as e:
        asyncio.run(import_export(conn, 7, path, MOVIES))
    assert e.value.reason == reason
    conn.close()


def test_zip_bomb_is_rejected_without_unpacking(tmp_path, monkeypatch):
    """Сжатый в архиве csv в сотни раз больше архива: распаковывать его целиком нельзя (память сервера)."""
    import data_pipeline.letterboxd as lb
    monkeypatch.setattr(lb, "MAX_FILE_BYTES", 1_000_000)
    zpath = tmp_path / "bomb.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("ratings.csv", "Date,Name,Year,Letterboxd URI,Rating\n" + "0" * 5_000_000)
    assert zpath.stat().st_size < 20_000  # архив маленький, внутри — 5 МБ
    read = []
    monkeypatch.setattr(zipfile.ZipExtFile, "read", lambda self, n=-1, _r=zipfile.ZipExtFile.read: read.append(n) or _r(self, n))
    with pytest.raises(ExportError) as e:
        asyncio.run(import_export(_db(tmp_path), 7, zpath, MOVIES))
    assert e.value.reason == "too_big"
    assert all(0 <= n <= 1_000_001 for n in read)  # целиком не читали


def test_too_many_entries_or_rows_rejected(tmp_path, monkeypatch):
    import data_pipeline.letterboxd as lb
    monkeypatch.setattr(lb, "MAX_ROWS", 3)
    path = tmp_path / "ratings.csv"
    path.write_text("Date,Name,Year,Letterboxd URI,Rating\n" + "2024-01-01,Film,2001,x,4\n" * 4)
    conn = _db(tmp_path)
    with pytest.raises(ExportError) as e:
        asyncio.run(import_export(conn, 7, path, MOVIES))
    assert e.value.reason == "too_big"
    zpath = tmp_path / "many.zip"
    monkeypatch.setattr(lb, "MAX_ZIP_ENTRIES", 5)
    with zipfile.ZipFile(zpath, "w") as z:
        for i in range(6):
            z.writestr(f"f{i}.txt", "x")
    with pytest.raises(ExportError) as e:
        asyncio.run(import_export(conn, 7, zpath, MOVIES))
    assert e.value.reason == "too_big"


def test_tmdb_lookups_are_capped(tmp_path, monkeypatch):
    import data_pipeline.letterboxd as lb
    monkeypatch.setattr(lb, "MAX_TMDB_LOOKUPS", 2)
    path = tmp_path / "ratings.csv"
    path.write_text("Date,Name,Year,Letterboxd URI,Rating\n" + "".join(f"2024-01-01,Fake {i},2001,x,4\n" for i in range(5)))
    calls = []

    class FakeTMDB:
        async def get(self, path, **params):
            calls.append(params["query"])
            return {"results": []}
    res = asyncio.run(import_export(_db(tmp_path), 7, path, MOVIES, FakeTMDB()))
    assert len(calls) == 2 and len(res.unmatched) == 5


def test_forged_size_in_zip_header_is_bad_zip(tmp_path):
    import struct
    zpath = tmp_path / "fake.zip"
    body = "Date,Name,Year,Letterboxd URI,Rating\n" + "0" * 100_000
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("ratings.csv", body)
    data = zpath.read_bytes().replace(struct.pack("<I", len(body)), struct.pack("<I", 1000))  # заголовок врёт
    zpath.write_bytes(data)
    with pytest.raises(ExportError) as e:
        asyncio.run(import_export(_db(tmp_path), 7, zpath, MOVIES))
    assert e.value.reason == "bad_zip"
