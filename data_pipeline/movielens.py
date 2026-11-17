"""MovieLens: скачивание и связь с фильмами IMDb в нашей БД.

    python -m data_pipeline.movielens download ml-latest-small  # 1 МБ, для опытов
    python -m data_pipeline.movielens download ml-32m           # 239 МБ, для итоговых замеров
    python -m data_pipeline.movielens link ml-latest-small      # доля фильмов и оценок, нашедших пару

Лицензия MovieLens запрещает распространять данные: они лежат только в data/.
https://grouplens.org/datasets/movielens/
"""
import argparse
import zipfile

import httpx
import pandas as pd

from db.database import connect
from data_pipeline.imdb import RAW_DIR

BASE_URL = "https://files.grouplens.org/datasets/movielens"
ML_DIR = RAW_DIR / "movielens"


def download(name: str) -> None:
    ML_DIR.mkdir(parents=True, exist_ok=True)
    if (ML_DIR / name / "ratings.csv").exists():
        print(f"{name}: уже скачан")
        return
    archive = ML_DIR / f"{name}.zip"
    with httpx.stream("GET", f"{BASE_URL}/{name}.zip", timeout=60, follow_redirects=True) as r:
        r.raise_for_status()
        print(f"{name}: качаю {int(r.headers['content-length']) / 1e6:.0f} МБ")
        with open(archive, "wb") as f:
            for block in r.iter_bytes(1 << 20):
                f.write(block)
    with zipfile.ZipFile(archive) as z:
        z.extractall(ML_DIR)  # в архиве уже есть папка с именем датасета
    archive.unlink()


def load_links(name: str) -> pd.DataFrame:
    """movieId MovieLens → tconst IMDb (в links.csv imdbId хранится без 'tt' и ведущих нулей)."""
    links = pd.read_csv(ML_DIR / name / "links.csv", usecols=["movieId", "imdbId"], dtype={"imdbId": str})
    links["tconst"] = "tt" + links["imdbId"].str.zfill(7)
    return links[["movieId", "tconst"]]


def load_ratings(name: str) -> pd.DataFrame:
    """Оценки MovieLens с колонкой tconst; оценки фильмов, которых нет в нашей БД, отброшены."""
    ratings = pd.read_csv(ML_DIR / name / "ratings.csv")
    ratings = ratings.merge(load_links(name), on="movieId")
    with connect() as conn:
        known = {row[0] for row in conn.execute("SELECT tconst FROM movies WHERE kind = 'movie'")}
    return ratings[ratings["tconst"].isin(known)]


def link(name: str) -> None:
    links = load_links(name)
    ratings = pd.read_csv(ML_DIR / name / "ratings.csv", usecols=["userId", "movieId"])
    with connect() as conn:
        known = {row[0] for row in conn.execute("SELECT tconst FROM movies WHERE kind = 'movie'")}
        db_total = conn.execute("SELECT count(*) FROM movies").fetchone()[0]

    links["in_db"] = links["tconst"].isin(known)
    ratings = ratings.merge(links[["movieId", "in_db"]], on="movieId", how="left")
    ratings["in_db"] = ratings["in_db"].fillna(False).astype(bool)
    kept = ratings[ratings["in_db"]]

    print(f"{name}")
    print(f"  фильмов MovieLens: {len(links)}, нашли пару в БД: {links['in_db'].sum()} ({links['in_db'].mean():.1%})")
    print(f"  оценок: {len(ratings)}, из них по фильмам из БД: {len(kept)} ({len(kept) / len(ratings):.1%})")
    print(f"  пользователей: {ratings['userId'].nunique()}, с оценками фильмов из БД: {kept['userId'].nunique()}")
    print(f"  фильмов БД, у которых есть оценки MovieLens: {links['in_db'].sum()} из {db_total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["download", "link"])
    parser.add_argument("name", choices=["ml-latest-small", "ml-32m"])
    args = parser.parse_args()
    download(args.name) if args.command == "download" else link(args.name)
