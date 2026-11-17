"""Загрузка датасетов IMDb в SQLite.

    python -m data_pipeline.imdb download              # скачать файлы в data/raw/
    python -m data_pipeline.imdb load --min-votes 5000 # отфильтровать и загрузить в БД (фильмы и сериалы)

Типы IMDb: movie → kind = movie; tvSeries и tvMiniSeries → kind = series. Связи с людьми и жанрами
пересобираются целиком, поэтому фильмы в --types должны быть всегда (иначе у них пропадут режиссёры и актёры).

Данные IMDb разрешено использовать только в некоммерческих целях:
https://developer.imdb.com/non-commercial-datasets/
"""
import argparse
import csv
import time
from pathlib import Path

import httpx
import pandas as pd

from db.database import ROOT, connect

RAW_DIR = ROOT / "data" / "raw"
TYPES = ("movie", "tvSeries", "tvMiniSeries")  # телефильмы (tvMovie) пока не берём — решение пользователя
BASE_URL = "https://datasets.imdbws.com"
FILES = ["title.basics", "title.ratings", "title.crew", "title.principals", "name.basics", "title.akas"]
CHUNK = 1_000_000
TOP_ACTORS = 5


def download() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        for name in FILES:
            url = f"{BASE_URL}/{name}.tsv.gz"
            path = RAW_DIR / f"{name}.tsv.gz"
            size = int(client.head(url).headers["content-length"])
            if path.exists() and path.stat().st_size == size:
                print(f"{name}: уже скачан")
                continue
            print(f"{name}: качаю {size / 1e6:.0f} МБ")
            tmp = path.with_suffix(".part")
            with client.stream("GET", url) as r, open(tmp, "wb") as f:
                r.raise_for_status()
                for block in r.iter_bytes(1 << 20):
                    f.write(block)
            tmp.rename(path)


def read_tsv(name: str, usecols: list[str], chunksize: int | None = CHUNK):
    # quoting=QUOTE_NONE: в названиях встречаются кавычки, IMDb их не экранирует
    return pd.read_csv(
        RAW_DIR / f"{name}.tsv.gz", sep="\t", usecols=usecols, dtype=str,
        na_values="\\N", keep_default_na=False, quoting=csv.QUOTE_NONE, chunksize=chunksize,
    )


def filter_chunks(name: str, usecols: list[str], keep) -> pd.DataFrame:
    """Читать файл частями и оставлять строки, для которых keep(chunk) истинно."""
    parts = [chunk[keep(chunk)] for chunk in read_tsv(name, usecols)]
    return pd.concat(parts, ignore_index=True)


def to_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def none_if_na(rows):
    # sqlite3 пишет числа numpy как BLOB, поэтому переводим их в обычные int/float
    return [tuple(None if pd.isna(v) else v.item() if hasattr(v, "item") else v for v in row) for row in rows]


def load(min_votes: int, types: tuple[str, ...] = TYPES) -> None:
    if "movie" not in types:
        raise ValueError("в --types нужен movie: связи с людьми и жанрами пересобираются для всего каталога")
    t0 = time.time()

    ratings = read_tsv("title.ratings", ["tconst", "averageRating", "numVotes"], chunksize=None)
    ratings["numVotes"] = to_int(ratings["numVotes"])
    ratings = ratings[ratings["numVotes"] >= min_votes]
    ids = set(ratings["tconst"])

    basics = filter_chunks(
        "title.basics",
        ["tconst", "titleType", "primaryTitle", "originalTitle", "isAdult", "startYear", "endYear", "runtimeMinutes",
         "genres"],
        lambda c: c["titleType"].isin(types) & (c["isAdult"] == "0") & c["tconst"].isin(ids),
    )
    movies = basics.merge(ratings, on="tconst")
    ids = set(movies["tconst"])
    movies["kind"] = (movies["titleType"] != "movie").map({True: "series", False: "movie"})
    print(f"С числом голосов ≥ {min_votes}: {movies['kind'].value_counts().to_dict()}")

    akas = filter_chunks(
        "title.akas", ["titleId", "ordering", "title", "region", "types"],
        lambda c: (c["region"] == "RU") & c["titleId"].isin(ids),
    )
    # если русских названий несколько, берём то, что IMDb показывает (imdbDisplay), иначе первое
    akas["display"] = akas["types"].fillna("").str.contains("imdbDisplay")
    akas["ordering"] = to_int(akas["ordering"])
    akas = akas.sort_values(["titleId", "display", "ordering"], ascending=[True, False, True])
    title_ru = akas.drop_duplicates("titleId").set_index("titleId")["title"]
    movies["title_ru"] = movies["tconst"].map(title_ru)

    crew = filter_chunks("title.crew", ["tconst", "directors"], lambda c: c["tconst"].isin(ids))
    directors = crew.dropna(subset=["directors"]).assign(nconst=lambda d: d["directors"].str.split(","))
    directors = directors.explode("nconst")[["tconst", "nconst"]].assign(role="director", ord=None)

    actors = filter_chunks(
        "title.principals", ["tconst", "ordering", "nconst", "category"],
        lambda c: c["category"].isin(["actor", "actress"]) & c["tconst"].isin(ids),
    )
    actors["ord"] = to_int(actors["ordering"])
    actors = actors.sort_values(["tconst", "ord"]).groupby("tconst").head(TOP_ACTORS)
    actors = actors[["tconst", "nconst"]].assign(role="actor", ord=actors["ord"])
    movie_people = pd.concat([directors, actors], ignore_index=True).drop_duplicates(["tconst", "nconst", "role"])

    person_ids = set(movie_people["nconst"])
    people = filter_chunks(
        "name.basics", ["nconst", "primaryName", "birthYear", "deathYear"],
        lambda c: c["nconst"].isin(person_ids),
    )
    movie_people = movie_people[movie_people["nconst"].isin(set(people["nconst"]))]

    genres = movies.dropna(subset=["genres"]).assign(genre=lambda d: d["genres"].str.split(","))
    genres = genres.explode("genre")[["tconst", "genre"]]

    for col in ["startYear", "endYear", "runtimeMinutes"]:
        movies[col] = to_int(movies[col])
    # у сериала в runtimeMinutes длина серии: в runtime её не кладём, иначе фильтры фильмов («до 2 часов») её поймают
    series = movies["kind"] == "series"
    movies["episodeRuntime"] = movies["runtimeMinutes"].where(series)
    movies["runtimeMinutes"] = movies["runtimeMinutes"].where(~series)
    movies["endYear"] = movies["endYear"].where(series)
    movies["averageRating"] = pd.to_numeric(movies["averageRating"])
    people["birthYear"] = to_int(people["birthYear"])
    people["deathYear"] = to_int(people["deathYear"])

    conn = connect()
    with conn:
        # фильмы и людей обновляем на месте: на них могут ссылаться оценки пользователей
        conn.executemany(
            # русское название IMDb не затирает уже дописанное из TMDB (у фильмов его там больше)
            """INSERT INTO movies (tconst, title, original_title, title_ru, year, runtime, genres, rating, votes,
                                   kind, end_year, episode_runtime)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(tconst) DO UPDATE SET
                 title = excluded.title, original_title = excluded.original_title,
                 title_ru = COALESCE(excluded.title_ru, movies.title_ru), year = excluded.year,
                 runtime = excluded.runtime, genres = excluded.genres, rating = excluded.rating,
                 votes = excluded.votes, kind = excluded.kind, end_year = excluded.end_year,
                 episode_runtime = excluded.episode_runtime""",
            none_if_na(movies[["tconst", "primaryTitle", "originalTitle", "title_ru", "startYear",
                               "runtimeMinutes", "genres", "averageRating", "numVotes", "kind", "endYear",
                               "episodeRuntime"]].itertuples(index=False)),
        )
        conn.executemany(
            """INSERT INTO people (nconst, name, birth_year, death_year) VALUES (?, ?, ?, ?)
               ON CONFLICT(nconst) DO UPDATE SET
                 name = excluded.name, birth_year = excluded.birth_year, death_year = excluded.death_year""",
            none_if_na(people[["nconst", "primaryName", "birthYear", "deathYear"]].itertuples(index=False)),
        )
        conn.execute("DELETE FROM movie_genres")
        conn.executemany("INSERT INTO movie_genres VALUES (?, ?)", genres.itertuples(index=False))
        conn.execute("DELETE FROM movie_people")
        conn.executemany(
            "INSERT INTO movie_people VALUES (?, ?, ?, ?)",
            none_if_na(movie_people[["tconst", "nconst", "role", "ord"]].itertuples(index=False)),
        )
    conn.close()

    print(f"Русских названий: {movies['title_ru'].notna().sum()}")
    print(f"Людей: {len(people)}, связей фильм—человек: {len(movie_people)}")
    print(f"Готово за {time.time() - t0:.0f} с")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["download", "load"])
    parser.add_argument("--min-votes", type=int, default=5000)
    parser.add_argument("--types", default=",".join(TYPES), help="типы IMDb через запятую; movie обязателен")
    args = parser.parse_args()
    if args.command == "download":
        download()
    else:
        load(args.min_votes, tuple(args.types.split(",")))
