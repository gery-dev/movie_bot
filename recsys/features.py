"""Признаки фильмов для контентной модели.

    python -m recsys.features embeddings  # посчитать эмбеддинги описаний всех фильмов (один раз)

Мета-признаки (жанры, режиссёры, актёры, десятилетие, страна, язык) собираются из БД за секунды.
Эмбеддинги описаний — векторы смысла текста от sentence-transformers — считаются долго,
поэтому сохраняются в data/processed/ и потом только загружаются.
"""
import argparse
import asyncio

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MultiLabelBinarizer, normalize

from data_pipeline.tmdb import TMDB
from db.database import ROOT, connect

PROCESSED = ROOT / "data" / "processed"
EMB_PATH = PROCESSED / "embeddings.npy"
EMB_IDS_PATH = PROCESSED / "embeddings_ids.txt"
EMB_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_ACTORS = 3


def meta_features(tconsts: list[str]) -> sparse.csr_matrix:
    """Строки — фильмы в порядке tconsts, столбцы — признаки вида genre:Drama, director:nm…

    Веса TF-IDF: редкий признак (конкретный режиссёр) значит больше частого (драма).
    Строки нормированы, так что произведение строк — косинусная близость.
    """
    conn = connect()
    movies = pd.read_sql("SELECT tconst, year, genres, country, language FROM movies", conn).set_index("tconst")
    people = pd.read_sql(
        f"SELECT tconst, nconst, role FROM movie_people WHERE role = 'director' OR ord <= {TOP_ACTORS}", conn
    )
    conn.close()

    movies = movies.reindex(tconsts)
    movies[["genres", "country", "language"]] = movies[["genres", "country", "language"]].fillna("")
    tags = {t: [] for t in tconsts}
    for t, row in movies.iterrows():
        tags[t] += [f"genre:{g}" for g in row.genres.split(",") if g]
        tags[t] += [f"country:{c}" for c in row.country.split(",") if c]
        if pd.notna(row.year):
            tags[t].append(f"decade:{int(row.year) // 10 * 10}")
        if row.language:
            tags[t].append(f"lang:{row.language}")
    for p in people.itertuples():
        if p.tconst in tags:
            tags[p.tconst].append(f"{p.role}:{p.nconst}")

    mlb = MultiLabelBinarizer(sparse_output=True)
    X = mlb.fit_transform([tags[t] for t in tconsts]).tocsr().astype(np.float32)
    X = TfidfTransformer().fit_transform(X)
    return normalize(X).tocsr()


async def _texts() -> pd.DataFrame:
    conn = connect()
    movies = pd.read_sql("SELECT tconst, tmdb_id, title, title_ru, overview_ru FROM movies", conn)
    missing = movies[movies.overview_ru.isna() & movies.tmdb_id.notna()]
    tmdb = TMDB(conn)
    try:
        en = await asyncio.gather(*(tmdb.overview_en(int(i)) for i in missing.tmdb_id))
    finally:
        conn.commit()
        await tmdb.close()
        conn.close()
    movies.loc[missing.index, "overview_en"] = en
    print(f"Описаний на русском: {movies.overview_ru.notna().sum()}, взято английских: {sum(e is not None for e in en)}")
    # в начало текста — название, так у фильмов без описания остаётся хоть что-то
    title = movies.title_ru.fillna(movies.title)
    body = movies.overview_ru.fillna(movies.get("overview_en")).fillna("")
    movies["text"] = (title + ". " + body).str.strip()
    return movies[["tconst", "text"]]


def build_embeddings() -> None:
    from sentence_transformers import SentenceTransformer

    texts = asyncio.run(_texts())
    model = SentenceTransformer(EMB_MODEL)
    emb = model.encode(texts.text.tolist(), batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    np.save(EMB_PATH, emb.astype(np.float32))
    EMB_IDS_PATH.write_text("\n".join(texts.tconst), encoding="utf-8")
    print(f"Сохранено {emb.shape[0]} векторов размерности {emb.shape[1]} в {EMB_PATH.relative_to(ROOT)}")


def load_embeddings(tconsts: list[str]) -> np.ndarray:
    """Эмбеддинги в порядке tconsts (строки нормированы)."""
    ids = EMB_IDS_PATH.read_text(encoding="utf-8").split("\n")
    index = {t: i for i, t in enumerate(ids)}
    return np.load(EMB_PATH)[[index[t] for t in tconsts]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["embeddings"])
    parser.parse_args()
    build_embeddings()
