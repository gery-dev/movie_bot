"""Подбор фильмов для калибровки вкуса: 8–10 карточек разных жанров, эпох и стран.

Следующий фильм выбирается с учётом уже показанных и ответов на них:
- новизна — жанры, десятилетие и страна, которых ещё не было;
- узнаваемость — чем чаще пользователь отвечает «не смотрел», тем известнее следующие фильмы;
- стартовые жанры из знакомства немного поднимают подходящие фильмы;
- небольшая случайность, чтобы разные пользователи видели разные наборы.
"""
import numpy as np
import pandas as pd

CALIBRATION_SIZE = 10
MIN_VOTES = 20_000   # фильм должен быть хоть сколько-то известен, иначе его почти никто не видел
MIN_RATING = 6.0


def calibration_pool(movies: pd.DataFrame) -> pd.DataFrame:
    pool = movies[(movies.votes >= MIN_VOTES) & (movies.rating >= MIN_RATING) & movies.genres.notna()].copy()
    pool["genre_set"] = pool.genres.str.split(",").map(set)
    pool["decade"] = (pool.year // 10 * 10).astype(int)
    pool["country1"] = pool.country.fillna("").str.split(",").str[0]
    fame = np.log10(pool.votes)
    pool["fame"] = (fame - fame.min()) / (fame.max() - fame.min())
    return pool


def next_film(pool: pd.DataFrame, shown: list[str], not_seen: int = 0, start_genres: set[str] | None = None,
              rng: np.random.Generator | None = None) -> str:
    """shown — уже показанные tconst; not_seen — сколько раз ответили «не смотрел»."""
    rng = rng or np.random.default_rng()
    rest = pool[~pool.index.isin(shown)]
    seen_rows = pool.loc[shown] if shown else pool.iloc[:0]
    shown_genres = set().union(*seen_rows.genre_set) if shown else set()

    new_genres = rest.genre_set.map(lambda g: len(g - shown_genres)).clip(upper=3) / 3
    new_decade = (~rest.decade.isin(set(seen_rows.decade))).astype(float)
    new_country = (~rest.country1.isin(set(seen_rows.country1))).astype(float)
    # веса подобраны в recsys/tune.py: при большем весе десятилетия калибровка уходит в 1920–30-е,
    # которые мало кто видел; при большем весе известности остаются одни хиты
    fame_weight = 1.0 + 0.3 * not_seen
    score = 1.5 * new_genres + 0.5 * new_decade + 0.5 * new_country + fame_weight * rest.fame
    if start_genres and len(shown) < 3:  # первые карточки — из любимых жанров, чтобы начать с понятного
        score += 0.7 * rest.genre_set.map(lambda g: bool(g & start_genres)).astype(float)
    score += rng.uniform(0, 0.3, len(rest))
    return score.idxmax()
