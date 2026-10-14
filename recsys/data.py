"""Подготовка оценок MovieLens для офлайн-проверки моделей."""
import pandas as pd

from data_pipeline.movielens import load_ratings

RELEVANT = 4.0  # оценка, начиная с которой фильм считаем понравившимся
TEST_SHARE = 0.2


def temporal_split(ratings: pd.DataFrame, test_share: float = TEST_SHARE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """У каждого пользователя последние по времени test_share оценок уходят в тест.

    Порядок при равном времени задаёт movieId, поэтому разбиение всегда одинаковое.
    """
    r = ratings.sort_values(["userId", "timestamp", "movieId"])
    pos = r.groupby("userId").cumcount()
    size = r.groupby("userId")["movieId"].transform("size")
    n_test = (size * test_share).round().clip(lower=1)
    is_test = pos >= size - n_test
    return r[~is_test].reset_index(drop=True), r[is_test].reset_index(drop=True)


def load_split(name: str = "ml-latest-small") -> tuple[pd.DataFrame, pd.DataFrame]:
    return temporal_split(load_ratings(name))
