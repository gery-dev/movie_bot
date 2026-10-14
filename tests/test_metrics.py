import math

import pandas as pd
import pytest

from recsys.data import temporal_split
from recsys.metrics import ndcg_at_k, precision_at_k, recall_at_k, rmse

RECOMMENDED = ["a", "b", "c", "d", "e"]
RELEVANT = {"b", "e", "x"}


def test_precision():
    assert precision_at_k(RECOMMENDED, RELEVANT, 5) == pytest.approx(2 / 5)
    assert precision_at_k(RECOMMENDED, RELEVANT, 1) == 0


def test_recall():
    assert recall_at_k(RECOMMENDED, RELEVANT, 5) == pytest.approx(2 / 3)
    assert recall_at_k(RECOMMENDED, set(), 5) == 0


def test_ndcg():
    # попадания на позициях 2 и 5; идеал — три попадания на позициях 1–3
    dcg = 1 / math.log2(3) + 1 / math.log2(6)
    ideal = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(RECOMMENDED, RELEVANT, 5) == pytest.approx(dcg / ideal)
    assert ndcg_at_k(["b", "e", "x"], RELEVANT, 3) == pytest.approx(1)


def test_rmse():
    assert rmse([3, 4], [3, 2]) == pytest.approx(math.sqrt(2))


def test_temporal_split_takes_latest_per_user():
    ratings = pd.DataFrame({
        "userId": [1] * 10 + [2] * 3,
        "movieId": list(range(10)) + [0, 1, 2],
        "rating": [4.0] * 13,
        "timestamp": list(range(10, 0, -1)) + [5, 6, 7],
    })
    train, test = temporal_split(ratings)
    # у пользователя 1 самые поздние оценки — у фильмов 0 и 1; у пользователя 2 — 1 из 3 (не меньше одной)
    assert set(test[test.userId == 1].movieId) == {0, 1}
    assert list(test[test.userId == 2].movieId) == [2]
    assert len(train) + len(test) == len(ratings)
