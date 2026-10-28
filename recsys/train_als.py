"""ALS для бота: обучается на Mac, на сервер уходит только файл с векторами фильмов (~2,5 МБ).

    python -m recsys.train_als            # ml-32m → data/processed/als_ml-32m.npz (~1 мин)

Почему 32M и только оценки от CF_MIN_RATING. В implicit оценка — сила интереса, а не знак: «1★» тоже
тянула вкус к похожим фильмам. Проверка 28.10: 1000 пользователей 32M с оценками 2019+ (в обучение
не входили) через настоящий рекомендатель бота, NDCG@10 знакомство / + калибровка / импорт:
    small, все оценки (было)  0,051 / 0,041 / 0,097
    small, только ≥4          0,049 / 0,042 / 0,096
    32M, все оценки           0,059 / 0,045 / 0,112
    32M, только ≥4            0,058 / 0,049 / 0,121   (± 0,004 / 0,004 / 0,006 к «было»)
    32M, со знаком (оценка−3) 0,057 / 0,049 / 0,122 — то же, что «только ≥4», но сложнее.
"""
import time

from data_pipeline.movielens import load_ratings
from db.database import ROOT, connect
from recsys.models import ALS, Catalog
from recsys.recommender import CF_MIN_RATING

DATASET = "ml-32m"
PARAMS = dict(factors=32, regularization=0.01, alpha=1.0)  # лучшие на 32M (reports/metrics_ml-32m.md)
ALS_FILE = ROOT / "data" / "processed" / f"als_{DATASET}.npz"


def main() -> None:
    t0 = time.time()
    with connect() as conn:
        ids = [r[0] for r in conn.execute("SELECT tconst FROM movies ORDER BY tconst")]
    ratings = load_ratings(DATASET)
    liked = ratings[ratings.rating >= CF_MIN_RATING]
    als = ALS(Catalog(ids), **PARAMS).fit(liked)
    als.save(ALS_FILE)
    print(f"{ALS_FILE.name}: {als.known.sum()} фильмов, {len(liked)} оценок от {CF_MIN_RATING}, "
          f"{ALS_FILE.stat().st_size / 1e6:.1f} МБ, {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main()
