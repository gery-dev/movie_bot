"""Офлайн-сравнение моделей на MovieLens.

    python -m recsys.evaluate ml-latest-small

Схема: у каждого пользователя последние 20% оценок — тест. Параметры моделей подбираются
на последних 20% обучающей части (валидация), затем модель переобучается на всей обучающей
части и проверяется на тесте. Понравившийся фильм — оценка ≥ 4. Таблица результатов
сохраняется в reports/metrics_<набор>.md.
"""
import argparse
import itertools
import time

import numpy as np
import pandas as pd

from data_pipeline.movielens import load_links
from db.database import ROOT, connect
from data_pipeline.movielens import load_ratings
from recsys.data import RELEVANT, temporal_split
from recsys.metrics import ndcg_at_k, precision_at_k, recall_at_k, rmse
from recsys.models import ALS, SVD, Catalog, Content, Hybrid, Popularity

K = 10
SEED = 42

GRIDS = {
    "popularity_weighted": {"min_votes": [10_000, 50_000, 200_000]},
    "content": {"emb_weight": [0.0, 0.25, 0.5, 0.75, 1.0]},
    "svd": {"n_factors": [50, 100], "n_epochs": [20, 40], "reg_all": [0.02, 0.1]},
    "als": {"factors": [32, 64, 128], "regularization": [0.01, 0.1], "alpha": [1.0, 5.0]},
    "hybrid": {"cf_weight": [0.25, 0.5, 0.75, 0.9]},
}


EVAL_USERS: set | None = None  # если задано — метрики считаются только по этим пользователям (большой набор)


def evaluate(model, catalog: Catalog, train: pd.DataFrame, test: pd.DataFrame, k: int = K) -> dict:
    if EVAL_USERS is not None:
        test = test[test.userId.isin(EVAL_USERS)]
        train = train[train.userId.isin(EVAL_USERS)]
    seen = train.groupby("userId").tconst.apply(catalog.positions)
    relevant = test[test.rating >= RELEVANT].groupby("userId").tconst.apply(set)
    p, r, n, shown = [], [], [], set()
    for user, rel in relevant.items():
        s = np.array(model.scores(user), float, copy=True)
        if user in seen.index:
            s[seen[user]] = -np.inf  # уже оценённое не рекомендуем
        top = np.argpartition(-s, k)[:k]
        top = top[np.argsort(-s[top])]
        rec = [catalog.ids[i] for i in top]
        shown.update(rec)
        p.append(precision_at_k(rec, rel, k))
        r.append(recall_at_k(rec, rel, k))
        n.append(ndcg_at_k(rec, rel, k))
    return {"P@10": np.mean(p), "R@10": np.mean(r), "NDCG@10": np.mean(n), "покрытие": len(shown) / len(catalog)}


def tune(name: str, make, catalog: Catalog, fit_part: pd.DataFrame, valid: pd.DataFrame):
    """Перебор сетки параметров; лучший вариант — по NDCG@10 на валидации."""
    grid = GRIDS[name]
    best = None
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid, values))
        score = evaluate(make(**params).fit(fit_part), catalog, fit_part, valid)["NDCG@10"]
        print(f"    {name} {params}: NDCG@10 = {score:.4f}")
        if best is None or score > best[1]:
            best = (params, score)
    return best[0]


def main(dataset: str, sample_users: int | None = None, eval_users: int | None = None) -> None:
    """sample_users — обучать на случайной выборке пользователей; eval_users — считать метрики на части из них.
    Нужно для MovieLens 32M: весь набор (200 тыс. пользователей) на ноутбуке считался бы много часов."""
    global EVAL_USERS
    t0 = time.time()
    ratings = load_ratings(dataset)
    rng = np.random.default_rng(SEED)
    if sample_users:
        users = rng.choice(ratings.userId.unique(), sample_users, replace=False)
        ratings = ratings[ratings.userId.isin(set(users))]
    if eval_users:
        EVAL_USERS = set(rng.choice(ratings.userId.unique(), eval_users, replace=False))
    train, test = temporal_split(ratings)
    fit_part, valid = temporal_split(train)
    with connect() as conn:
        known = {row[0] for row in conn.execute("SELECT tconst FROM movies")}
    catalog = Catalog(sorted(set(load_links(dataset).tconst) & known))
    print(f"{dataset}: каталог {len(catalog)} фильмов, обучение {len(train)}, тест {len(test)} оценок, "
          f"пользователей {train.userId.nunique()}")

    print("Подбор параметров на валидации:")
    content_cache = Content(catalog)  # признаки считаются один раз, меняется только вес

    def content(emb_weight):
        content_cache.emb_weight = emb_weight
        return content_cache

    best = {
        "popularity_weighted": tune("popularity_weighted", lambda **p: Popularity(catalog, "imdb_weighted", **p),
                                    catalog, fit_part, valid),
        "content": tune("content", content, catalog, fit_part, valid),
        "svd": tune("svd", lambda **p: SVD(catalog, **p), catalog, fit_part, valid),
        "als": tune("als", lambda **p: ALS(catalog, **p), catalog, fit_part, valid),
    }
    # для гибрида берём ту коллаборативную модель, что лучше на валидации
    content_valid = content(**best["content"]).fit(fit_part)
    cf_scores = {}
    for name, cls in [("svd", SVD), ("als", ALS)]:
        cf_scores[name] = cls(catalog, **best[name]).fit(fit_part)
    cf_name = max(cf_scores, key=lambda n: evaluate(cf_scores[n], catalog, fit_part, valid)["NDCG@10"])
    best["hybrid"] = tune("hybrid", lambda **p: Hybrid(cf_scores[cf_name], content_valid, **p), catalog, fit_part, valid)

    print("Проверка на тесте (обучение на всей обучающей части):")
    content_full = Content(catalog, **best["content"]).fit(train)
    svd = SVD(catalog, **best["svd"]).fit(train)
    als = ALS(catalog, **best["als"]).fit(train)
    models = {
        "Популярность: рейтинг IMDb": Popularity(catalog, "imdb").fit(train),
        "Популярность: взвешенный рейтинг IMDb": Popularity(catalog, "imdb_weighted", **best["popularity_weighted"]).fit(train),
        "Популярность: MovieLens": Popularity(catalog, "ml").fit(train),
        "Контентная": content_full,
        "SVD": svd,
        "ALS": als,
        f"Гибрид ({cf_name.upper()} + контентная)": Hybrid(svd if cf_name == "svd" else als, content_full, **best["hybrid"]),
    }
    rows = []
    for name, model in models.items():
        row = {"Модель": name, **evaluate(model, catalog, train, test)}
        row["RMSE"] = rmse(model.predict(test), test.rating) if isinstance(model, SVD) else np.nan
        rows.append(row)
        print(f"  {name}: " + ", ".join(f"{k} {v:.4f}" for k, v in row.items() if k != "Модель" and pd.notna(v)))

    table = pd.DataFrame(rows)
    sample = (f"Случайная выборка (seed {SEED}): обучение на {train.userId.nunique()} пользователях"
              + (f", метрики — по {len(EVAL_USERS)} из них" if EVAL_USERS else "") + ". ") if sample_users else ""
    save_report(dataset, table, best, cf_name, catalog, train, test, time.time() - t0, sample)


def save_report(dataset, table, best, cf_name, catalog, train, test, seconds, sample: str = "") -> None:
    fmt = table.copy()
    for col in ["P@10", "R@10", "NDCG@10", "RMSE"]:
        fmt[col] = fmt[col].map(lambda v: "—" if pd.isna(v) else f"{v:.4f}")
    fmt["покрытие"] = table["покрытие"].map(lambda v: f"{v:.1%}")
    header = "| " + " | ".join(fmt.columns) + " |\n|" + "---|" * len(fmt.columns) + "\n"
    body = "".join("| " + " | ".join(str(v) for v in row) + " |\n" for row in fmt.itertuples(index=False))
    params = "\n".join(f"- {name}: {p}" for name, p in best.items())
    text = (
        f"# Офлайн-сравнение моделей: {dataset}\n\n"
        f"Каталог {len(catalog)} фильмов, пользователей {train.userId.nunique()}, "
        f"оценок в обучении {len(train)}, в тесте {len(test)}. "
        f"Тест — последние 20% оценок каждого пользователя, понравившийся фильм — оценка ≥ {RELEVANT:g}, K = {K}. "
        f"Покрытие — доля каталога, попавшая хоть в одну выдачу. RMSE — только для модели, предсказывающей оценку. "
        f"{sample}\n\n"
        f"{header}{body}\n"
        f"Параметры, подобранные на валидации (гибрид построен на {cf_name.upper()}):\n\n{params}\n\n"
        f"Время расчёта: {seconds / 60:.1f} мин. Команда: `python -m recsys.evaluate {dataset}"
        + (f" --sample-users {train.userId.nunique()}" if sample else "")
        + (f" --eval-users {len(EVAL_USERS)}" if EVAL_USERS else "") + "`.\n"
    )
    path = ROOT / "reports" / f"metrics_{dataset}.md"
    path.write_text(text, encoding="utf-8")
    print(f"Сохранено в {path.relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="ml-latest-small", choices=["ml-latest-small", "ml-32m"])
    parser.add_argument("--sample-users", type=int)
    parser.add_argument("--eval-users", type=int)
    args = parser.parse_args()
    main(args.dataset, args.sample_users, args.eval_users)
