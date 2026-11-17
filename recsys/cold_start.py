"""Холодный старт (этап 6): насколько хорошо бот советует новому пользователю в разных сценариях.

    python -m recsys.cold_start

Пользователи MovieLens (small), которых ALS при обучении не видела, проходят через настоящий
рекомендатель бота (профиль + ALS с встраиванием по оценкам + MMR):
  1. без данных — сразу /recommend;
  2. знакомство — 2 любимых фильма (самые высоко оценённые в обучающей части), оценка 5;
  3. знакомство + калибровка — 10 карточек из пула калибровки; фильм, который человек оценивал
     в обучающей части, получает его оценку, остальные — «не смотрел»;
  4. импорт Letterboxd — все оценки обучающей части.
Качество — по понравившимся (оценка ≥ 4) фильмам тестовой части, K = 10. Результат — reports/cold_start.md.
"""
import sqlite3
import time

import numpy as np
import pandas as pd

from data_pipeline.movielens import load_ratings
from db.database import ROOT, connect
from recsys import calibration
from recsys.data import RELEVANT, temporal_split
from recsys.metrics import ndcg_at_k, precision_at_k, recall_at_k
from recsys.models import ALS, Catalog
from recsys.profile import record_rating, record_reaction
from recsys.recommender import CF_MIN_RATING, Recommender

DATASET = "ml-latest-small"
N_USERS = 150
K = 10
ALS_PARAMS = dict(factors=32, regularization=0.1, alpha=1.0)
BEFORE_FIX = ('\nДо исправления (ALS включалась с 5 оценок, без подмешивания популярности) знакомство и калибровка давали NDCG@10 0,019 и 0,031 — хуже, чем совсем без данных (0,048): профиль по 2–3 фильмам цепляется за конкретных режиссёров и актёров. Исправлено: ALS с первой оценки и доля популярности N0 / (n + N0), N0 = 5 (подбор — в комментарии в recsys/recommender.py).\n')
SCENARIOS = ["без данных", "знакомство (2 любимых фильма)", "знакомство + калибровка (10 карточек)",
             "импорт Letterboxd (все оценки)"]


def stars(r: float) -> int:
    return int(max(1, round(r)))


def main() -> None:
    t0 = time.time()
    src = connect()
    conn = sqlite3.connect(":memory:")
    src.backup(conn)
    src.close()
    conn.row_factory = sqlite3.Row

    ratings = load_ratings(DATASET)
    train, test = temporal_split(ratings)
    rng = np.random.default_rng(42)
    eval_users = rng.choice(test[test.rating >= RELEVANT].userId.unique(), N_USERS, replace=False)
    others = train[~train.userId.isin(set(eval_users))]  # ALS не видит оцениваемых пользователей

    ids = [r[0] for r in conn.execute("SELECT tconst FROM movies WHERE kind = 'movie' ORDER BY tconst")]
    als = ALS(Catalog(ids), **ALS_PARAMS).fit(others[others.rating >= CF_MIN_RATING])  # как в боте
    rec = Recommender(conn, als=als)
    movies = pd.read_sql("SELECT tconst, year, genres, country, rating, votes FROM movies", conn).set_index("tconst")
    pool = calibration.calibration_pool(movies)

    results = {s: {"P@10": [], "R@10": [], "NDCG@10": [], "оценок": []} for s in SCENARIOS}
    for n, user in enumerate(eval_users):
        tr = train[train.userId == user]
        relevant = set(test[(test.userId == user) & (test.rating >= RELEVANT)].tconst)
        known = dict(zip(tr.tconst, tr.rating))
        for s_i, scenario in enumerate(SCENARIOS):
            uid = int(user) * 10 + s_i
            count = 0
            with conn:
                if s_i == 1 or s_i == 2:  # знакомство: два самых любимых фильма
                    for t in tr.sort_values(["rating", "timestamp"], ascending=False).tconst[:2]:
                        record_rating(conn, uid, t, 5, "onboarding")
                        count += 1
                if s_i == 2:  # калибровка: 10 карточек, ответ — по известным оценкам человека
                    shown, not_seen = [], 0
                    for _ in range(calibration.CALIBRATION_SIZE):
                        t = calibration.next_film(pool, shown, not_seen, None, rng)
                        shown.append(t)
                        if t in known:
                            record_rating(conn, uid, t, stars(known[t]), "calibration")
                            count += 1
                        else:
                            record_reaction(conn, uid, t, "not_seen")
                            not_seen += 1
                if s_i == 3:
                    for t, r in known.items():
                        record_rating(conn, uid, t, stars(r), "letterboxd")
                    count = len(known)
            top = rec.recommend(conn, uid, K).tconst.tolist()
            res = results[scenario]
            res["P@10"].append(precision_at_k(top, relevant, K))
            res["R@10"].append(recall_at_k(top, relevant, K))
            res["NDCG@10"].append(ndcg_at_k(top, relevant, K))
            res["оценок"].append(count)
        if (n + 1) % 50 == 0:
            print(f"  {n + 1}/{N_USERS} пользователей")

    lines = ["| Сценарий | оценок в среднем | P@10 | R@10 | NDCG@10 |", "|---|---|---|---|---|"]
    for scenario, r in results.items():
        lines.append(f"| {scenario} | {np.mean(r['оценок']):.1f} | {np.mean(r['P@10']):.4f} | "
                     f"{np.mean(r['R@10']):.4f} | {np.mean(r['NDCG@10']):.4f} |")
    table = "\n".join(lines)
    text = (f"# Холодный старт ({DATASET})\n\n{N_USERS} случайных пользователей с понравившимися фильмами в тесте; "
            f"ALS обучена без них, их оценки попадают в модель только через бота (встраивание по оценкам). "
            f"Тест — последние 20% оценок, понравившийся фильм — оценка ≥ {RELEVANT:g}, K = {K}. "
            f"Рекомендации — настоящий рекомендатель бота (профиль, ALS, MMR). "
            f"Команда: `python -m recsys.cold_start`, время {(time.time() - t0) / 60:.1f} мин.\n\n{table}\n{BEFORE_FIX}")
    (ROOT / "reports" / "cold_start.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
