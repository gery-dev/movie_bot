"""Подбор параметров профиля на смоделированных пользователях MovieLens.

    python -m recsys.tune

1. Затухание старых оценок (период полураспада) и нейтральная оценка (3 или средняя пользователя) —
   по NDCG@10 профиля на валидации.
2. Вес любимых людей C: любимыми считаем тех, чьи ≥ 3 фильма пользователь оценил на 4+ в обучении
   (то же правило, что и автопредложение в боте).
3. Вес профиля на сегодня B: у пользователя «желание на сегодня» — самый частый жанр среди
   понравившихся ему фильмов теста. Цель — среднее NDCG@10 по всем понравившимся фильмам и по
   понравившимся фильмам этого жанра: и вкус не потерять, и пожелание выполнить.
4. Разнообразие калибровки: 200 случайных пользователей, по 10 карточек.
Результаты — в reports/profile_tuning.md.
"""
import time

import numpy as np
import pandas as pd

from data_pipeline.movielens import load_links
from db.database import ROOT, connect
from recsys import calibration
from recsys.data import RELEVANT, load_split, temporal_split
from recsys.evaluate import evaluate
from recsys.favorites import SUGGEST_MIN_FILMS, SUGGEST_MIN_RATING
from recsys.models import ALS, Catalog, _unit
from recsys.profile import Profiler, rating_signal
from recsys.recommender import rerank

DATASET = "ml-latest-small"
HALF_LIVES = [None, 1460, 730, 365, 180, 90]
CENTERS = ["3", "user_mean"]
C_GRID = [0, 0.1, 0.25, 0.5, 1.0, 2.0]
B_GRID = [0, 0.02, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0]
ALS_PARAMS = dict(factors=32, regularization=0.1, alpha=1.0)  # лучшие на этапе 2
CF_WEIGHT = 0.5


class ProfileModel:
    """Профиль из recsys.profile на оценках MovieLens; время «сейчас» — последняя оценка пользователя."""

    def __init__(self, profiler: Profiler, train: pd.DataFrame, half_life, center: str):
        self.profiler, self.half_life = profiler, half_life
        mean = train.groupby("userId").rating.transform("mean")
        train = train.assign(signal=[rating_signal(r, m if center == "user_mean" else 3.0)
                                     for r, m in zip(train.rating, mean)])
        self.events = {u: list(zip(g.tconst, g.signal, g.timestamp)) for u, g in train.groupby("userId")}
        self.now = train.groupby("userId").timestamp.max().to_dict()
        self._cache = {}

    def scores(self, user_id) -> np.ndarray:
        if user_id not in self._cache:
            v = self.profiler.vector(self.events.get(user_id, []), now=self.now.get(user_id, 0),
                                     half_life_days=self.half_life)
            self._cache[user_id] = self.profiler.scores(v)
        return self._cache[user_id]


class Combined:
    """Как в recsys/recommender.py: кандидаты по A·долгосрочная + C·любимые, затем перестановка с B·сегодня.
    seen — позиции уже оценённых фильмов: их убираем до отбора кандидатов, как в боте."""

    def __init__(self, long_term, seen: dict, today=None, fav=None, b=0.0, c=0.0):
        self.long_term, self.today, self.fav, self.b, self.c = long_term, today or {}, fav or {}, b, c
        self.seen = seen

    def scores(self, user_id) -> np.ndarray:
        s = _unit(self.long_term.scores(user_id))
        fav = self.fav.get(user_id, np.zeros(len(s)))
        s = s + self.c * fav
        s[self.seen.get(user_id, [])] = -np.inf
        today = self.today.get(user_id, np.zeros(len(s)))
        return rerank(s, today, self.b, fav, self.c)


class Hybrid:
    def __init__(self, als, profile):
        self.als, self.profile = als, profile

    def scores(self, user_id) -> np.ndarray:
        return CF_WEIGHT * _unit(self.als.scores(user_id)) + (1 - CF_WEIGHT) * _unit(self.profile.scores(user_id))


def favorites_bonus(catalog: Catalog, train: pd.DataFrame) -> dict:
    conn = connect()
    people = pd.read_sql("SELECT tconst, nconst FROM movie_people", conn)
    conn.close()
    people = people[people.tconst.isin(catalog.index)]
    liked = train[train.rating >= SUGGEST_MIN_RATING].merge(people, on="tconst")
    counts = liked.groupby(["userId", "nconst"]).tconst.nunique()
    fav = counts[counts >= SUGGEST_MIN_FILMS].reset_index()
    pos = people.assign(pos=catalog.positions(people.tconst))
    bonus = {}
    for user, g in fav.groupby("userId"):
        b = np.zeros(len(catalog))
        np.add.at(b, pos[pos.nconst.isin(set(g.nconst))].pos.to_numpy(), 1.0)
        bonus[user] = b
    return bonus


def genre_wishes(catalog: Catalog, test: pd.DataFrame):
    """Для каждого пользователя: жанр-желание, индикатор этого жанра по каталогу, понравившиеся фильмы жанра."""
    conn = connect()
    genres = pd.read_sql("SELECT tconst, genre FROM movie_genres", conn)
    conn.close()
    liked = test[test.rating >= RELEVANT].merge(genres, on="tconst")
    by_genre = genres[genres.tconst.isin(catalog.index)].groupby("genre").tconst.apply(catalog.positions)
    wish, today, wish_test = {}, {}, []
    for user, g in liked.groupby("userId"):
        top = g.genre.value_counts()
        if top.iloc[0] < 2:
            continue
        wish[user] = top.index[0]
        ind = np.zeros(len(catalog))
        ind[by_genre[wish[user]]] = 1.0
        today[user] = ind
        wish_test.append(test[(test.userId == user) & test.tconst.isin(set(g[g.genre == wish[user]].tconst))])
    return wish, today, pd.concat(wish_test)


def calibration_check(n_users: int = 200) -> dict:
    conn = connect()
    movies = pd.read_sql("SELECT tconst, year, genres, country, rating, votes FROM movies", conn).set_index("tconst")
    conn.close()
    pool = calibration.calibration_pool(movies)
    all_genres = sorted(set().union(*pool.genre_set))
    rng = np.random.default_rng(42)
    n_genres, n_decades = [], []
    for _ in range(n_users):
        start = set(rng.choice(all_genres, 2, replace=False))
        shown, not_seen = [], 0
        for _ in range(calibration.CALIBRATION_SIZE):
            shown.append(calibration.next_film(pool, shown, not_seen, start, rng))
            not_seen += rng.random() < 0.4  # примерно 40% фильмов пользователь не видел
        n_genres.append(len(set().union(*pool.loc[shown].genre_set)))
        n_decades.append(pool.loc[shown].decade.nunique())
    n_genres, n_decades = np.array(n_genres), np.array(n_decades)
    return {"pool": len(pool), "genres_mean": n_genres.mean(), "decades_mean": n_decades.mean(),
            "ok_share": ((n_genres >= 6) & (n_decades >= 3)).mean(), "genres_min": n_genres.min(),
            "decades_min": n_decades.min()}


def main() -> None:
    t0 = time.time()
    train, test = load_split(DATASET)
    fit_part, valid = temporal_split(train)
    with connect() as conn:
        known = {r[0] for r in conn.execute("SELECT tconst FROM movies WHERE kind = 'movie'")}
    catalog = Catalog(sorted(set(load_links(DATASET).tconst) & known))
    profiler = Profiler(catalog.ids)
    lines = []

    print("1. Затухание и нейтральная оценка (валидация):")
    results = {}
    for hl in HALF_LIVES:
        for center in CENTERS:
            ndcg = evaluate(ProfileModel(profiler, fit_part, hl, center), catalog, fit_part, valid)["NDCG@10"]
            results[(hl, center)] = ndcg
            print(f"   полураспад {hl or '∞'} дн., центр {center}: NDCG@10 {ndcg:.4f}")
    best_hl, best_center = max(results, key=results.get)
    lines.append("## 1. Затухание старых оценок и нейтральная оценка\n\nNDCG@10 профиля на валидации:\n")
    lines.append("| Полураспад, дней | центр 3 | центр — средняя пользователя |\n|---|---|---|")
    for hl in HALF_LIVES:
        lines.append(f"| {hl or 'без затухания'} | {results[(hl, '3')]:.4f} | {results[(hl, 'user_mean')]:.4f} |")
    lines.append(f"\nЛучшее: полураспад {best_hl or 'без затухания'}, центр {best_center}.\n")

    print("2. Долгосрочная часть на тесте:")
    profile = ProfileModel(profiler, train, best_hl, best_center)
    als = ALS(catalog, **ALS_PARAMS).fit(train)
    long_term = Hybrid(als, profile)
    rows = {"Профиль (признаки)": evaluate(profile, catalog, train, test),
            "ALS": evaluate(als, catalog, train, test),
            "Гибрид ALS + профиль": evaluate(long_term, catalog, train, test)}
    lines.append("## 2. Долгосрочная часть на тесте\n\n| Модель | P@10 | R@10 | NDCG@10 | покрытие |\n|---|---|---|---|---|")
    for name, r in rows.items():
        print(f"   {name}: NDCG@10 {r['NDCG@10']:.4f}")
        lines.append(f"| {name} | {r['P@10']:.4f} | {r['R@10']:.4f} | {r['NDCG@10']:.4f} | {r['покрытие']:.1%} |")

    print("3. Вес любимых людей C:")
    fav_fit, fav_full = favorites_bonus(catalog, fit_part), favorites_bonus(catalog, train)
    seen_fit = fit_part.groupby("userId").tconst.apply(catalog.positions).to_dict()
    seen = train.groupby("userId").tconst.apply(catalog.positions).to_dict()
    long_fit = Hybrid(ALS(catalog, **ALS_PARAMS).fit(fit_part), ProfileModel(profiler, fit_part, best_hl, best_center))
    c_valid = {c: evaluate(Combined(long_fit, seen_fit, fav=fav_fit, c=c), catalog, fit_part, valid)["NDCG@10"]
               for c in C_GRID}
    best_c = max(c_valid, key=c_valid.get)
    c_test = {c: evaluate(Combined(long_term, seen, fav=fav_full, c=c), catalog, train, test)["NDCG@10"]
              for c in [0, best_c]}
    print(f"   валидация { {c: round(v, 4) for c, v in c_valid.items()} }, выбран C = {best_c}, тест {c_test}")
    lines.append(f"\n## 3. Вес любимых актёров и режиссёров C\n\nЛюбимые есть у {len(fav_full)} из "
                 f"{train.userId.nunique()} пользователей.\n\n| C | NDCG@10 валидация |\n|---|---|")
    lines += [f"| {c} | {v:.4f} |" for c, v in c_valid.items()]
    lines.append(f"\nЛучший по данным C = {best_c} (на тесте NDCG@10 {c_test[best_c]:.4f}). Здесь любимые выведены из "
                 f"тех же высоких оценок, что и профиль, поэтому бонус лишь сильнее толкает уже учтённых людей. "
                 f"В боте любимых добавляют вручную — это новая информация, поэтому там взят небольшой C "
                 f"(см. recsys/recommender.py); перепроверить на живых пользователях (этап 6).\n")

    print("4. Вес профиля на сегодня B:")
    wish, today, wish_test = genre_wishes(catalog, test)
    users = set(wish)
    t_all, t_wish = test[test.userId.isin(users)], wish_test
    lines.append(f"## 4. Вес профиля на сегодня B\n\nМодель: у {len(users)} пользователей желание на сегодня — их "
                 f"самый частый жанр среди понравившихся фильмов теста.\n\n"
                 f"| B | NDCG@10 по всем понравившимся | NDCG@10 по жанру-желанию | среднее | доля жанра в топ-10 |\n|---|---|---|---|---|")
    b_res = {}
    for b in B_GRID:
        model = Combined(long_term, seen, today=today, b=b, fav=fav_full, c=best_c)
        n_all = evaluate(model, catalog, train, t_all)["NDCG@10"]
        n_wish = evaluate(model, catalog, train, t_wish)["NDCG@10"]
        share = np.mean([today[u][np.argsort(-model.scores(u))[:10]].mean() for u in sorted(users)[:200]])
        b_res[b] = (n_all + n_wish) / 2
        print(f"   B = {b}: все {n_all:.4f}, желание {n_wish:.4f}, доля жанра {share:.0%}")
        lines.append(f"| {b} | {n_all:.4f} | {n_wish:.4f} | {b_res[b]:.4f} | {share:.0%} |")
    # после некоторого B метрика выходит на плато, а пожелание уже вытесняет вкус — берём наименьший B,
    # дающий не меньше 99% лучшего среднего
    best_b = min(b for b, v in b_res.items() if v >= 0.99 * max(b_res.values()))
    lines.append(f"\nВыбран B = {best_b}: наименьший, при котором среднее двух NDCG не меньше 99% лучшего. "
                 f"Это упрощённая модель: настоящий опрос поднимает несколько жанров и эпоху, поэтому B стоит "
                 f"перепроверить на живых пользователях (этап 6).\n")

    print("5. Разнообразие калибровки:")
    cal = calibration_check()
    print(f"   {cal}")
    lines.append(f"## 5. Разнообразие калибровки\n\n200 смоделированных пользователей (2 случайных стартовых жанра, "
                 f"40% ответов «не смотрел»), по {calibration.CALIBRATION_SIZE} карточек из пула {cal['pool']} фильмов "
                 f"(с {calibration.MIN_YEAR} года, голосов ≥ {calibration.MIN_VOTES}, рейтинг ≥ {calibration.MIN_RATING}).\n\n"
                 f"- жанров за калибровку: в среднем {cal['genres_mean']:.1f}, минимум {cal['genres_min']};\n"
                 f"- десятилетий: в среднем {cal['decades_mean']:.1f}, минимум {cal['decades_min']};\n"
                 f"- условие «≥ 6 жанров и все 3 десятилетия (2000-е, 2010-е, 2020-е)» выполнено у {cal['ok_share']:.0%} пользователей.\n")

    header = (f"# Подбор параметров профиля ({DATASET})\n\nТест — последние 20% оценок каждого пользователя, "
              f"валидация — последние 20% обучающей части, понравившийся фильм — оценка ≥ {RELEVANT:g}, K = 10. "
              f"Команда: `python -m recsys.tune`, время {(time.time() - t0) / 60:.1f} мин.\n\n"
              f"**Лучшее по данным:** полураспад {best_hl or 'без затухания'}, центр {best_center}, "
              f"B = {best_b}, C = {best_c}. Какие значения взяты в бот и почему — в recsys/recommender.py "
              f"и recsys/profile.py.\n")
    path = ROOT / "reports" / "profile_tuning.md"
    path.write_text(header + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"Сохранено в {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
