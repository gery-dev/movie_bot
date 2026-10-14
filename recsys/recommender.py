"""Итоговая оценка фильма для пользователя бота, в два шага:

    1) кандидаты = лучшие CANDIDATES фильмов по  A · долгосрочный профиль + C · число любимых людей в фильме;
    2) кандидаты переставляются по  ранг по шагу 1 + B · профиль на сегодня + C · число любимых людей.

Два шага нужны из-за масштаба: у долгосрочной оценки длинный хвост (у лучших фильмов до 14 стандартных
отклонений), и при простом сложении пожелания на сегодня почти не меняли бы верх выдачи. Ранги внутри
кандидатов ограничены по масштабу, поэтому B действительно задаёт вес пожеланий.

Долгосрочная часть — гибрид из этапа 2: ALS (обучена на MovieLens, пользователь бота
«встраивается» в неё по своим оценкам) и профиль по признакам. Пока оценок мало, работает
только профиль; если событий нет совсем — взвешенный рейтинг IMDb.
Коэффициенты подобраны на пользователях MovieLens: python -m recsys.tune.
"""
import numpy as np
import pandas as pd
from scipy import sparse

from recsys.favorites import favorite_people
from recsys.models import Catalog, _unit
from recsys.profile import Profiler, excluded_films, user_events
from recsys.today import load_daily, today_filter, today_scores

# Подбор — recsys/tune.py, результаты — reports/profile_tuning.md.
# B: на модели «желание = частый жанр теста» метрика растёт и выходит на плато; берём наименьший B,
#    дающий не меньше 99% лучшего результата — 0,5. Больше — пожелание вытесняет вкус без выигрыша.
# C: на MovieLens лучший C = 0: там любимые выведены из тех же высоких оценок, и бонус только сильнее
#    толкает уже учтённых людей (C = 0,1 стоит ~5% NDCG). В боте любимых добавляют вручную — это новая
#    информация, поэтому оставлен небольшой C = 0,1. Главное место любимых — отдельный список их
#    непросмотренных фильмов (recsys/favorites.py). Перепроверить на живых пользователях (этап 6).
A, B, C = 1.0, 0.5, 0.1
CANDIDATES = 300
CF_WEIGHT = 0.5            # доля ALS в долгосрочной части (этап 2)
MIN_RATINGS_FOR_CF = 5
POPULARITY_MIN_VOTES = 200_000  # поправка на число голосов во взвешенном рейтинге (этап 2)


def rerank(base: np.ndarray, today: np.ndarray, b: float, fav: np.ndarray, c: float,
           n: int = CANDIDATES) -> np.ndarray:
    """Шаг 2: взять n лучших по base (-inf — исключённые) и переставить их с учётом пожеланий на сегодня
    и любимых людей. Остальные фильмы получают -inf."""
    finite = np.flatnonzero(np.isfinite(base))
    cand = finite[np.argsort(-base[finite])[:n]]
    score = np.full(len(base), -np.inf)
    if len(cand) == 0:
        return score
    ranks = np.empty(len(cand))
    ranks[np.argsort(-base[cand])] = np.arange(len(cand))
    s = _unit(-ranks) + c * fav[cand]
    if b and np.ptp(today[cand]) > 0:
        s += b * _unit(today[cand])
    score[cand] = s
    return score


class Recommender:
    def __init__(self, conn, als=None, ml_ratings: pd.DataFrame | None = None):
        """als — обученная модель recsys.models.ALS на каталоге всех фильмов БД (необязательно)."""
        self.movies = pd.read_sql(
            "SELECT tconst, title, title_ru, year, runtime, genres, language, rating, votes FROM movies ORDER BY tconst",
            conn,
        )
        self.catalog = Catalog(self.movies.tconst.tolist())
        self.profiler = Profiler(self.catalog.ids, conn=conn)
        self.als = als
        v, r = self.movies.votes.to_numpy(float), self.movies.rating.to_numpy(float)
        self.popularity = (v * r + POPULARITY_MIN_VOTES * r.mean()) / (v + POPULARITY_MIN_VOTES)
        people = pd.read_sql("SELECT tconst, nconst FROM movie_people", conn)
        self.people = people.assign(pos=self.catalog.positions(people.tconst))
        self.cf_known = np.ones(len(self.catalog), bool)
        if als is not None and ml_ratings is not None:
            known = np.zeros(len(self.catalog), bool)
            known[self.catalog.positions(ml_ratings.tconst.unique())] = True
            self.cf_known = known  # фильмы без оценок MovieLens ALS не знает

    def long_term(self, conn, user_id: int, events=None) -> np.ndarray:
        events = user_events(conn, user_id) if events is None else events
        if not events:
            return _unit(self.popularity)
        profile = _unit(self.profiler.scores(self.profiler.vector(events)))
        ratings = conn.execute("SELECT tconst, rating FROM ratings WHERE user_id = ?", (user_id,)).fetchall()
        if self.als is None or len(ratings) < MIN_RATINGS_FOR_CF:
            return profile
        row = sparse.csr_matrix(
            ([float(r) for _, r in ratings], ([0] * len(ratings), self.catalog.positions([t for t, _ in ratings]))),
            shape=(1, len(self.catalog)),
        )
        factors = self.als.model.recalculate_user(0, row)
        cf = np.where(self.cf_known, self.als.model.item_factors @ np.asarray(factors).ravel(), -np.inf)
        return CF_WEIGHT * _unit(cf) + (1 - CF_WEIGHT) * profile

    def favorites_bonus(self, conn, user_id: int) -> np.ndarray:
        fav = favorite_people(conn, user_id)
        bonus = np.zeros(len(self.catalog))
        if fav:
            hits = self.people[self.people.nconst.isin(fav)]
            np.add.at(bonus, hits.pos.to_numpy(), 1.0)
        return bonus

    def recommend(self, conn, user_id: int, k: int = 5, answers: dict | None = None) -> pd.DataFrame:
        """Топ-k фильмов с частями оценки. answers — опрос на сегодня (по умолчанию из БД)."""
        answers = load_daily(conn, user_id) if answers is None else answers
        long_term = self.long_term(conn, user_id)
        today = today_scores(self.movies, answers) if answers else np.zeros(len(self.catalog))
        fav = self.favorites_bonus(conn, user_id)

        ok = np.ones(len(self.catalog), bool)
        if answers:
            ok &= today_filter(self.movies, answers)
        skip = excluded_films(conn, user_id)
        if skip:
            ok[self.catalog.positions([t for t in skip if t in self.catalog.index])] = False
        base = np.where(ok, A * long_term + C * fav, -np.inf)
        score = rerank(base, today, B, fav, C)

        top = np.argsort(-score)[:k]
        top = top[np.isfinite(score[top])]
        out = self.movies.iloc[top][["tconst", "title_ru", "title", "year", "runtime", "rating"]].copy()
        out["score"], out["long_term"], out["today"], out["favorites"] = score[top], long_term[top], today[top], fav[top]
        return out.reset_index(drop=True)
