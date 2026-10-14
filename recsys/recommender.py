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
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import MultiLabelBinarizer, normalize

from recsys.explain import CF_STRONG, explain, feature_label, top_features
from recsys.favorites import favorite_people
from recsys.features import EMB_MODEL, load_embeddings
from recsys.models import Catalog, _unit
from recsys.profile import Profiler, excluded_films, user_events
from recsys.query import Parsed, describe, parse, query_scores, understood
from recsys.today import today_filter, today_scores

# Подбор — recsys/tune.py, результаты — reports/profile_tuning.md.
# B: на модели «желание = частый жанр теста» метрика растёт до плато (лучший по правилу 99% — 0,5,
#    там 100% выдачи под желание). По решению пользователя настроение должно перевешивать вкус
#    умеренно — «примерно 4 фильма из 5 под настроение». Проверено на настоящих ответах опроса
#    (30 пользователей MovieLens, настроения «весёлое», «напряжённое», «задуматься», «романтичное»):
#    B = 0,1 — 56% топ-10 под настроение, 0,2 — 73%, 0,3 — 82%, 0,5 — 92%. Берём 0,3; при нём
#    с обычной подборкой /recommend совпадает около трети фильмов.
# C: на MovieLens лучший C = 0: там любимые выведены из тех же высоких оценок, и бонус только сильнее
#    толкает уже учтённых людей (C = 0,1 стоит ~5% NDCG). В боте любимых добавляют вручную — это новая
#    информация, поэтому оставлен небольшой C = 0,1. Главное место любимых — отдельный список их
#    непросмотренных фильмов (recsys/favorites.py). Перепроверить на живых пользователях (этап 6).
A, B, C = 1.0, 0.3, 0.1
CANDIDATES = 300
CF_WEIGHT = 0.5            # доля ALS в долгосрочной части (этап 2)
MIN_RATINGS_FOR_CF = 5
POPULARITY_MIN_VOTES = 200_000  # поправка на число голосов во взвешенном рейтинге (этап 2)
# MMR (разнообразие выдачи). Подбор на 30 пользователях MovieLens, пятёрка фильмов:
#   без MMR — у 27% все 5 фильмов одного жанра, у 33% повтор режиссёра;
#   λ = 0,6 и жанры 0,3 — 0% и 7%, при этом 94% фильмов из той же лучшей десятки.
MMR_LAMBDA = 0.6        # доля «подходит» против «не похож на выбранные»
MMR_POOL = 100          # из скольких лучших выбирать разнообразную выдачу
GENRE_SIM_WEIGHT = 0.3  # доля похожести по жанрам (остальное — режиссёр, актёры, десятилетие…)
GROUP_MIN_WEIGHT = 0.5  # /together: доля «минимума по участникам» против среднего


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
        self.names = dict(conn.execute("SELECT nconst, name FROM people").fetchall())
        genres = self.movies.genres.fillna("").str.split(",").map(lambda g: [x for x in g if x])
        self.genre_matrix = normalize(MultiLabelBinarizer(sparse_output=True).fit_transform(genres).astype(float)).tocsr()
        self._embeddings = self._text_model = None
        self.cf_known = np.ones(len(self.catalog), bool)
        if als is not None and ml_ratings is not None:
            known = np.zeros(len(self.catalog), bool)
            known[self.catalog.positions(ml_ratings.tconst.unique())] = True
            self.cf_known = known  # фильмы без оценок MovieLens ALS не знает

    def long_term_parts(self, conn, user_id: int, events=None) -> dict:
        """Долгосрочная оценка и её части — нужны и для выдачи, и для объяснения."""
        events = user_events(conn, user_id) if events is None else events
        if not events:
            return {"total": _unit(self.popularity), "profile": None, "profile_score": None, "cf": None}
        profile = self.profiler.vector(events)
        profile_score = _unit(self.profiler.scores(profile))
        parts = {"total": profile_score, "profile": profile, "profile_score": profile_score, "cf": None}
        ratings = conn.execute("SELECT tconst, rating FROM ratings WHERE user_id = ?", (user_id,)).fetchall()
        if self.als is None or len(ratings) < MIN_RATINGS_FOR_CF:
            return parts
        row = sparse.csr_matrix(
            ([float(r) for _, r in ratings], ([0] * len(ratings), self.catalog.positions([t for t, _ in ratings]))),
            shape=(1, len(self.catalog)),
        )
        factors = self.als.model.recalculate_user(0, row)
        cf = _unit(np.where(self.cf_known, self.als.model.item_factors @ np.asarray(factors).ravel(), -np.inf))
        parts.update(cf=cf, total=CF_WEIGHT * cf + (1 - CF_WEIGHT) * profile_score)
        return parts

    def long_term(self, conn, user_id: int, events=None) -> np.ndarray:
        return self.long_term_parts(conn, user_id, events)["total"]

    def favorites_bonus(self, conn, user_id: int, fav: set[str] | None = None) -> np.ndarray:
        fav = favorite_people(conn, user_id) if fav is None else fav
        bonus = np.zeros(len(self.catalog))
        if fav:
            hits = self.people[self.people.nconst.isin(fav)]
            np.add.at(bonus, hits.pos.to_numpy(), 1.0)
        return bonus

    def diversify(self, score: np.ndarray, k: int, lam: float | None = None, genre_weight: float | None = None,
                  pool: int = MMR_POOL) -> np.ndarray:
        """MMR: следующий фильм — с лучшим балансом «подходит» (λ) и «не похож на уже выбранные» (1 − λ).
        Похожесть — косинус по признакам (жанры, режиссёр, актёры…), так в подборке не идут подряд
        три части одной франшизы или пять фильмов одного жанра."""
        lam = MMR_LAMBDA if lam is None else lam
        genre_weight = GENRE_SIM_WEIGHT if genre_weight is None else genre_weight
        finite = np.flatnonzero(np.isfinite(score))
        cand = finite[np.argsort(-score[finite])[:max(pool, k)]]
        if len(cand) <= 1:
            return cand
        rel = score[cand]
        rel = (rel - rel.min()) / (np.ptp(rel) or 1.0)
        X, G = self.profiler.X[cand], self.genre_matrix[cand]
        # признаки ловят франшизы и одного режиссёра, жанры отдельно — иначе частая «драма» почти не весит
        sim = (1 - genre_weight) * (X @ X.T).toarray() + genre_weight * (G @ G.T).toarray()
        chosen, max_sim = [], np.zeros(len(cand))
        free = np.ones(len(cand), bool)
        for _ in range(min(k, len(cand))):
            mmr = np.where(free, lam * rel - (1 - lam) * max_sim, -np.inf)
            i = int(np.argmax(mmr))
            chosen.append(i)
            free[i] = False
            max_sim = np.maximum(max_sim, sim[i])
        return cand[chosen]

    def recommend(self, conn, user_id: int, k: int = 5, answers: dict | None = None,
                  diverse: bool = True) -> pd.DataFrame:
        """Топ-k фильмов с объяснением. answers — опрос на сегодня (/today); без него — обычная
        подборка по вкусу (/recommend). diverse=False — без MMR (для сравнения)."""
        answers = answers or {}
        parts = self.long_term_parts(conn, user_id)
        long_term = parts["total"]
        today = today_scores(self.movies, answers) if answers else np.zeros(len(self.catalog))
        fav_people = favorite_people(conn, user_id)
        fav = self.favorites_bonus(conn, user_id, fav_people)

        ok = np.ones(len(self.catalog), bool)
        if answers:
            ok &= today_filter(self.movies, answers)
        skip = excluded_films(conn, user_id)
        if skip:
            ok[self.catalog.positions([t for t in skip if t in self.catalog.index])] = False
        base = np.where(ok, A * long_term + C * fav, -np.inf)
        score = rerank(base, today, B, fav, C)

        if diverse:
            top = self.diversify(score, k)
        else:
            top = np.argsort(-score)[:k]
            top = top[np.isfinite(score[top])]
        out = self.movies.iloc[top][["tconst", "title_ru", "title", "year", "runtime", "rating"]].copy()
        out["score"], out["long_term"], out["today"], out["favorites"] = score[top], long_term[top], today[top], fav[top]
        out["why"] = [self._why(pos, parts, fav_people, today) for pos in top]
        return out.reset_index(drop=True)

    def _why(self, pos: int, parts: dict, fav_people: set[str], today: np.ndarray) -> str:
        film_people = self.people[(self.people.pos == pos) & self.people.nconst.isin(fav_people)].nconst
        taste = []
        if parts["profile"] is not None:
            feats = top_features(parts["profile"], self.profiler.X[pos], self.profiler.names)
            taste = [lbl for f in feats if (lbl := feature_label(f, self.names))]
        cf_driven = (parts["cf"] is not None and parts["cf"][pos] - parts["profile_score"][pos] > CF_STRONG)
        return explain(taste=taste, favorites=[self.names.get(n, n) for n in dict.fromkeys(film_people)],
                       mood=bool(today[pos] > 0), cf_driven=cf_driven, popular=parts["profile"] is None)

    # --- запрос обычным текстом ---

    def search(self, conn, user_id: int, text: str, k: int = 30) -> tuple[pd.DataFrame, Parsed]:
        """Фильмы по запросу вроде «как „Интерстеллар“, но повеселее» (см. recsys/query.py)."""
        p = parse(self.movies, text)
        ref_sim = text_sim = None
        if p.reference is not None:
            i = self.catalog.index[p.reference]
            meta = np.asarray((self.profiler.X @ self.profiler.X[i].T).todense()).ravel()
            ref_sim = 0.5 * (self.embeddings @ self.embeddings[i]) + 0.5 * meta
        elif not understood(p):  # ни образца, ни пожеланий — ищем по смыслу всего текста
            text_sim = self.embeddings @ self.text_model().encode(text, normalize_embeddings=True)
        score = query_scores(p, text_sim, ref_sim, self.movies, self.long_term(conn, user_id),
                             datetime.now().year)
        skip = [t for t in excluded_films(conn, user_id) if t in self.catalog.index]
        if p.reference:
            skip.append(p.reference)
        if skip:
            score[self.catalog.positions(skip)] = -np.inf
        top = self.diversify(score, k)
        out = self.movies.iloc[top][["tconst", "title_ru", "title", "year"]].copy()
        out["why"] = [describe(p, self.movies, t) for t in out.tconst]
        return out.reset_index(drop=True), p

    @property
    def embeddings(self) -> np.ndarray:
        if self._embeddings is None:
            self._embeddings = load_embeddings(self.catalog.ids)
        return self._embeddings

    def text_model(self):
        """Модель эмбеддингов загружается при первом запросе без фильма-образца (~0,5 ГБ)."""
        if self._text_model is None:
            from sentence_transformers import SentenceTransformer
            self._text_model = SentenceTransformer(EMB_MODEL)
        return self._text_model

    # --- кино на компанию (/together) ---

    def recommend_group(self, conn, members: list[int], k: int = 20) -> pd.DataFrame:
        """Фильмы для нескольких человек: у каждого вкус переводим в ранги (чтобы ни чей хвост любимых
        фильмов не перевешивал), затем смешиваем среднее и минимум — фильм не должен быть плохим ни для кого."""
        ranks = []
        for m in members:
            lt = self.long_term(conn, m)
            r = np.argsort(np.argsort(np.nan_to_num(lt, neginf=-1e9))).astype(float)
            ranks.append(r / (len(r) - 1))  # 0 — хуже всего, 1 — лучше всего для этого человека
        R = np.vstack(ranks)
        score = (1 - GROUP_MIN_WEIGHT) * R.mean(axis=0) + GROUP_MIN_WEIGHT * R.min(axis=0)
        skip = set().union(*(excluded_films(conn, m) for m in members))
        skip = [t for t in skip if t in self.catalog.index]
        if skip:
            score[self.catalog.positions(skip)] = -np.inf
        top = self.diversify(score, k)
        out = self.movies.iloc[top][["tconst", "title_ru", "title", "year"]].copy()
        out["min_fit"] = R.min(axis=0)[top]           # насколько фильм подходит тому, кому подходит меньше всех
        out["best"] = [members[i] for i in R[:, top].argmax(axis=0)]
        return out.reset_index(drop=True)
