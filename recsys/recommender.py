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
from recsys.profile import Profiler, excluded_films, load_genres, refused_films, user_events
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
# Пишется в recommendations.model_version при каждом показе. Менять вручную, когда меняется то, что влияет
# на подборку: ALS-векторы, веса (A, B, C, W_NEW, W_GENRE, W_ERA, W_LANG, MMR), MIN_VOTES, формула.
MODEL_VERSION = "als32m-wnew0.35-eraref"  # eraref (10.11) — в подстройке под эпоху учитываются отказы
CANDIDATES = 300
CF_WEIGHT = 0.5            # доля ALS в долгосрочной части (этап 2)
# Холодный старт (recsys/cold_start.py, 150 пользователей MovieLens, NDCG@10; без данных — 0,048):
#   было: ALS с 5 оценок, без популярности — знакомство 0,019, + калибровка 0,031, импорт 0,102;
#   ALS с 1-й оценки и доля популярности N0 / (n + N0), N0 = 5 — 0,058, 0,062, 0,107.
MIN_RATINGS_FOR_CF = 1
# В ALS идут только оценки от CF_MIN_RATING: для неё оценка — сила интереса, и «1★» тянула бы к похожим.
# Нелюбимое учитывает профиль. Проверка и выбор 32M — в recsys/train_als.py.
CF_MIN_RATING = 4
POPULARITY_PRIOR = 5
POPULARITY_MIN_VOTES = 200_000  # поправка на число голосов во взвешенном рейтинге (этап 2)
# MMR (разнообразие выдачи). Подбор на 30 пользователях MovieLens, пятёрка фильмов:
#   без MMR — у 27% все 5 фильмов одного жанра, у 33% повтор режиссёра;
#   λ = 0,6 и жанры 0,3 — 0% и 7%, при этом 94% фильмов из той же лучшей десятки.
MMR_LAMBDA = 0.6        # доля «подходит» против «не похож на выбранные»
MMR_POOL = 100          # из скольких лучших выбирать разнообразную выдачу
GENRE_SIM_WEIGHT = 0.3  # доля похожести по жанрам (остальное — режиссёр, актёры, десятилетие…)
GROUP_MIN_WEIGHT = 0.5  # /together: доля «минимума по участникам» против среднего
# Свежесть: r растёт от 0 у фильмов старше RECENT_SPAN лет до 1 у фильмов этого года; W_NEW · r
# прибавляется на шаге 2, как пожелания на сегодня (на шаге 1 прибавка тонет в разбросе долгосрочной оценки).
# Мягкий сдвиг по решению пользователя. 40 пользователей MovieLens по 10 оценок, топ-10
# (медиана года / доля фильмов с 2016 года / доля до 2000):
#   до правки ALS (новые фильмы получали минимум ALS) — 1994 / 0% / 73%;
#   UNKNOWN_POP = 0,2:  W_NEW = 0,1 — 1999 / 12% / 54%;  0,2 — 2009 / 27% / 30%;  0,25 — 2010 / 36% / 20%;
#   0,3 — 2014 / 44% / 12%;  0,35 — 2016 / 50% / 6%. Берём 0,25: новые фильмы — треть выдачи, классика остаётся.
# С ALS на 32M (28.10) она сама меньше тянет новые фильмы — пересчёт на 1000 пользователях 32M с оценками 2019+,
# после знакомства NDCG@10 / доля с 2016 / с 2019 / до 2000 (без данных — NDCG):
#   было (small, 0,25) — 0,051 / 38% / 36% / 5% (0,048);  32M: 0,15 — 0,061 / 16% / 11% / 28% (0,045);
#   0,25 — 0,058 / 29% / 23% / 9% (0,048);  0,35 — 0,055 / 41% / 33% / 1% (0,047);  0,45 — 0,052 / 50% / 42% / 0%.
# Берём 0,35 — прежний баланс. После калибровки и импорта новых 2–8% при любом W: их задаёт эпоха понравившихся.
# UNKNOWN_POP — доля популярности вместо ALS у фильмов, которых нет в MovieLens. В /new у этих 40 человек
#   0,5 — 16 разных фильмов на всех (одни хиты), 0,3 — 29, 0,2 — 46; медиана голосов IMDb везде ~575 тыс.
W_NEW = 0.35
UNKNOWN_POP = 0.2
# Любимые жанры из знакомства: прибавка W_GENRE · (доля жанров фильма из любимых) на шаге 2, среди кандидатов.
# MIN_VOTES — в подборки по вкусу, под настроение и новинки попадают фильмы не меньше чем с MIN_VOTES голосов IMDb.
# История: сначала жанры шли в профиль (вес 2) и уменьшали долю популярности как 8 оценок. На модели MovieLens
# это выглядело хорошо, но у живых пользователей наверх полезли малоизвестные фильмы, прежде всего новые и
# документальные: медиана голосов показанных фильмов упала с 1,76 млн до 31 тыс., 60% показов — меньше 50 тыс.,
# оценки почти прекратились. Профиль из одних жанров сильнее всего поднимает фильмы с малым числом признаков.
# Проверка на копии базы сервера (5 человек с жанрами) и 60 пользователях MovieLens (4 оценки, 2 жанра):
#                                   живые: жанры / медиана голосов / < 50 тыс.   MovieLens: жанры / NDCG@10
#   до правки жанров                       52% / 130 тыс. / 34%                         39% / 0,153
#   жанры в профиле (было)                 92% /  73 тыс. / 46%                         76% / 0,196
#   порог 50 тыс. + шаг 2, вес 0,3         80% / 272 тыс. /  0%                         91% / 0,184
#   то же + UNKNOWN_POP 0,5                86% / 478 тыс. /  0%                         84% / 0,149
# Берём третий вариант: малоизвестного нет, жанры соблюдаются, точность на MovieLens лучшая.
W_GENRE = 0.3
MIN_VOTES = 50_000
# Эпоха и язык, которые нравятся человеку, — по фильмам с его оценкой 4–5 (знакомство, калибровка, подборки).
# era — насколько год фильма близок к годам понравившихся (ядро шириной ERA_BANDWIDTH лет), lang — доля
# понравившихся фильмов на языке фильма. Нужно, когда понравилось в основном новое, а бот тянул классику
# 80–90-х, и когда по высокому рейтингу IMDb наверх выходили фильмы на языках, которых человек не смотрит.
# 150 пользователей MovieLens, NDCG@10 (знакомство + калибровка / импорт всех оценок):
#   без них — 0,027 / 0,032;  язык 0,15 — 0,029 / 0,034;  эпоха 0,3 — 0,035 / 0,096;  эпоха 0,5 — 0,042 / 0,097;
#   эпоха 0,5 + язык 0,15 — 0,041 / 0,094;  + язык 0,3 — 0,041 / 0,092;  эпоха 0,75 — 0,042 / 0,094;  1,0 — меньше.
# Берём эпоху 0,5: главный прирост (свежесть W_NEW без неё тянула к годам, которые человеку не нравятся).
# Язык на MovieLens почти нейтрален; на копии данных сервера при 0,15 у тестировщика, смотрящего только
# англоязычное, в десятке оставался индийский фильм с накрученным рейтингом IMDb, при 0,3 — уже нет.
# Отказы в эпохе (10.11): годы отвергнутых («не интересно») вычитаются из годов понравившихся. На живых данных
# после первого отказа от фильма до 2000 года таких показали ещё 41 (5 человек): 17 отказов, 5 оценок 4–5 —
# подстройка строилась только по понравившимся и отказы не видела. Проверка на 264 пользователях MovieLens
# small (отказ — фильм с оценкой ≤ 2), 10 оценок + 3 отказа: близость десятки к годам отвергнутых
# 0,605 → 0,457 (−0,148 ± 0,024), NDCG@10 0,0378 → 0,0337 (−0,004 ± 0,006, в пределах шума).
# Проверено и не взято: язык «понравившиеся минус отвергнутые» — он считает доли, отказы от английских фильмов
# (их показывают больше всего) снимают преимущество английского, и доля фильмов на языках отвергнутых в десятке
# не упала, а выросла (0 → 0,005–0,023); включать подстройку по одним отказам (без 3 понравившихся) — NDCG при
# 0 оценок + 3 отказа −0,010 ± 0,006; отказы в доле популярности N0 / (n + N0) — там же −0,004 ± 0,003.
W_ERA = 0.5
W_LANG = 0.3
ERA_BANDWIDTH = 8
MIN_LIKED_FOR_FIT = 3
RECENT_SPAN = 30
NEW_YEARS = 5           # /new — фильмы за столько последних календарных лет, включая текущий


def recency(years: np.ndarray, now_year: int | None = None) -> np.ndarray:
    now_year = now_year or datetime.now().year
    r = (np.asarray(years, float) - (now_year - RECENT_SPAN)) / RECENT_SPAN
    return np.nan_to_num(np.clip(r, 0, 1))


def rerank(base: np.ndarray, today: np.ndarray, b: float, fav: np.ndarray, c: float,
           n: int = CANDIDATES, new: np.ndarray | None = None, w_new: float = 0.0,
           genre: np.ndarray | None = None, w_genre: float = 0.0,
           extra: list[tuple[np.ndarray, float]] | None = None) -> np.ndarray:
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
    if w_new and new is not None and np.ptp(new[cand]) > 0:
        s += w_new * _unit(new[cand])
    if w_genre and genre is not None and np.ptp(genre[cand]) > 0:
        s += w_genre * _unit(genre[cand])
    for vec, w in extra or []:  # эпоха и язык понравившихся фильмов
        if w and np.ptp(vec[cand]) > 0:
            s += w * _unit(vec[cand])
    score[cand] = s
    return score


class Recommender:
    def __init__(self, conn, als=None, kinds: tuple[str, ...] = ("movie",)):
        """als — recsys.models.ALS на том же каталоге, обученная или из файла (необязательно).
        kinds — что входит в каталог. Фильмы считаются отдельно от сериалов: общие нормировки и веса
        TF-IDF по каталогу с сериалами сдвигали подборки фильмов (на данных сервера менялась бо́льшая часть
        десяток). Для /series — каталог ("movie", "series"): вкус берётся и из оценок фильмов."""
        self.kinds = kinds
        self.movies = pd.read_sql(
            f"""SELECT tconst, title, title_ru, year, runtime, genres, language, rating, votes, kind, end_year FROM movies
                WHERE kind IN ({",".join("?" * len(kinds))}) ORDER BY tconst""",
            conn, params=list(kinds),
        )
        self.catalog = Catalog(self.movies.tconst.tolist())
        self.profiler = Profiler(self.catalog.ids, conn=conn)
        self.als = als
        v, r = self.movies.votes.to_numpy(float), self.movies.rating.to_numpy(float)
        self.popularity = (v * r + POPULARITY_MIN_VOTES * r.mean()) / (v + POPULARITY_MIN_VOTES)
        people = pd.read_sql("SELECT tconst, nconst FROM movie_people", conn)
        people = people[people.tconst.isin(self.catalog.index)]
        self.people = people.assign(pos=self.catalog.positions(people.tconst))
        self.names = dict(conn.execute("SELECT nconst, coalesce(name_ru, name) FROM people").fetchall())
        genres = self.movies.genres.fillna("").str.split(",").map(lambda g: [x for x in g if x])
        self.genre_matrix = normalize(MultiLabelBinarizer(sparse_output=True).fit_transform(genres).astype(float)).tocsr()
        self.recency = recency(self.movies.year.to_numpy(float))
        self.genre_sets = [set(g) for g in genres]
        self._embeddings = self._text_model = None
        # фильмы без оценок MovieLens ALS не знает
        self.cf_known = als.known if als is not None else np.ones(len(self.catalog), bool)

    def long_term_parts(self, conn, user_id: int, events=None) -> dict:
        """Долгосрочная оценка и её части — нужны и для выдачи, и для объяснения."""
        events = user_events(conn, user_id) if events is None else events
        if not events:
            return {"total": _unit(self.popularity), "profile": None, "profile_score": None, "cf": None}
        profile = self.profiler.vector(events)
        profile_score = _unit(self.profiler.scores(profile))
        parts = {"total": profile_score, "profile": profile, "profile_score": profile_score, "cf": None}
        ratings = conn.execute("SELECT tconst, rating FROM ratings WHERE user_id = ?", (user_id,)).fetchall()
        # фильмы, добавленные после запуска (например, из проката), модели ещё не известны
        ratings = [(t, r) for t, r in ratings if t in self.catalog.index]
        liked = [(t, r) for t, r in ratings if r >= CF_MIN_RATING]
        if self.als is not None and len(liked) >= MIN_RATINGS_FOR_CF:
            row = sparse.csr_matrix(
                ([float(r) for _, r in liked], ([0] * len(liked), self.catalog.positions([t for t, _ in liked]))),
                shape=(1, len(self.catalog)),
            )
            factors = self.als.model.recalculate_user(0, row)
            cf = _unit(np.where(self.cf_known, self.als.model.item_factors @ np.asarray(factors).ravel(), -np.inf))
            # фильмы, которых нет в MovieLens (в 32M — почти всё новее 2023 года), ALS не знает. Раньше они получали минимум
            # ALS и почти не попадали в выдачу; теперь вместо ALS у них взвешенный рейтинг IMDb — ALS тоже
            # тянет вверх известные фильмы, а с одним профилем наверх лезли малоизвестные новинки
            mix = CF_WEIGHT * cf + (1 - CF_WEIGHT) * profile_score
            unknown = UNKNOWN_POP * _unit(self.popularity) + (1 - UNKNOWN_POP) * profile_score
            parts.update(cf=cf, total=np.where(self.cf_known, mix, unknown))
        # пока оценок мало, вкус известен плохо — подмешиваем популярность: её доля N0 / (n + N0)
        if POPULARITY_PRIOR:
            share = POPULARITY_PRIOR / (len(ratings) + POPULARITY_PRIOR)
            parts["total"] = (1 - share) * parts["total"] + share * _unit(self.popularity)
        return parts

    def genre_match(self, genres: set[str]) -> np.ndarray:
        """Доля жанров фильма из любимых (0 — ни одного)."""
        if not genres:
            return np.zeros(len(self.catalog))
        return np.array([len(g & genres) / len(g) if g else 0.0 for g in self.genre_sets])

    def _year_closeness(self, films: list[str]) -> tuple[np.ndarray, int]:
        """Сумма близостей года каждого фильма каталога к годам films и число films с известным годом."""
        years = self.movies.year.to_numpy(float)
        if not films:
            return np.zeros(len(years)), 0
        y = years[self.catalog.positions(films)]
        # по уникальным годам с их числом, а не по каждому фильму: у импортировавших тысячу оценок
        # матрица «каталог × понравившиеся» заняла бы сотни мегабайт
        uniq, counts = np.unique(y[np.isfinite(y)], return_counts=True)
        kernel = np.exp(-0.5 * ((years[:, None] - uniq[None, :]) / ERA_BANDWIDTH) ** 2)
        return kernel @ counts, int(counts.sum())

    def taste_fit(self, conn, user_id: int) -> tuple[np.ndarray, np.ndarray] | None:
        """(эпоха, язык) по фильмам с оценкой 4–5; None — понравившихся пока меньше MIN_LIKED_FOR_FIT.
        В эпохе годы отвергнутых фильмов вычитаются (см. комментарий у W_ERA)."""
        liked = [t for (t,) in conn.execute("SELECT tconst FROM ratings WHERE user_id = ? AND rating >= 4", (user_id,))
                 if t in self.catalog.index]
        if len(liked) < MIN_LIKED_FOR_FIT:
            return None
        pos = self.catalog.positions(liked)
        refused = [t for t in refused_films(conn, user_id) if t in self.catalog.index]
        (near_liked, n_liked), (near_refused, n_refused) = self._year_closeness(liked), self._year_closeness(refused)
        era = (near_liked - near_refused) / max(n_liked + n_refused, 1)
        langs = self.movies.language.fillna("")
        share = langs.iloc[pos].value_counts(normalize=True)
        lang = langs.map(share).fillna(0).to_numpy(float)
        return np.nan_to_num(era), lang

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
                  diverse: bool = True, min_year: int | None = None, kind: str = "movie") -> pd.DataFrame:
        """Топ-k фильмов с объяснением. answers — опрос на сегодня (/today); без него — обычная
        подборка по вкусу (/recommend). min_year — только фильмы не старше этого года (/new).
        kind — movie (фильмы) или series (/series). diverse=False — без MMR (для сравнения)."""
        answers = answers or {}
        genres = load_genres(conn, user_id)
        parts = self.long_term_parts(conn, user_id)
        long_term = parts["total"]
        today = today_scores(self.movies, answers) if answers else np.zeros(len(self.catalog))
        fav_people = favorite_people(conn, user_id)
        fav = self.favorites_bonus(conn, user_id, fav_people)

        ok = (self.movies.kind.fillna("movie") == kind).to_numpy(copy=True)
        if answers:
            ok &= today_filter(self.movies, answers)
        if min_year:
            ok &= self.movies.year.to_numpy(float) >= min_year
        if MIN_VOTES:
            ok &= self.movies.votes.fillna(0).to_numpy(float) >= MIN_VOTES
        skip = excluded_films(conn, user_id)
        if skip:
            ok[self.catalog.positions([t for t in skip if t in self.catalog.index])] = False
        base = np.where(ok, A * long_term + C * fav, -np.inf)
        # если в /today выбрана эпоха, свежесть ей не мешает
        # в /new и так одни новые фильмы — там свежесть только тянула бы к самому последнему году
        w_new = 0.0 if answers.get("era") not in (None, "any") or min_year else W_NEW
        fit = self.taste_fit(conn, user_id)
        extra = None
        if fit is not None:
            era, lang = fit
            # выбранная в /today эпоха и /new сами задают годы — там подстройка под годы не нужна
            w_era = 0.0 if answers.get("era") not in (None, "any") or min_year else W_ERA
            extra = [(era, w_era), (lang, W_LANG)]
        score = rerank(base, today, B, fav, C, new=self.recency, w_new=w_new,
                       genre=self.genre_match(genres), w_genre=W_GENRE, extra=extra)

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
            self._embeddings = load_embeddings(self.catalog.ids)  # у фильмов без описания — нули
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
