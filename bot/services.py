"""Общие объекты бота: БД, рекомендательная модель, клиент TMDB и действия пользователя."""
import logging
import re
import sqlite3
import time
from dataclasses import dataclass, field

import pandas as pd

from data_pipeline.movielens import load_ratings
from data_pipeline.tmdb import TMDB
from db.database import connect
from recsys import calibration
from recsys.favorites import search_people as search_people_db
from recsys.models import ALS, Catalog
from recsys.profile import ensure_user, rebuild_profile, record_rating, record_reaction, set_list
from recsys.recommender import Recommender
from recsys.sentiment import load as load_sentiment, sentiment

log = logging.getLogger(__name__)
CYRILLIC = re.compile("[а-яё]", re.IGNORECASE)
ALS_DATASET = "ml-latest-small"
ALS_PARAMS = dict(factors=32, regularization=0.1, alpha=1.0)  # лучшие на этапе 2
REVIEW_WINDOW = 15 * 60  # секунд после оценки, когда обычный текст считается отзывом


@dataclass
class Carousel:
    """Подборка, которая листается в одном сообщении."""
    films: list[str]
    idx: int = 0
    message_id: int | None = None
    statuses: dict[str, str] = field(default_factory=dict)  # фильм → итог ответа («Оценка 4 ★»)
    logged: set[str] = field(default_factory=set)            # какие фильмы уже записаны как показанные
    answers: dict | None = None  # ответы /today; None — обычная подборка /recommend
    person: str | None = None    # имя человека, если листаем его фильмы из /favorites (список не догружается)
    why: dict[str, str] = field(default_factory=dict)  # фильм → «Почему: …»
    query: str | None = None     # текст запроса, если это результаты поиска обычным текстом
    group: str | None = None     # имена участников, если это подборка для компании (/together)

    @property
    def fixed(self) -> bool:
        """Готовый список (фильмы человека, результаты запроса) — в конце не догружается."""
        return bool(self.person or self.query or self.group)

    @property
    def current(self) -> str:
        return self.films[self.idx]


class App:
    def __init__(self, conn: sqlite3.Connection | None = None, with_als: bool = True):
        self.conn = conn or connect()
        ml = load_ratings(ALS_DATASET) if with_als else None
        als = None
        if with_als:
            ids = [r[0] for r in self.conn.execute("SELECT tconst FROM movies ORDER BY tconst")]
            als = ALS(Catalog(ids), **ALS_PARAMS).fit(ml)
        self.rec = Recommender(self.conn, als=als, ml_ratings=ml)
        if with_als:  # модели для запросов текстом и тональности грузятся секунды — лучше при запуске
            self.rec.text_model()
            load_sentiment()
        self.movies = self.rec.movies.set_index("tconst")
        self.pool = calibration.calibration_pool(
            pd.read_sql("SELECT tconst, year, genres, country, rating, votes FROM movies", self.conn).set_index("tconst")
        )
        self.tmdb = TMDB(self.conn)
        people = pd.read_sql(
            """SELECT mp.tconst, mp.role, mp.ord, p.name FROM movie_people mp JOIN people p USING (nconst)
               ORDER BY mp.tconst, mp.role, mp.ord""", self.conn)
        self.credits = {t: g for t, g in people.groupby("tconst")}
        self.last_rated: dict[int, tuple[str, float]] = {}  # пользователь → (фильм, время) для отзыва текстом
        self.carousels: dict[int, Carousel] = {}             # пользователь → подборка, которую он листает
        self.pending_text: dict[int, str] = {}               # текст, про который спросили «отзыв или поиск?»
        self._titles = (self.movies.title_ru.fillna("") + " | " + self.movies.title.fillna("")).str.lower()

    # --- поиск ---

    def search_movies(self, query: str, limit: int = 5) -> list[tuple[str, str]]:
        """[(tconst, «Название (год)»)]: сначала точное совпадение, потом начало названия,
        потом вхождение в середину; внутри группы — самые известные первыми."""
        q = query.strip().lower()
        if len(q) < 2:
            return []
        hits = self.movies[self._titles.str.contains(q, regex=False)]
        names = [hits.title_ru.fillna("").str.lower(), hits.title.fillna("").str.lower()]
        exact = (names[0] == q) | (names[1] == q)
        prefix = names[0].str.startswith(q) | names[1].str.startswith(q)
        order = hits.assign(group=2 - exact.astype(int) - prefix.astype(int)).sort_values(
            ["group", "votes"], ascending=[True, False])
        return [(t, self.label(t)) for t in order.index[:limit]]

    async def search_people(self, query: str, limit: int = 5) -> list[dict]:
        """По-русски ищем через TMDB (имена в IMDb латиницей), латиницей — прямо в БД."""
        if not CYRILLIC.search(query):
            return search_people_db(self.conn, query, limit)
        found = await self.tmdb.search_person(query, limit)
        self.conn.commit()  # кэш TMDB
        result = []
        for nconst, name in found:
            row = self.conn.execute(
                """SELECT p.name, count(DISTINCT mp.tconst) FROM people p JOIN movie_people mp USING (nconst)
                   WHERE p.nconst = ?""", (nconst,)).fetchone()
            if row and row[1]:
                result.append({"nconst": nconst, "name": row[0], "films": row[1]})
        return result

    # --- описание фильма ---

    def label(self, tconst: str) -> str:
        m = self.movies.loc[tconst]
        year = f" ({int(m.year)})" if pd.notna(m.year) else ""
        return f"{m.title_ru or m.title}{year}"

    def people_of(self, tconst: str) -> tuple[list[str], list[str]]:
        g = self.credits.get(tconst)
        if g is None:
            return [], []
        return g[g.role == "director"].name.tolist(), g[g.role == "actor"].name.tolist()[:3]

    # --- действия пользователя ---

    def awaiting_review(self, user_id: int) -> str | None:
        """Фильм, оценённый за последние REVIEW_WINDOW секунд, — к нему относим текст пользователя."""
        tconst, ts = self.last_rated.get(user_id, (None, 0.0))
        return tconst if time.time() - ts <= REVIEW_WINDOW else None

    def save_review(self, user_id: int, tconst: str, text: str) -> float:
        """Сохранить отзыв с тональностью (−1…+1); она немного уточняет профиль."""
        tone = sentiment(text)
        self.react(user_id, tconst, "review", review=text[:2000], sentiment=tone)
        self.last_rated.pop(user_id, None)
        return tone

    def rate(self, user_id: int, tconst: str, rating: int, source: str) -> None:
        if source == "recommendation":
            self.last_rated[user_id] = (tconst, time.time())
        with self.conn:
            record_rating(self.conn, user_id, tconst, rating, source)
            self._answer(user_id, tconst, "rated")
            rebuild_profile(self.conn, self.rec.profiler, user_id)
        log.info("user %s rated %s: %s (%s)", user_id, tconst, rating, source)

    def react(self, user_id: int, tconst: str, reaction: str, review: str | None = None,
              sentiment: float | None = None) -> None:
        with self.conn:
            record_reaction(self.conn, user_id, tconst, reaction, review, sentiment)
            if reaction in ("seen", "not_interested", "later", "ignored"):
                self._answer(user_id, tconst, reaction)
            rebuild_profile(self.conn, self.rec.profiler, user_id)
        log.info("user %s reacted to %s: %s", user_id, tconst, reaction)

    def undo_calibration(self, user_id: int, tconst: str) -> None:
        """Кнопка «Назад» в калибровке: убрать ответ на фильм, чтобы ответить заново."""
        with self.conn:
            self.conn.execute("DELETE FROM ratings WHERE user_id = ? AND tconst = ? AND source = 'calibration'",
                              (user_id, tconst))
            self.conn.execute("""DELETE FROM reactions WHERE user_id = ? AND tconst = ?
                                 AND reaction IN ('not_seen', 'skip')""", (user_id, tconst))
            self.conn.execute("DELETE FROM user_lists WHERE user_id = ? AND tconst = ? AND list = 'seen'",
                              (user_id, tconst))
            rebuild_profile(self.conn, self.rec.profiler, user_id)
        log.info("user %s undid calibration answer for %s", user_id, tconst)

    def hide(self, user_id: int, tconst: str, hidden: bool = True) -> None:
        with self.conn:
            if hidden:
                set_list(self.conn, user_id, tconst, "hidden")
                self._answer(user_id, tconst, "hidden")
            else:
                self.conn.execute("DELETE FROM user_lists WHERE user_id = ? AND tconst = ? AND list = 'hidden'",
                                  (user_id, tconst))

    def log_shown(self, user_id: int, tconsts: list[str], mode: str = "recommend") -> None:
        with self.conn:
            ensure_user(self.conn, user_id)  # подборку могут попросить до /start
            self.conn.executemany("INSERT INTO recommendations (user_id, tconst, mode) VALUES (?, ?, ?)",
                                  [(user_id, t, mode) for t in tconsts])

    def _answer(self, user_id: int, tconst: str, answer: str) -> None:
        self.conn.execute(
            "UPDATE recommendations SET answer = ? WHERE user_id = ? AND tconst = ? AND answer IS NULL",
            (answer, user_id, tconst),
        )
