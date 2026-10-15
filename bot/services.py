"""Общие объекты бота: БД, рекомендательная модель, клиент TMDB и действия пользователя."""
import asyncio
import logging
import re
import sqlite3
import time
from dataclasses import dataclass, field

import pandas as pd

from data_pipeline.movielens import load_ratings
from data_pipeline.tmdb import TMDB, movie_row
from db.database import connect
from recsys import calibration
from recsys.models import ALS, Catalog
from recsys.profile import ensure_user, excluded_films, rebuild_profile, record_rating, record_reaction, set_list
from recsys.recommender import Recommender
from recsys.search import NameIndex
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
    kind: str | None = None      # new — новинки по вкусу (/new), cinema — сейчас в прокате (/cinema)

    @property
    def fixed(self) -> bool:
        """Готовый список (фильмы человека, результаты запроса, прокат) — в конце не догружается."""
        return bool(self.person or self.query or self.group or self.kind == "cinema")

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
        self._build_search()

    # --- поиск ---

    def _build_search(self) -> None:
        """Индексы нечёткого поиска: фильмы по трём названиям, люди по имени латиницей и русским вариантам."""
        m = pd.read_sql("SELECT tconst, title_ru, title, original_title, votes FROM movies", self.conn)
        self.film_index = NameIndex(m.tconst, m[["title_ru", "title", "original_title"]].values.tolist(), m.votes)
        p = pd.read_sql(
            """SELECT p.nconst, p.name, p.name_ru, p.aka_ru, count(DISTINCT mp.tconst) AS films, sum(m.votes) AS fame
               FROM people p JOIN movie_people mp USING (nconst) JOIN movies m USING (tconst)
               GROUP BY p.nconst""", self.conn)
        variants = [[name, ru] + (aka.split("|") if isinstance(aka, str) else [])
                    for name, ru, aka in zip(p.name, p.name_ru, p.aka_ru)]
        self.people_index = NameIndex(p.nconst, variants, p.fame)
        self.people_info = p.set_index("nconst")[["name", "films"]]

    def search_movies(self, query: str, limit: int = 5) -> list[tuple[str, str]]:
        """[(tconst, «Название (год)»)] — с опечатками, по-русски и латиницей; при равном сходстве известнее выше."""
        return [(t, self.label(t)) for t in self.film_index.search(query, limit)]

    async def search_people(self, query: str, limit: int = 5) -> list[dict]:
        """Ищем у себя (имена латиницей и по-русски из Wikidata, опечатки прощаются). Если по-русски
        никого нет — спрашиваем TMDB: там русские имена есть и у тех, кого нет в Wikidata."""
        found = self.people_index.search(query, limit)
        if not found and CYRILLIC.search(query):
            found = [n for n, _ in await self.tmdb.search_person(query, limit) if n in self.people_info.index]
            self.conn.commit()  # кэш TMDB
        info = self.people_info
        return [{"nconst": n, "name": info.at[n, "name"], "films": int(info.at[n, "films"])} for n in found]

    # --- прокат ---

    async def cinema_films(self, user_id: int) -> list[str]:
        """Фильмы в прокате, самые обсуждаемые первыми, без уже оценённых, скрытых и недавно показанных.
        Новинок, которых нет в выгрузке IMDb, добавляем в movies по карточке TMDB."""
        skip = excluded_films(self.conn, user_id)
        playing = await self.tmdb.now_playing()
        known = dict(self.conn.execute(
            f"SELECT tmdb_id, tconst FROM movies WHERE tmdb_id IN ({','.join('?' * len(playing))})",
            [m["id"] for m in playing]).fetchall()) if playing else {}
        missing = [m["id"] for m in playing if m["id"] not in known]
        added = dict(zip(missing, await asyncio.gather(*(self._add_movie(i) for i in missing))))  # параллельно
        films = []
        for m in playing:
            tconst = known.get(m["id"]) or added.get(m["id"])
            if tconst and tconst not in skip and tconst not in films:
                films.append(tconst)
        self.conn.commit()  # кэш TMDB и новые фильмы
        return films

    async def _add_movie(self, tmdb_id: int) -> str | None:
        row = movie_row(await self.tmdb.movie(tmdb_id))
        if row is None:  # без IMDb ID фильм не связать с оценками и остальным каталогом
            return None
        tconst = row[0]
        if tconst not in self.movies.index:
            self.conn.execute(
                """INSERT OR IGNORE INTO movies (tconst, title, title_ru, year, runtime, genres, tmdb_id, country,
                   language, overview_ru, poster_path, votes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)""", row)
            self.movies.loc[tconst, ["title", "title_ru", "year", "runtime", "genres", "votes"]] = \
                [row[1], row[2], row[3], row[4], row[5], 0]
        return tconst

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
