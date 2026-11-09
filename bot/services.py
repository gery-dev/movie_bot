"""Общие объекты бота: БД, рекомендательная модель, клиент TMDB и действия пользователя."""
import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass, field

import pandas as pd

from data_pipeline.tmdb import TMDB
from db.database import connect
from recsys import calibration
from recsys.models import ALS, Catalog
from recsys.profile import ensure_user, rebuild_profile, record_rating, record_reaction, set_list
from recsys.recommender import MODEL_VERSION, Recommender
from recsys.search import NameIndex
from recsys.sentiment import load as load_sentiment, sentiment
from recsys.train_als import ALS_FILE

log = logging.getLogger(__name__)
CYRILLIC = re.compile("[а-яё]", re.IGNORECASE)
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
    kind: str | None = None      # new — новинки по вкусу (/new)
    search_id: int | None = None  # строка search_log, если это результаты поиска текстом

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
        als = None
        if with_als:  # векторы фильмов готовит python -m recsys.train_als
            ids = [r[0] for r in self.conn.execute("SELECT tconst FROM movies ORDER BY tconst")]
            als = ALS.load(ALS_FILE, Catalog(ids))
        self.rec = Recommender(self.conn, als=als)
        if with_als:  # модели для запросов текстом и тональности грузятся секунды — лучше при запуске
            self.rec.text_model()
            load_sentiment()
        self.movies = self.rec.movies.set_index("tconst")
        self.pool = calibration.calibration_pool(
            pd.read_sql("SELECT tconst, year, genres, country, rating, votes FROM movies", self.conn).set_index("tconst")
        )
        self.tmdb = TMDB(self.conn)
        people = pd.read_sql(
            """SELECT mp.tconst, mp.role, mp.ord, coalesce(p.name_ru, p.name) AS name FROM movie_people mp JOIN people p USING (nconst)
               ORDER BY mp.tconst, mp.role, mp.ord""", self.conn)
        self.credits = {t: g for t, g in people.groupby("tconst")}
        self.last_rated: dict[int, tuple[str, float]] = {}  # пользователь → (фильм, время) для отзыва текстом
        self.carousels: dict[int, Carousel] = {}             # пользователь → подборка, которую он листает
        self.pending_text: dict[int, str] = {}               # текст, про который спросили «отзыв или поиск?»
        self.survey_offered: set[tuple[int, int]] = set()     # (пользователь, волна) — опрос уже задан
        self.reason_msgs: dict[int, int] = {}                 # пользователь → сообщение «Почему не интересно?» без ответа
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
        p["shown"] = p.name_ru.fillna(p.name)  # в боте — русское имя из Wikidata, если оно есть
        self.people_info = p.set_index("nconst")[["shown", "films"]]

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
        return [{"nconst": n, "name": info.at[n, "shown"], "films": int(info.at[n, "films"])} for n in found]

    def person_name(self, nconst: str) -> str:
        """Имя для показа: русское, если есть, иначе как в IMDb."""
        return self.conn.execute("SELECT coalesce(name_ru, name) FROM people WHERE nconst = ?", (nconst,)).fetchone()[0]

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

    def undo_answer(self, user_id: int, tconst: str, source: str) -> None:
        """Убрать ответ на фильм: «Назад» в калибровке (source = calibration) или «Отменить» у любимого
        фильма в знакомстве (source = onboarding)."""
        with self.conn:
            self.conn.execute("DELETE FROM ratings WHERE user_id = ? AND tconst = ? AND source = ?",
                              (user_id, tconst, source))
            self.conn.execute("""DELETE FROM reactions WHERE user_id = ? AND tconst = ?
                                 AND reaction IN ('not_seen', 'skip')""", (user_id, tconst))
            self.conn.execute("DELETE FROM user_lists WHERE user_id = ? AND tconst = ? AND list = 'seen'",
                              (user_id, tconst))
            rebuild_profile(self.conn, self.rec.profiler, user_id)
        log.info("user %s undid %s answer for %s", user_id, source, tconst)

    def hide(self, user_id: int, tconst: str, hidden: bool = True) -> None:
        with self.conn:
            if hidden:
                set_list(self.conn, user_id, tconst, "hidden")
                self._answer(user_id, tconst, "hidden")
            else:
                self.conn.execute("DELETE FROM user_lists WHERE user_id = ? AND tconst = ? AND list = 'hidden'",
                                  (user_id, tconst))

    def log_shown(self, user_id: int, tconsts: list[str], mode: str = "recommend",
                  start_position: int | None = None, mood: dict | None = None) -> None:
        """Записать показ. start_position — место первого фильма в подборке (с 1); mood — ответы /today."""
        with self.conn:
            ensure_user(self.conn, user_id)  # подборку могут попросить до /start
            n_ratings = self.conn.execute("SELECT count(*) FROM ratings WHERE user_id = ?", (user_id,)).fetchone()[0]
            mood_json = json.dumps(mood, ensure_ascii=False, sort_keys=True) if mood else None
            self.conn.executemany(
                """INSERT INTO recommendations (user_id, tconst, mode, position, n_ratings, model_version, mood)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [(user_id, t, mode, None if start_position is None else start_position + i, n_ratings,
                  MODEL_VERSION, mood_json) for i, t in enumerate(tconsts)])

    def log_search(self, user_id: int, kind: str, query: str, found: int) -> int:
        """Записать поиск (kind: film | person | text): длина запроса и число найденного, без самого текста."""
        with self.conn:
            cur = self.conn.execute("INSERT INTO search_log (user_id, kind, query_len, found) VALUES (?, ?, ?, ?)",
                                    (user_id, kind, len(query), found))
        return cur.lastrowid

    def mark_search_chosen(self, user_id: int, kind: str, search_id: int | None = None) -> None:
        """Человек выбрал результат: по search_id, а без него — в последнем поиске этого вида, где что-то нашлось."""
        with self.conn:
            if search_id is not None:
                self.conn.execute("UPDATE search_log SET chosen = 1 WHERE id = ?", (search_id,))
            else:
                self.conn.execute(
                    """UPDATE search_log SET chosen = 1 WHERE id = (SELECT max(id) FROM search_log
                       WHERE user_id = ? AND kind = ? AND found > 0)""", (user_id, kind))

    def set_reason(self, user_id: int, tconst: str, reason: str) -> None:
        """Причина отказа «не интересно» (коды — REASONS в keyboards.py) к последнему такому показу фильма."""
        with self.conn:
            self.conn.execute(
                """UPDATE recommendations SET reason = ? WHERE id = (SELECT max(id) FROM recommendations
                   WHERE user_id = ? AND tconst = ? AND answer = 'not_interested')""", (reason, user_id, tconst))

    def log_survey_offer(self, user_id: int, wave: int) -> None:
        with self.conn:
            self.conn.execute("INSERT INTO survey_offers (user_id, wave) VALUES (?, ?)", (user_id, wave))

    def _answer(self, user_id: int, tconst: str, answer: str) -> None:
        self.conn.execute(
            "UPDATE recommendations SET answer = ? WHERE user_id = ? AND tconst = ? AND answer IS NULL",
            (answer, user_id, tconst),
        )
