"""Долгосрочный профиль пользователя: веса признаков (жанры, режиссёры, актёры, десятилетия,
страны, язык, длительность).

Профиль не правится по кусочку после каждой оценки, а каждый раз пересчитывается из всех
событий пользователя: профиль = Σ сигнал · затухание(давность) · вектор признаков фильма.
Это то же самое, что сдвиг весов после каждой оценки, но без накопления ошибок.
"""
import time
from datetime import datetime, timezone

import numpy as np

from recsys.features import meta_features

DAY = 86_400
HALF_LIFE_DAYS = 730  # через сколько дней оценка весит вдвое меньше; подобрано в recsys/tune.py
CENTER = 3.0          # нейтральная оценка, пока оценок мало
MIN_RATINGS_FOR_MEAN = 5  # дальше нейтральной считается средняя оценка пользователя (лучше на валидации)

# Сигналы реакций, кроме оценки. Ответ «не смотрел» и проигнорированная рекомендация —
# слабый минус; «не интересно» — заметный минус; «уже видел» вкус не меняет, только список.
SIGNALS = {
    "not_seen": -0.1,
    "ignored": -0.1,
    "not_interested": -0.5,
    "later": 0.1,
    "seen": 0.0,
    "skip": 0.0,
    "review": 0.0,  # сам факт отзыва вкус не меняет — меняет его тональность (SENTIMENT_WEIGHT)
}
# тональность отзыва (−1…+1) — добавка к оценке: 5★ дают +1, восторженный отзыв ещё +0,25
SENTIMENT_WEIGHT = 0.25
REPEAT_PAUSE_DAYS = 30  # сколько не показывать снова «не смотрел» из калибровки и пролистанное
REACTION_LISTS = {"seen": "seen", "later": "planned"}  # реакция → список пользователя


def rating_signal(rating: float, center: float = CENTER) -> float:
    """Оценка 1–5 → сигнал от −1 до +1."""
    return float(np.clip((rating - center) / 2, -1, 1))


def decay(age_seconds: float, half_life_days: float | None = HALF_LIFE_DAYS) -> float:
    if not half_life_days:
        return 1.0
    return 0.5 ** (max(age_seconds, 0) / (half_life_days * DAY))


class Profiler:
    """Строит профиль из событий. Матрица признаков считается один раз на весь каталог."""

    def __init__(self, tconsts: list[str], conn=None):
        self.X, self.names = meta_features(tconsts, with_names=True, conn=conn)
        self.ids = list(tconsts)
        self.index = {t: i for i, t in enumerate(tconsts)}

    def vector(self, events: list[tuple[str, float, float]], now: float | None = None,
               half_life_days: float | None = HALF_LIFE_DAYS) -> np.ndarray:
        """events — (tconst, сигнал, время в секундах); результат — вес каждого признака."""
        now = time.time() if now is None else now
        rows, weights = [], []
        for tconst, signal, ts in events:
            if tconst in self.index and signal:
                rows.append(self.index[tconst])
                weights.append(signal * decay(now - ts, half_life_days))
        if not rows:
            return np.zeros(self.X.shape[1])
        return np.asarray(self.X[rows].T @ np.array(weights)).ravel()

    def scores(self, profile: np.ndarray) -> np.ndarray:
        """Насколько каждый фильм каталога подходит профилю."""
        return self.X @ profile

    def as_dict(self, profile: np.ndarray) -> dict[str, float]:
        nz = np.flatnonzero(profile)
        return {self.names[i]: float(profile[i]) for i in nz}

    def from_dict(self, weights: dict[str, float]) -> np.ndarray:
        col = {n: i for i, n in enumerate(self.names)}
        v = np.zeros(len(self.names))
        for name, w in weights.items():
            if name in col:
                v[col[name]] = w
        return v


# --- Работа с БД --------------------------------------------------------------

def _ts(text: str) -> float:
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc).timestamp()


def ensure_user(conn, user_id: int) -> None:
    conn.execute("INSERT OR IGNORE INTO users (user_id) VALUES (?)", (user_id,))


def record_rating(conn, user_id: int, tconst: str, rating: int, source: str = "recommendation") -> None:
    ensure_user(conn, user_id)
    conn.execute(
        """INSERT INTO ratings (user_id, tconst, rating, source) VALUES (?, ?, ?, ?)
           ON CONFLICT(user_id, tconst) DO UPDATE SET rating = excluded.rating, source = excluded.source,
             created_at = CURRENT_TIMESTAMP""",
        (user_id, tconst, rating, source),
    )
    set_list(conn, user_id, tconst, "seen")


def record_reaction(conn, user_id: int, tconst: str, reaction: str, review: str | None = None,
                    sentiment: float | None = None) -> None:
    if reaction not in SIGNALS:
        raise ValueError(f"неизвестная реакция: {reaction}")
    ensure_user(conn, user_id)
    conn.execute(
        "INSERT INTO reactions (user_id, tconst, reaction, review, sentiment) VALUES (?, ?, ?, ?, ?)",
        (user_id, tconst, reaction, review, sentiment),
    )
    if reaction in REACTION_LISTS:
        set_list(conn, user_id, tconst, REACTION_LISTS[reaction])


def set_list(conn, user_id: int, tconst: str, list_name: str) -> None:
    """Списки: seen — видел, planned — в планах, hidden — скрыть навсегда."""
    ensure_user(conn, user_id)
    conn.execute("INSERT OR IGNORE INTO user_lists (user_id, tconst, list) VALUES (?, ?, ?)",
                 (user_id, tconst, list_name))
    if list_name == "seen":  # посмотренный фильм больше не «в планах»
        conn.execute("DELETE FROM user_lists WHERE user_id = ? AND tconst = ? AND list = 'planned'", (user_id, tconst))


def excluded_films(conn, user_id: int) -> set[str]:
    """Что не рекомендовать. Навсегда: оценённое, просмотренное, скрытое, «не интересно» и «в планах»
    (он уже в списке). На REPEAT_PAUSE_DAYS: «не смотрел» и «пропустить» в калибровке и показанное
    в подборке без ответа — чтобы те же фильмы не возвращались сразу."""
    pause = f"-{REPEAT_PAUSE_DAYS} days"
    rows = conn.execute(
        """SELECT tconst FROM ratings WHERE user_id = :u
           UNION SELECT tconst FROM user_lists WHERE user_id = :u AND list IN ('seen', 'hidden', 'planned')
           UNION SELECT tconst FROM reactions WHERE user_id = :u AND reaction = 'not_interested'
           UNION SELECT tconst FROM reactions WHERE user_id = :u AND reaction IN ('not_seen', 'skip')
                 AND created_at > datetime('now', :pause)
           UNION SELECT tconst FROM recommendations WHERE user_id = :u AND (answer IS NULL OR answer = 'ignored')
                 AND shown_at > datetime('now', :pause)""",
        {"u": user_id, "pause": pause},
    )
    return {r[0] for r in rows}


def user_events(conn, user_id: int) -> list[tuple[str, float, float]]:
    """Оценки и реакции пользователя как сигналы. Реакции на фильм, который потом оценён, не учитываются;
    исключение — тональность отзыва: она уточняет оценку."""
    rated = conn.execute("SELECT tconst, rating, created_at FROM ratings WHERE user_id = ?", (user_id,)).fetchall()
    center = np.mean([r for _, r, _ in rated]) if len(rated) >= MIN_RATINGS_FOR_MEAN else CENTER
    events = [(t, rating_signal(r, center), _ts(c)) for t, r, c in rated]
    rated_ids = {t for t, _, _ in rated}
    for t, reaction, c, sent in conn.execute(
        "SELECT tconst, reaction, created_at, sentiment FROM reactions WHERE user_id = ?", (user_id,)
    ):
        if reaction == "review" and sent is not None:
            events.append((t, SENTIMENT_WEIGHT * sent, _ts(c)))
        elif t not in rated_ids:
            events.append((t, SIGNALS[reaction], _ts(c)))
    return events


def rebuild_profile(conn, profiler: Profiler, user_id: int) -> np.ndarray:
    """Пересчитать профиль по всем событиям и сохранить в user_profile."""
    profile = profiler.vector(user_events(conn, user_id))
    conn.execute("DELETE FROM user_profile WHERE user_id = ?", (user_id,))
    conn.executemany(
        "INSERT INTO user_profile (user_id, feature, weight) VALUES (?, ?, ?)",
        [(user_id, f, w) for f, w in profiler.as_dict(profile).items()],
    )
    return profile


def load_profile(conn, profiler: Profiler, user_id: int) -> np.ndarray:
    rows = conn.execute("SELECT feature, weight FROM user_profile WHERE user_id = ?", (user_id,))
    return profiler.from_dict(dict(rows.fetchall()))
