"""Профиль на сегодня: короткий опрос, действующий несколько часов.

Длительность и язык — жёсткие фильтры (фильм либо подходит, либо нет).
Настроение, компания, новизна и эпоха — приоритеты: поднимают подходящие фильмы в выдаче.
"""
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from recsys.profile import ensure_user

VALID_HOURS = 6

# варианты ответов: ключ → подпись для кнопки
QUESTIONS = {
    "mood": {"fun": "Повеселее", "tense": "Напряжённое", "thoughtful": "Задуматься", "romantic": "Романтичное", "any": "Любое"},
    "time": {"90": "До 1,5 часа", "120": "До 2 часов", "any": "Не важно"},
    "company": {"alone": "Один", "couple": "Вдвоём", "friends": "С друзьями", "family": "С семьёй"},
    "novelty": {"new": "Что-то новое", "proven": "Проверенное", "any": "Не важно"},
    "language": {"ru": "Русское", "foreign": "Зарубежное", "any": "Не важно"},
    "era": {"modern": "Современное", "2000s": "2000-е", "90s": "90-е", "classic": "Классика до 1990", "any": "Не важно"},
}

# приоритеты жанров (жанры IMDb): +1 — подходит, отрицательное — лучше не надо
MOOD_GENRES = {
    "fun": {"Comedy": 1, "Animation": 0.5, "Family": 0.5, "Adventure": 0.5, "Horror": -1},
    "tense": {"Thriller": 1, "Crime": 1, "Action": 0.5, "Mystery": 0.5, "Horror": 0.5},
    "thoughtful": {"Drama": 1, "Biography": 0.5, "History": 0.5, "Documentary": 0.5, "Sci-Fi": 0.5},
    "romantic": {"Romance": 1, "Drama": 0.3, "Musical": 0.3},
}
COMPANY_GENRES = {
    "couple": {"Romance": 0.5, "Comedy": 0.3},
    "friends": {"Comedy": 0.5, "Action": 0.5, "Horror": 0.3},
    "family": {"Family": 1, "Animation": 1, "Adventure": 0.5, "Horror": -2, "Thriller": -0.5},
}
ERAS = {"modern": (2010, 9999), "2000s": (2000, 2009), "90s": (1990, 1999), "classic": (0, 1989)}
NEW_YEARS = 5  # «новое» — вышедшее за последние 5 лет


def today_filter(movies: pd.DataFrame, answers: dict) -> np.ndarray:
    """Маска фильмов, проходящих жёсткие условия (длительность и язык)."""
    ok = np.ones(len(movies), dtype=bool)
    limit = answers.get("time", "any")
    if limit != "any":
        ok &= movies.runtime.fillna(999).to_numpy() <= int(limit)
    lang = answers.get("language", "any")
    if lang == "ru":
        ok &= movies.language.fillna("").to_numpy() == "ru"
    elif lang == "foreign":
        ok &= movies.language.fillna("").to_numpy() != "ru"
    return ok


def today_scores(movies: pd.DataFrame, answers: dict, now_year: int | None = None) -> np.ndarray:
    """Приоритет каждого фильма по ответам опроса (0 — без предпочтений)."""
    now_year = now_year or datetime.now().year
    genres = movies.genres.fillna("").str.split(",")
    boosts = {}
    for table, key in [(MOOD_GENRES, "mood"), (COMPANY_GENRES, "company")]:
        for g, w in table.get(answers.get(key), {}).items():
            boosts[g] = boosts.get(g, 0) + w
    s = np.array([sum(boosts.get(g, 0) for g in gs) for gs in genres], dtype=float)

    year = movies.year.fillna(0).to_numpy()
    era = ERAS.get(answers.get("era"))
    if era:
        s += ((year >= era[0]) & (year <= era[1])).astype(float)
    novelty = answers.get("novelty")
    if novelty == "new":
        s += (year >= now_year - NEW_YEARS).astype(float)
    elif novelty == "proven":
        # проверенное — высокий рейтинг при большом числе голосов
        v, r = movies.votes.fillna(0).to_numpy(float), movies.rating.fillna(0).to_numpy(float)
        s += np.clip((r - 7) / 1.5, 0, 1) * np.clip(np.log10(np.maximum(v, 1)) - 4, 0, 2) / 2
    return s


def save_daily(conn, user_id: int, answers: dict, hours: int = VALID_HOURS) -> None:
    expires = (datetime.now() + timedelta(hours=hours)).isoformat(timespec="seconds")
    ensure_user(conn, user_id)
    conn.execute(
        "INSERT OR REPLACE INTO daily_session (user_id, answers, expires_at) VALUES (?, ?, ?)",
        (user_id, json.dumps(answers, ensure_ascii=False), expires),
    )


def load_daily(conn, user_id: int) -> dict | None:
    """Ответы на сегодня или None, если опроса не было или срок вышел."""
    row = conn.execute("SELECT answers, expires_at FROM daily_session WHERE user_id = ?", (user_id,)).fetchone()
    if row is None or datetime.fromisoformat(row[1]) < datetime.now():
        return None
    return json.loads(row[0])
