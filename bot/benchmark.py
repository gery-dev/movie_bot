"""Замеры времени ответа (этап 6, требования ТЗ п. 4.3): обычная команда ≤ 3 с, рекомендации ≤ 10 с,
импорт Letterboxd ≤ 60 с.

    python -m bot.benchmark

Работает на копии БД в памяти — настоящие данные не меняются. Сеть Telegram не учитывается (это задержка
доставки, а не работы бота), поэтому дополнительно берётся время обработки настоящих нажатий из logs/bot.log.
Результат — reports/timing.md.
"""
import asyncio
import csv
import io
import re
import sqlite3
import statistics
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

from bot.cards import build_card
from bot.services import App
from data_pipeline.letterboxd import import_export
from data_pipeline.movielens import load_ratings
from db.database import ROOT, connect
from recsys.favorites import person_unseen, suggest_favorites
from recsys.profile import rebuild_profile, record_rating

LIMITS = {"команда": 3, "рекомендации": 10, "импорт": 60}
REPEAT = 5


def timed(fn, repeat: int = REPEAT) -> list[float]:
    out = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t)
    return out


async def atimed(fn, repeat: int = REPEAT) -> list[float]:
    out = []
    for _ in range(repeat):
        t = time.perf_counter()
        await fn()
        out.append(time.perf_counter() - t)
    return out


def letterboxd_zip(movies, path: Path, n: int = 1000, fake_share: float = 0.05) -> None:
    """Выгрузка в формате Letterboxd: n известных фильмов из БД и ~5% выдуманных (их ищут через TMDB)."""
    rng = np.random.default_rng(0)
    real = movies.sample(int(n * (1 - fake_share)), random_state=0)
    rows = [(r.title, int(r.year), float(rng.integers(1, 11)) / 2) for r in real.itertuples()]
    rows += [(f"Nonexistent Film {i}", 2020, 3.0) for i in range(n - len(rows))]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Date", "Name", "Year", "Letterboxd URI", "Rating"])
    for name, year, rating in rows:
        w.writerow(["2024-01-01", name, year, "https://boxd.it/x", rating])
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("ratings.csv", buf.getvalue())


def from_log() -> list[float]:
    """Время обработки настоящих нажатий и сообщений: aiogram пишет «Duration N ms»."""
    log = ROOT / "logs" / "bot.log"
    if not log.exists():
        return []
    return [int(m) / 1000 for m in re.findall(r"is handled\. Duration (\d+) ms", log.read_text(encoding="utf-8"))]


def row(name: str, kind: str, times: list[float]) -> str:
    worst = max(times)
    ok = "да" if worst <= LIMITS[kind] else "**нет**"
    return f"| {name} | {kind} (≤ {LIMITS[kind]} с) | {statistics.median(times):.3f} | {worst:.3f} | {ok} |"


async def main() -> None:
    src = connect()
    conn = sqlite3.connect(":memory:")
    src.backup(conn)
    src.close()
    conn.row_factory = sqlite3.Row

    t = time.perf_counter()
    app = App(conn=conn)
    startup = time.perf_counter() - t

    # пользователи: новый (0 оценок), после калибровки (10), «импортировал Letterboxd» (300 оценок MovieLens)
    ml = load_ratings("ml-latest-small")
    heavy = ml[ml.userId == ml.userId.value_counts().index[30]].head(300)
    users = {"новый": 900001, "10 оценок": 900002, "300 оценок": 900003}
    with conn:
        for t_, r in zip(heavy.tconst[:10], heavy.rating[:10]):
            record_rating(conn, users["10 оценок"], t_, int(max(1, round(r))), "calibration")
        for t_, r in zip(heavy.tconst, heavy.rating):
            record_rating(conn, users["300 оценок"], t_, int(max(1, round(r))), "letterboxd")
    u = users["300 оценок"]
    films = app.rec.recommend(conn, u, 20).tconst.tolist()

    rows = []
    rows.append(row("Карточка фильма (TMDB из кэша)", "команда",
                    await atimed(lambda: build_card(app, films[np.random.randint(len(films))]), 20)))
    rows.append(row("Поиск фильма по названию", "команда", timed(lambda: app.search_movies("интерстеллар"), 20)))
    rows.append(row("Оценка фильма + пересчёт профиля", "команда",
                    timed(lambda: (record_rating(conn, u, films[0], 4), rebuild_profile(conn, app.rec.profiler, u)))))
    rows.append(row("Списки и любимые (/lists, /favorites)", "команда",
                    timed(lambda: (person_unseen(conn, u, "nm0634240"), suggest_favorites(conn, u)))))
    for name, uid in users.items():
        rows.append(row(f"/recommend, пользователь «{name}»", "рекомендации",
                        timed(lambda uid=uid: app.rec.recommend(conn, uid, 10))))
    rows.append(row("/today (весёлое, до 2 ч, с друзьями)", "рекомендации",
                    timed(lambda: app.rec.recommend(conn, u, 10, answers={"mood": "fun", "time": "120",
                                                                           "company": "friends"}))))
    rows.append(row("Поиск текстом с образцом", "рекомендации",
                    timed(lambda: app.rec.search(conn, u, "как «Интерстеллар», но повеселее", 20))))
    rows.append(row("Поиск текстом по смыслу", "рекомендации",
                    timed(lambda: app.rec.search(conn, u, "фильм про путешествия во времени", 20))))
    rows.append(row("/together, 3 человека", "рекомендации",
                    timed(lambda: app.rec.recommend_group(conn, list(users.values()), 20))))

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "letterboxd.zip"
        movies = app.rec.movies.merge(
            __import__("pandas").read_sql("SELECT tconst, original_title, tmdb_id FROM movies", conn), on="tconst")
        letterboxd_zip(movies, path)

        async def imp():
            conn.execute("DELETE FROM ratings WHERE user_id = 900004")
            await import_export(conn, 900004, path, movies, app.tmdb)
            rebuild_profile(conn, app.rec.profiler, 900004)
        rows.append(row("Импорт Letterboxd, 1 000 фильмов (5% ищутся в TMDB)", "импорт", await atimed(imp, 3)))
    await app.tmdb.close()

    live = from_log()
    live_text = (f"Настоящие нажатия из logs/bot.log ({len(live)} шт.): медиана {statistics.median(live):.3f} с, "
                 f"95-й процентиль {np.percentile(live, 95):.3f} с, максимум {max(live):.3f} с; "
                 f"дольше 3 с — {sum(x > 3 for x in live)}.") if live else "Лог бота пуст."
    text = (
        "# Время ответа бота (ТЗ п. 4.3)\n\n"
        f"Команда: `python -m bot.benchmark`. Копия БД в памяти, каждая операция — {REPEAT}–20 повторов. "
        "Время — работа бота без сети Telegram (доставка сообщения добавляет обычно 0,1–0,5 с).\n\n"
        "| Операция | Требование | Медиана, с | Максимум, с | Укладывается |\n|---|---|---|---|---|\n"
        + "\n".join(rows)
        + f"\n\nЗапуск бота (загрузка данных, ALS, моделей текста и тональности): {startup:.1f} с — в ТЗ не нормируется.\n\n"
        + live_text + "\n"
    )
    out = ROOT / "reports" / "timing.md"
    out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    asyncio.run(main())
