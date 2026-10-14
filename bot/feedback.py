"""Отложенная обратная связь: через FEEDBACK_DELAY после подборки бот спрашивает про один фильм
«посмотрел?» — не чаще раза в ASK_GAP. Рекомендация без ответа дольше IGNORE_AFTER считается
проигнорированной (слабый отрицательный сигнал). Интервалы задаются в .env в минутах."""
import asyncio
import logging
import os

from aiogram import Bot

from bot.cards import send_card
from bot.services import App

log = logging.getLogger(__name__)
CHECK_EVERY = 60  # секунд


def minutes(name: str, default: int) -> int:
    return int(os.getenv(name, default))


async def feedback_loop(bot: Bot, app: App) -> None:
    delay = minutes("FEEDBACK_DELAY_MIN", 24 * 60)
    gap = minutes("FEEDBACK_ASK_GAP_MIN", 6 * 60)
    ignore = minutes("FEEDBACK_IGNORE_AFTER_MIN", 7 * 24 * 60)
    while True:
        try:
            await ask_once(bot, app, delay, gap)
            mark_ignored(app, ignore)
        except Exception:
            log.exception("ошибка в цикле обратной связи")
        await asyncio.sleep(CHECK_EVERY)


async def ask_once(bot: Bot, app: App, delay: int, gap: int) -> None:
    rows = app.conn.execute(
        """SELECT r.id, r.user_id, r.tconst FROM recommendations r
           WHERE r.answer IS NULL AND r.asked_at IS NULL AND r.shown_at <= datetime('now', ?)
             AND NOT EXISTS (SELECT 1 FROM recommendations q WHERE q.user_id = r.user_id
                             AND q.asked_at > datetime('now', ?))
           ORDER BY r.shown_at""",
        (f"-{delay} minutes", f"-{gap} minutes"),
    ).fetchall()
    asked = set()
    for rec_id, user_id, tconst in rows:
        if user_id in asked:  # одному пользователю — один вопрос за раз
            continue
        asked.add(user_id)
        with app.conn:
            app.conn.execute("UPDATE recommendations SET asked_at = CURRENT_TIMESTAMP WHERE id = ?", (rec_id,))
        await send_card(bot, user_id, app, tconst, "fb",
                        note="Я недавно советовал этот фильм. Посмотрел? Оцени или выбери, почему нет.")
        log.info("asked user %s about %s", user_id, tconst)


def mark_ignored(app: App, ignore: int) -> None:
    rows = app.conn.execute(
        """SELECT DISTINCT user_id, tconst FROM recommendations
           WHERE answer IS NULL AND shown_at <= datetime('now', ?)""", (f"-{ignore} minutes",)).fetchall()
    for user_id, tconst in rows:
        app.react(user_id, tconst, "ignored")
