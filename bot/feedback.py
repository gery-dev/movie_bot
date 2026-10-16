"""Фоновая отметка «проигнорировано»: рекомендация без ответа дольше IGNORE_AFTER считается
проигнорированной (слабый отрицательный сигнал). Интервал задаётся в .env в минутах.

Раньше здесь же бот сам спрашивал «посмотрел?» про пролистанные фильмы. Карточка с постером выглядела как
непрошеная подборка, приходила и ночью, а ответили на 1 вопрос из 9 — по решению пользователя вопрос выключен.
Бот больше ничего не присылает сам, кроме ответа на действие пользователя."""
import asyncio
import logging
import os

from aiogram import Bot

from bot.services import App

log = logging.getLogger(__name__)
CHECK_EVERY = 60  # секунд


def minutes(name: str, default: int) -> int:
    return int(os.getenv(name, default))


async def feedback_loop(bot: Bot, app: App) -> None:
    ignore = minutes("FEEDBACK_IGNORE_AFTER_MIN", 7 * 24 * 60)
    while True:
        try:
            mark_ignored(app, ignore)
        except Exception:
            log.exception("ошибка в цикле обратной связи")
        await asyncio.sleep(CHECK_EVERY)


def mark_ignored(app: App, ignore: int) -> None:
    rows = app.conn.execute(
        """SELECT DISTINCT user_id, tconst FROM recommendations
           WHERE answer IS NULL AND shown_at <= datetime('now', ?)""", (f"-{ignore} minutes",)).fetchall()
    for user_id, tconst in rows:
        app.react(user_id, tconst, "ignored")
