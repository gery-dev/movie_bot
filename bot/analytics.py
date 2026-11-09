"""Журнал действий пользователей (таблица events) — данные для будущего сайта с аналитикой.

Пишем, кто, когда и что нажал, на каком шаге сценария и сколько бот отвечал. Текст сообщений
и имена не сохраняем: из текста — только то, что это был текст, из команды — её название.
"""
import sqlite3
import time
from contextvars import ContextVar

from aiogram.dispatcher.event.bases import UNHANDLED
from aiogram.types import Update

DETAIL_LIMIT = 64  # данные кнопки в Telegram не длиннее 64 байт
# Ставится заглушками «не понял» и «кнопка устарела»: действие не попало в настоящий обработчик.
# Обработчик и middleware работают в одной задаче, поэтому значение видно после handler().
_caught_by_fallback: ContextVar[bool] = ContextVar("caught_by_fallback", default=False)


def mark_fallback() -> None:
    _caught_by_fallback.set(True)


def describe(update: Update) -> tuple[str, str | None, str | None]:
    """Обновление → (kind, name, detail)."""
    if update.message:
        m = update.message
        if m.text and m.text.startswith("/"):
            command, _, args = m.text.partition(" ")
            return "command", command[1:].split("@")[0].lower(), args.strip()[:DETAIL_LIMIT] or None
        if m.text:
            return "text", None, None
        if m.document:
            return "file", m.document.mime_type, None
        return "other", m.content_type, None
    if update.callback_query:
        prefix, _, rest = (update.callback_query.data or "").partition(":")
        return "callback", prefix, rest[:DETAIL_LIMIT] or None
    if update.my_chat_member:  # пользователь остановил или заново запустил бота
        return "member", update.my_chat_member.new_chat_member.status, None
    return "other", update.event_type, None


def log_event(conn: sqlite3.Connection, user_id: int | None, update: Update, state: str | None,
              duration_ms: int, error: bool, matched: bool | None = None) -> None:
    kind, name, detail = describe(update)
    with conn:
        conn.execute(
            """INSERT INTO events (user_id, kind, name, detail, state, duration_ms, error, matched)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, kind, name, detail, state, duration_ms, int(error), None if matched is None else int(matched)))


def middleware(conn: sqlite3.Connection):
    """Внешний обработчик обновлений: засекает время и пишет событие, даже если обработчик упал."""
    async def record(handler, event: Update, data: dict):
        user = data.get("event_from_user")
        state = data.get("raw_state")  # состояние до обработки: на каком шаге было действие
        start, error, unhandled = time.perf_counter(), False, False
        _caught_by_fallback.set(False)  # в одной задаче могут идти несколько обновлений подряд (и в тестах)
        try:
            result = await handler(event, data)
            unhandled = result is UNHANDLED
            return result
        except Exception:
            error = True
            raise
        finally:
            log_event(conn, user.id if user else None, event, state,
                      round((time.perf_counter() - start) * 1000), error,
                      matched=not (unhandled or _caught_by_fallback.get()))
    return record
