"""Всё, что не поймали другие обработчики, и ошибки. Подключается последним."""
import logging

from aiogram import F, Router
from aiogram.types import CallbackQuery, ErrorEvent, Message

from bot import texts
from bot.keyboards import NavCb
from bot.services import App

router = Router()
log = logging.getLogger(__name__)


@router.callback_query(NavCb.filter(F.a == "noop"))
async def noop(callback: CallbackQuery) -> None:
    await callback.answer()


@router.callback_query()
async def stale(callback: CallbackQuery) -> None:
    """Кнопка из старого сообщения, которая уже не к месту (например, шаг знакомства пройден)."""
    await callback.answer("Эта кнопка уже не работает. Посмотри /help.")


@router.message(F.text, ~F.text.startswith("/"))
async def late_review(message: Message, app: App) -> None:
    """Текст вскоре после оценки — отзыв, даже если перед этим нажали «Пропустить»."""
    tconst = app.awaiting_review(message.from_user.id)
    if tconst is None:
        await message.answer(texts.NOT_UNDERSTOOD)
        return
    app.save_review(message.from_user.id, tconst, message.text)
    await message.answer(f"Записал отзыв о «{app.label(tconst)}». Спасибо!")


@router.message()
async def fallback(message: Message) -> None:
    await message.answer(texts.NOT_UNDERSTOOD)


@router.errors()
async def on_error(event: ErrorEvent) -> None:
    log.exception("ошибка при обработке обновления", exc_info=event.exception)
    update = event.update
    target = update.message or (update.callback_query.message if update.callback_query else None)
    if target:
        try:
            await target.answer(texts.ERROR)
        except Exception:  # ответить не получилось — ошибка уже в логе
            pass
