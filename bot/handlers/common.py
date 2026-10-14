"""Всё, что не поймали другие обработчики, и ошибки. Подключается последним."""
import logging

from aiogram import Bot, F, Router
from aiogram.types import CallbackQuery, ErrorEvent, Message

from bot import texts
from bot.handlers.recommend import start_query
from bot.keyboards import NavCb, nav_kb
from bot.services import App

router = Router()
log = logging.getLogger(__name__)


@router.callback_query(NavCb.filter(F.a == "noop"))
async def noop(callback: CallbackQuery) -> None:
    await callback.answer()


@router.message(F.text, ~F.text.startswith("/"))
async def free_text(message: Message, bot: Bot, app: App) -> None:
    """Текст вне шагов — запрос «найди фильм». Если недавно была оценка, спрашиваем: это отзыв или поиск."""
    user = message.from_user.id
    tconst = app.awaiting_review(user)
    if tconst is None:
        await message.answer("Ищу…")
        await start_query(bot, user, app, message.text)
        return
    app.pending_text[user] = message.text
    await message.answer("Это отзыв или поиск фильма?", reply_markup=nav_kb(
        (f"Отзыв о «{app.label(tconst)}»", "text_review"), ("Найти фильмы по этому тексту", "text_search")))


@router.callback_query(NavCb.filter(F.a.in_({"text_review", "text_search"})))
async def text_kind(callback: CallbackQuery, callback_data: NavCb, bot: Bot, app: App) -> None:
    user = callback.from_user.id
    text = app.pending_text.pop(user, None)
    await callback.answer()
    if text is None:
        return
    if callback_data.a == "text_search":
        await callback.message.edit_text("Ищу…")
        await start_query(bot, user, app, text)
        return
    tconst = app.awaiting_review(user)
    if tconst:
        app.save_review(user, tconst, text)
        await callback.message.edit_text(f"Записал отзыв о «{app.label(tconst)}». Спасибо!")


@router.callback_query()
async def stale(callback: CallbackQuery) -> None:
    """Кнопка из старого сообщения, которая уже не к месту (например, шаг знакомства пройден).
    Должна стоять после всех остальных обработчиков кнопок — она ловит любую."""
    await callback.answer("Эта кнопка уже не работает. Посмотри /help.")


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
