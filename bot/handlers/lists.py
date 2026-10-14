"""Списки (/lists), любимые актёры и режиссёры (/favorites), справка (/help)."""
from aiogram import Bot, F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message
from aiogram.utils.keyboard import InlineKeyboardBuilder

from bot import texts
from bot.handlers.recommend import start_person_films
from bot.keyboards import ListCb, MovieCb, NavCb, PersonCb, status_kb
from bot.services import App
from bot.states import FavoriteSearch
from recsys.favorites import add_favorite, person_unseen, remove_favorite

router = Router()
LISTS = {"seen": "Видел", "planned": "В планах", "hidden": "Скрыто"}
LIST_LIMIT = 20
PERSON_FILMS = 30


@router.message(Command("help"))
async def help_(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer(texts.HELP)


@router.message(Command("lists"))
async def lists(message: Message, state: FSMContext) -> None:
    await state.clear()
    kb = InlineKeyboardBuilder()
    for name, label in LISTS.items():
        kb.button(text=label, callback_data=ListCb(name=name))
    await message.answer("Какой список показать?", reply_markup=kb.as_markup())


@router.callback_query(ListCb.filter())
async def show_list(callback: CallbackQuery, callback_data: ListCb, app: App) -> None:
    rows = app.conn.execute(
        "SELECT tconst FROM user_lists WHERE user_id = ? AND list = ? ORDER BY created_at DESC LIMIT ?",
        (callback.from_user.id, callback_data.name, LIST_LIMIT),
    ).fetchall()
    await callback.answer()
    if not rows:
        await callback.message.answer(f"Список «{LISTS[callback_data.name]}» пока пуст.")
        return
    ctx = "hid" if callback_data.name == "hidden" else "lst"
    kb = InlineKeyboardBuilder()
    for (tconst,) in rows:
        kb.button(text=app.label(tconst), callback_data=MovieCb(a="op", t=tconst, c=ctx))
    kb.adjust(1)
    await callback.message.answer(f"<b>{LISTS[callback_data.name]}</b> — нажми на фильм, чтобы открыть карточку:",
                                  reply_markup=kb.as_markup())


# --- любимые люди ---

FAVORITES_TEXT = ("<b>Любимые актёры и режиссёры</b>\n"
                  "Нажми на имя — покажу их фильмы, которых ты ещё не видел. ✕ — убрать из любимых.\n\n"
                  "Чтобы добавить, нажми кнопку ниже или просто напиши имя (можно по-русски).")
NO_FAVORITES_TEXT = ("Любимых актёров и режиссёров пока нет — их фильмы будут чаще попадать в подборку.\n\n"
                     "Напиши имя (можно по-русски) или нажми кнопку ниже.")


def favorites_kb(app: App, user_id: int):
    rows = app.conn.execute(
        """SELECT f.nconst, p.name FROM favorites f JOIN people p USING (nconst)
           WHERE f.user_id = ? AND f.added_by != 'declined' ORDER BY p.name""", (user_id,)).fetchall()
    kb = InlineKeyboardBuilder()
    kb.button(text="➕ Добавить актёра или режиссёра", callback_data=NavCb(a="fav_add"))
    for nconst, name in rows:
        kb.button(text=name, callback_data=PersonCb(a="films", n=nconst))
        kb.button(text="✕", callback_data=PersonCb(a="del", n=nconst))
    kb.adjust(1, *([2] * len(rows)))
    return kb.as_markup(), len(rows)


async def show_favorites(bot: Bot, user_id: int, state: FSMContext, app: App) -> None:
    """Список любимых; сразу ждём имя — добавить можно, просто написав его."""
    kb, n = favorites_kb(app, user_id)
    await state.set_state(FavoriteSearch.query)
    await bot.send_message(user_id, FAVORITES_TEXT if n else NO_FAVORITES_TEXT, reply_markup=kb)


@router.message(Command("favorites"))
async def favorites(message: Message, state: FSMContext, bot: Bot, app: App) -> None:
    await state.clear()
    await show_favorites(bot, message.from_user.id, state, app)


@router.callback_query(NavCb.filter(F.a == "fav_list"))
async def favorites_again(callback: CallbackQuery, state: FSMContext, bot: Bot, app: App) -> None:
    await callback.answer()
    await state.clear()
    await show_favorites(bot, callback.from_user.id, state, app)


@router.callback_query(NavCb.filter(F.a == "fav_add"))
async def favorite_add(callback: CallbackQuery, state: FSMContext) -> None:
    await state.set_state(FavoriteSearch.query)
    await callback.message.answer("Напиши имя актёра или режиссёра (можно по-русски).")
    await callback.answer()


@router.message(FavoriteSearch.query, F.text, ~F.text.startswith("/"))
async def favorite_search(message: Message, app: App) -> None:
    found = await app.search_people(message.text)
    if not found:
        await message.answer("Никого не нашёл. Попробуй иначе — например, фамилию латиницей.")
        return
    kb = InlineKeyboardBuilder()
    for p in found:
        kb.button(text=f"{p['name']} ({p['films']} фильм.)", callback_data=PersonCb(a="add", n=p["nconst"]))
    kb.adjust(1)
    await message.answer("Кого добавить? Если нужного нет — напиши имя иначе.", reply_markup=kb.as_markup())


@router.callback_query(PersonCb.filter(F.a == "add"))
async def favorite_pick(callback: CallbackQuery, callback_data: PersonCb, state: FSMContext, bot: Bot,
                        app: App) -> None:
    with app.conn:
        add_favorite(app.conn, callback.from_user.id, callback_data.n, "manual")
    name = app.conn.execute("SELECT name FROM people WHERE nconst = ?", (callback_data.n,)).fetchone()[0]
    await callback.message.edit_reply_markup(reply_markup=status_kb(f"В любимых: {name}"))
    await callback.answer()
    await show_favorites(bot, callback.from_user.id, state, app)


@router.callback_query(PersonCb.filter(F.a == "del"))
async def favorite_remove(callback: CallbackQuery, callback_data: PersonCb, app: App) -> None:
    with app.conn:
        remove_favorite(app.conn, callback.from_user.id, callback_data.n)
    kb, _ = favorites_kb(app, callback.from_user.id)
    await callback.message.edit_reply_markup(reply_markup=kb)
    await callback.answer("Убрал")


@router.callback_query(PersonCb.filter(F.a == "films"))
async def favorite_films(callback: CallbackQuery, callback_data: PersonCb, state: FSMContext, bot: Bot,
                         app: App) -> None:
    films = person_unseen(app.conn, callback.from_user.id, callback_data.n, limit=PERSON_FILMS)
    name = app.conn.execute("SELECT name FROM people WHERE nconst = ?", (callback_data.n,)).fetchone()[0]
    await callback.answer()
    if not films:
        await callback.message.answer(f"Все фильмы {name} из базы ты уже видел.")
        return
    await state.clear()
    await start_person_films(bot, callback.from_user.id, app, name, [f["tconst"] for f in films])
