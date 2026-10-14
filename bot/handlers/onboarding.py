"""Знакомство (/start): жанры → любимые фильмы → любимые люди → калибровка на 10 фильмах.
На каждом шаге можно вернуться назад; калибровка идёт в одном сообщении, «← Назад» отменяет
последний ответ."""
import numpy as np
from aiogram import Bot, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message
from aiogram.utils.keyboard import InlineKeyboardBuilder

from bot import texts
from bot.cards import build_card, show
from bot.handlers.recommend import start_recommendations
from bot.keyboards import GenreCb, MovieCb, NavCb, PersonCb, PickCb, genres_kb, movie_kb, nav_kb, status_kb
from bot.services import App
from bot.states import Calibration, Onboarding
from recsys import calibration
from recsys.favorites import add_favorite
from recsys.profile import ensure_user, excluded_films

router = Router()
NOT_COMMAND = ~F.text.startswith("/")
FILMS_NAV = (("Следующий этап: актёры и режиссёры →", "films_done"), ("← Назад к жанрам", "back_genres"))
PEOPLE_NAV = (("Следующий этап: калибровка →", "people_done"), ("← Назад к фильмам", "back_films"))


@router.message(CommandStart())
async def start(message: Message, state: FSMContext, app: App) -> None:
    """Новому — знакомство. Тому, кто его уже прошёл, /start ничего не сбрасывает (его легко нажать
    случайно): только приветствие и команды. Пройти знакомство заново — /restart."""
    with app.conn:
        ensure_user(app.conn, message.from_user.id)
    done = app.conn.execute("SELECT onboarding_done FROM users WHERE user_id = ?",
                            (message.from_user.id,)).fetchone()[0]
    if done:
        await state.clear()
        await message.answer(texts.WELCOME_BACK)
        return
    await begin_onboarding(message, state)


@router.message(Command("restart"))
async def restart(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer(texts.RESTART_CONFIRM, reply_markup=nav_kb(("Да, пройти заново", "restart_yes"),
                                                                     ("Отмена", "restart_no")))


@router.callback_query(NavCb.filter(F.a.in_({"restart_yes", "restart_no"})))
async def restart_answer(callback: CallbackQuery, callback_data: NavCb, state: FSMContext) -> None:
    await callback.answer()
    if callback_data.a == "restart_no":
        await callback.message.edit_text("Хорошо, всё остаётся как было.")
        return
    await callback.message.edit_reply_markup(reply_markup=None)
    await begin_onboarding(callback.message, state)


async def begin_onboarding(message: Message, state: FSMContext) -> None:
    await state.clear()
    await state.set_state(Onboarding.genres)
    await state.update_data(genres=[])
    await message.answer(texts.WELCOME, reply_markup=genres_kb(set()))


# --- шаг 1: жанры ---

@router.callback_query(Onboarding.genres, GenreCb.filter())
async def toggle_genre(callback: CallbackQuery, callback_data: GenreCb, state: FSMContext) -> None:
    selected = set((await state.get_data()).get("genres", []))
    selected ^= {callback_data.g}
    await state.update_data(genres=sorted(selected))
    await callback.message.edit_reply_markup(reply_markup=genres_kb(selected))
    await callback.answer()


@router.callback_query(Onboarding.genres, NavCb.filter(F.a == "genres_done"))
async def genres_done(callback: CallbackQuery, state: FSMContext) -> None:
    genres = (await state.get_data()).get("genres", [])
    chosen = ", ".join(texts.GENRES_RU[g] for g in genres) or "без предпочтений"
    await callback.message.edit_reply_markup(reply_markup=status_kb(f"Жанры: {chosen}"))
    await state.set_state(Onboarding.films)
    await callback.message.answer(texts.ASK_FILMS, reply_markup=nav_kb(*FILMS_NAV))
    await callback.answer()


@router.callback_query(Onboarding.films, NavCb.filter(F.a == "back_genres"))
async def back_to_genres(callback: CallbackQuery, state: FSMContext) -> None:
    selected = set((await state.get_data()).get("genres", []))
    await state.set_state(Onboarding.genres)
    await callback.message.edit_reply_markup(reply_markup=None)
    await callback.message.answer("Поправь жанры и нажми «Готово».", reply_markup=genres_kb(selected))
    await callback.answer()


# --- шаг 2: любимые фильмы ---

@router.message(Onboarding.films, F.text, NOT_COMMAND)
async def search_film(message: Message, app: App) -> None:
    found = app.search_movies(message.text)
    if not found:
        await message.answer("Не нашёл такой фильм. Попробуй иначе — можно по-английски.",
                             reply_markup=nav_kb(*FILMS_NAV))
        return
    kb = InlineKeyboardBuilder()
    for tconst, label in found:
        kb.button(text=label, callback_data=PickCb(t=tconst))
    for text, action in FILMS_NAV:
        kb.button(text=text, callback_data=NavCb(a=action))
    kb.adjust(1)
    await message.answer("Какой из них?", reply_markup=kb.as_markup())


@router.callback_query(Onboarding.films, PickCb.filter())
async def pick_film(callback: CallbackQuery, callback_data: PickCb, app: App) -> None:
    app.rate(callback.from_user.id, callback_data.t, 5, "onboarding")
    await callback.message.edit_reply_markup(reply_markup=status_kb(f"Добавил: {app.label(callback_data.t)}"))
    await callback.message.answer("Ещё один фильм? Напиши название или переходи к следующему этапу.",
                                  reply_markup=nav_kb(*FILMS_NAV))
    await callback.answer()


@router.callback_query(Onboarding.films, NavCb.filter(F.a == "films_done"))
async def films_done(callback: CallbackQuery, state: FSMContext) -> None:
    await state.set_state(Onboarding.people)
    await callback.message.edit_reply_markup(reply_markup=None)
    await callback.message.answer(texts.ASK_PEOPLE, reply_markup=nav_kb(*PEOPLE_NAV))
    await callback.answer()


@router.callback_query(Onboarding.people, NavCb.filter(F.a == "back_films"))
async def back_to_films(callback: CallbackQuery, state: FSMContext) -> None:
    await state.set_state(Onboarding.films)
    await callback.message.edit_reply_markup(reply_markup=None)
    await callback.message.answer(texts.ASK_FILMS, reply_markup=nav_kb(*FILMS_NAV))
    await callback.answer()


# --- шаг 3: любимые люди ---

@router.message(Onboarding.people, F.text, NOT_COMMAND)
async def search_person(message: Message, app: App) -> None:
    found = await app.search_people(message.text)
    if not found:
        await message.answer("Никого не нашёл. Попробуй иначе — например, фамилию латиницей.",
                             reply_markup=nav_kb(*PEOPLE_NAV))
        return
    kb = InlineKeyboardBuilder()
    for p in found:
        kb.button(text=f"{p['name']} ({p['films']} фильм.)", callback_data=PersonCb(a="add", n=p["nconst"]))
    for text, action in PEOPLE_NAV:
        kb.button(text=text, callback_data=NavCb(a=action))
    kb.adjust(1)
    await message.answer("Кого добавить?", reply_markup=kb.as_markup())


@router.callback_query(Onboarding.people, PersonCb.filter(F.a == "add"))
async def pick_person(callback: CallbackQuery, callback_data: PersonCb, app: App) -> None:
    with app.conn:
        add_favorite(app.conn, callback.from_user.id, callback_data.n, "manual")
    name = app.conn.execute("SELECT name FROM people WHERE nconst = ?", (callback_data.n,)).fetchone()[0]
    await callback.message.edit_reply_markup(reply_markup=status_kb(f"В любимых: {name}"))
    await callback.message.answer("Ещё кого-нибудь? Напиши имя или переходи к следующему этапу.",
                                  reply_markup=nav_kb(*PEOPLE_NAV))
    await callback.answer()


@router.callback_query(Onboarding.people, NavCb.filter(F.a == "people_done"))
async def people_done(callback: CallbackQuery, state: FSMContext, bot: Bot, app: App) -> None:
    genres = (await state.get_data()).get("genres", [])
    await state.set_state(Calibration.rating)
    # answers — коды ответов по порядку (r / ns / sk): нужны, чтобы «Назад» вернул счётчик «не смотрел»
    await state.update_data(shown=[], answers=[], genres=genres, msg=None, current=None)
    await callback.message.edit_reply_markup(reply_markup=None)
    await callback.message.answer(texts.CALIBRATION_START)
    await callback.answer()
    await show_calibration(bot, callback.from_user.id, state, app)


# --- шаг 4: калибровка в одном сообщении ---

async def show_calibration(bot: Bot, user_id: int, state: FSMContext, app: App, tconst: str | None = None) -> None:
    """Показать фильм калибровки (новый или переданный — при «Назад») на месте прежнего."""
    data = await state.get_data()
    if tconst is None:
        skip = [t for t in excluded_films(app.conn, user_id) if t in app.pool.index]
        not_seen = data["answers"].count("ns")
        tconst = calibration.next_film(app.pool, data["shown"] + skip, not_seen, set(data["genres"]),
                                       np.random.default_rng())
    n = len(data["shown"])
    card = await build_card(app, tconst, note=f"Фильм {n + 1} из {calibration.CALIBRATION_SIZE}")
    nav = [("← Назад", "cal_back")] if n else None
    msg = await show(bot, user_id, card, movie_kb(tconst, "cal", card.trailer, card.watch, nav), data.get("msg"))
    await state.update_data(current=tconst, msg=msg)


@router.callback_query(MovieCb.filter(F.c == "cal"))
async def calibration_answer(callback: CallbackQuery, callback_data: MovieCb, state: FSMContext, bot: Bot,
                             app: App) -> None:
    data = await state.get_data()
    if await state.get_state() != Calibration.rating.state or data.get("current") != callback_data.t:
        await callback.answer("Этот ответ уже записан")
        return
    user, t = callback.from_user.id, callback_data.t
    if callback_data.a == "r":
        app.rate(user, t, callback_data.v, "calibration")
    else:
        app.react(user, t, "not_seen" if callback_data.a == "ns" else "skip")
    await callback.answer()
    await state.update_data(shown=data["shown"] + [t], answers=data["answers"] + [callback_data.a], current=None)
    if len(data["shown"]) + 1 < calibration.CALIBRATION_SIZE:
        await show_calibration(bot, user, state, app)
        return
    with app.conn:
        app.conn.execute("UPDATE users SET onboarding_done = 1, calibration_done = 1 WHERE user_id = ?", (user,))
    await state.clear()
    await callback.message.edit_reply_markup(reply_markup=status_kb("Калибровка завершена"))
    await bot.send_message(user, texts.CALIBRATION_DONE)
    await start_recommendations(bot, user, app)


@router.callback_query(Calibration.rating, NavCb.filter(F.a == "cal_back"))
async def calibration_back(callback: CallbackQuery, state: FSMContext, bot: Bot, app: App) -> None:
    data = await state.get_data()
    if not data["shown"] or callback.message.message_id != data.get("msg"):
        await callback.answer()
        return
    prev = data["shown"][-1]
    app.undo_calibration(callback.from_user.id, prev)
    await state.update_data(shown=data["shown"][:-1], answers=data["answers"][:-1])
    await callback.answer("Ответ отменён")
    await show_calibration(bot, callback.from_user.id, state, app, tconst=prev)
