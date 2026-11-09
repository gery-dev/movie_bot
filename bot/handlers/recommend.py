"""Подборка (/recommend) — одна карточка, которая листается на месте; опрос на сегодня (/today);
новинки (/new); ответы на карточках; отзыв и автопредложение любимых."""
from datetime import datetime
from html import escape

from aiogram import Bot, F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.filters.callback_data import CallbackData
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message
from aiogram.utils.keyboard import InlineKeyboardBuilder

from bot.cards import build_card, send_card, show
from bot.keyboards import (MovieCb, NavCb, PersonCb, REASONS, ReasonCb, STATUS, TodayCb, movie_kb, nav_kb, reason_kb,
                           status_kb, today_kb)
from bot.services import App, Carousel
from bot.states import Review, Today
from bot.texts import films_word
from recsys.recommender import NEW_YEARS
from recsys.favorites import add_favorite, suggest_favorites
from recsys.today import QUESTIONS, load_daily, load_last, save_daily, summary

router = Router()
BATCH = 10  # сколько фильмов считать за раз; дальше подборка догружается
QUERY_RESULTS = 20  # сколько фильмов показывать по запросу обычным текстом
QUESTION_TEXT = {
    "mood": "Какое настроение?", "time": "Сколько есть времени?", "company": "С кем смотришь?",
    "novelty": "Хочется нового или проверенного?", "language": "Русское или зарубежное?", "era": "Какая эпоха?",
}
REACTIONS = {"sn": "seen", "ni": "not_interested", "lt": "later"}
NOTHING_MORE = "Пока это всё. Оцени пару фильмов или смени настроение (/today) — и я подберу новые."


def new_since() -> int:
    """/new — фильмы за последние NEW_YEARS календарных лет, включая текущий."""
    return datetime.now().year - NEW_YEARS + 1


# --- подборка: /recommend — по вкусу, /today — по вкусу с приоритетом настроения ---

async def start_recommendations(bot: Bot, user_id: int, app: App, answers: dict | None = None) -> None:
    """answers=None — обычная подборка по вкусу; ответы /today — подборка под настроение."""
    df = app.rec.recommend(app.conn, user_id, BATCH, answers=answers)
    films = df.tconst.tolist()
    if not films:
        await bot.send_message(user_id, "Под эти условия ничего не нашлось. Попробуй другие ответы: /today",
                               reply_markup=nav_kb(("Пройти опрос заново", "today_again")))
        return
    await start_carousel(bot, user_id, app, Carousel(films=films, answers=answers, why=dict(zip(df.tconst, df.why))))


async def start_new(bot: Bot, user_id: int, app: App) -> None:
    """Новинки последних лет по вкусу — та же подборка, что /recommend, но только свежие фильмы."""
    df = app.rec.recommend(app.conn, user_id, BATCH, min_year=new_since())
    if df.empty:
        await bot.send_message(user_id, "Новинок под твой вкус пока не нашлось. Загляни позже или открой /recommend.")
        return
    await start_carousel(bot, user_id, app, Carousel(films=df.tconst.tolist(), kind="new",
                                                     why=dict(zip(df.tconst, df.why))))


async def start_person_films(bot: Bot, user_id: int, app: App, name: str, films: list[str]) -> None:
    """Фильмы любимого актёра или режиссёра, которых пользователь не видел, — тоже листаемой карточкой."""
    await start_carousel(bot, user_id, app, Carousel(films=films, person=name))


async def start_query(bot: Bot, user_id: int, app: App, text: str) -> None:
    """Поиск обычным текстом: «как „Интерстеллар“, но повеселее»."""
    df, parsed = app.rec.search(app.conn, user_id, text, k=QUERY_RESULTS)
    search_id = app.log_search(user_id, "text", text, len(df))
    if df.empty:
        await bot.send_message(user_id, "Ничего не нашёл под такой запрос. Попробуй сформулировать иначе.")
        return
    await start_carousel(bot, user_id, app, Carousel(films=df.tconst.tolist(), query=text[:60], search_id=search_id,
                                                     why=dict(zip(df.tconst, df.why))))


async def start_carousel(bot: Bot, user_id: int, app: App, carousel: Carousel) -> None:
    old = app.carousels.get(user_id)
    if old and old.message_id:  # у прежней подборки убираем кнопки, листать можно только новую
        await _drop_keyboard(bot, user_id, old.message_id)
    app.carousels[user_id] = carousel
    await render(bot, user_id, app, new_message=True)


def nothing_more(c: Carousel) -> str:
    if c.group:
        return "Это все фильмы для компании. Обновить подборку — /together."
    if c.query:
        return "Это все варианты по запросу. Можно написать другой — просто текстом."
    if c.person:
        return f"Это все фильмы {c.person}, которых ты ещё не видел."
    return NOTHING_MORE


async def render(bot: Bot, user_id: int, app: App, new_message: bool = False, keyboard_only: bool = False) -> None:
    """Показать текущий фильм подборки: на месте или новым сообщением (после отзыва).
    keyboard_only — фильм тот же, поменялись только кнопки (после оценки)."""
    c = app.carousels[user_id]
    t = c.current
    if c.group:
        note = f"Для компании: {c.group} · фильм {c.idx + 1} из {len(c.films)}"
    elif c.query:
        note = f"По запросу «{c.query}» · фильм {c.idx + 1} из {len(c.films)}"
    elif c.person:
        note = f"{c.person} · фильм {c.idx + 1} из {len(c.films)}"
    elif c.kind == "new":
        note = f"Новинки с {new_since()} года по твоему вкусу · фильм {c.idx + 1}"
    elif c.answers is None:
        note = f"Подборка по твоему вкусу · фильм {c.idx + 1}"
        note += " · под настроение — /today" if c.idx == 0 else ""
    else:
        note = f"Под настроение · фильм {c.idx + 1}"
        note += f" · {summary(c.answers)}" if c.idx == 0 else ""
    if t in c.why:
        note += "\n" + c.why[t]
    card = await build_card(app, t, note)
    has_next = not c.fixed or c.idx + 1 < len(c.films)  # готовые списки не догружаются
    nav = ([("← Назад", "prev")] if c.idx > 0 else []) + ([("Далее →", "next")] if has_next else [])
    if c.person:
        extra = [("К списку любимых", "fav_list")]
    elif c.group or c.query:
        extra = None
    else:
        extra = [("Другое настроение", "today_again")] if c.answers is not None else None
    if t in c.statuses:
        kb = status_kb(c.statuses[t], card.trailer, card.watch, nav, extra)
    else:
        kb = movie_kb(t, "rec", card.trailer, card.watch, nav, extra)
    if keyboard_only and c.message_id:
        await bot.edit_message_reply_markup(chat_id=user_id, message_id=c.message_id, reply_markup=kb)
        return
    old_id = c.message_id
    c.message_id = await show(bot, user_id, card, kb, None if new_message else old_id)
    if new_message and old_id and old_id != c.message_id:
        await _drop_keyboard(bot, user_id, old_id)
    if t not in c.logged and not (c.person or c.group):  # фильмы человека — не рекомендация, в журнал показов не пишем
        mode = c.kind or ("query" if c.query else "today" if c.answers is not None else "recommend")
        app.log_shown(user_id, [t], mode, start_position=c.idx + 1, mood=c.answers)
        c.logged.add(t)


async def advance(bot: Bot, user_id: int, app: App, new_message: bool = False) -> bool:
    """Следующий фильм; в конце подборки догружаем ещё. False — предложить больше нечего."""
    c = app.carousels[user_id]
    if c.idx + 1 >= len(c.films) and not c.fixed:
        more = app.rec.recommend(app.conn, user_id, len(c.films) + BATCH, answers=c.answers,
                                 min_year=new_since() if c.kind == "new" else None)
        c.why.update(dict(zip(more.tconst, more.why)))
        c.films += [t for t in more.tconst if t not in c.films]
    if c.idx + 1 >= len(c.films):
        return False
    c.idx += 1
    await render(bot, user_id, app, new_message)
    return True


async def _drop_keyboard(bot: Bot, chat_id: int, message_id: int) -> None:
    try:
        await bot.edit_message_reply_markup(chat_id=chat_id, message_id=message_id, reply_markup=None)
    except TelegramBadRequest:
        pass


def _carousel_of(callback: CallbackQuery, app: App) -> Carousel | None:
    """Подборка, к которой относится нажатая кнопка (None — кнопка из старого сообщения)."""
    c = app.carousels.get(callback.from_user.id)
    return c if c and c.message_id == callback.message.message_id else None


@router.message(Command("recommend"))
async def recommend(message: Message, state: FSMContext, bot: Bot, app: App) -> None:
    await state.clear()
    await start_recommendations(bot, message.from_user.id, app)


@router.message(Command("new"))
async def new_films(message: Message, state: FSMContext, bot: Bot, app: App) -> None:
    await state.clear()
    await start_new(bot, message.from_user.id, app)


@router.callback_query(NavCb.filter(F.a.in_({"next", "prev"})))
async def navigate(callback: CallbackQuery, callback_data: NavCb, bot: Bot, app: App) -> None:
    c = _carousel_of(callback, app)
    if c is None:
        await callback.answer("Эта подборка устарела — открой новую: /recommend или /today")
        return
    await callback.answer()
    if callback_data.a == "prev":
        c.idx = max(c.idx - 1, 0)
        await render(bot, callback.from_user.id, app)
    elif not await advance(bot, callback.from_user.id, app):
        await bot.send_message(callback.from_user.id, nothing_more(c))


# --- /today: опрос в одном сообщении, затем подборка под настроение ---

@router.message(Command("today"))
async def today(message: Message, state: FSMContext, bot: Bot, app: App) -> None:
    """Если ответы ещё действуют — сразу подборка (кнопка «Другое настроение» на карточке);
    если устарели — опрос заново, с кнопкой «Как в прошлый раз»."""
    await state.clear()
    await open_today(bot, message.from_user.id, state, app)


async def open_today(bot: Bot, user_id: int, state: FSMContext, app: App) -> None:
    current = load_daily(app.conn, user_id)
    if current is not None:
        await start_recommendations(bot, user_id, app, answers=current)
    else:
        await start_survey(bot, user_id, state, load_last(app.conn, user_id))


async def start_survey(bot: Bot, user_id: int, state: FSMContext, last: dict | None = None) -> None:
    await state.set_state(Today.asking)
    await state.update_data(answers={})
    first = next(iter(QUESTIONS))
    same = [(f"Как в прошлый раз: {summary(last)}", "today_same")] if last else None
    await bot.send_message(user_id, QUESTION_TEXT[first], reply_markup=today_kb(first, back=False, extra=same))


async def _ask(callback: CallbackQuery, answers: dict) -> None:
    rest = [q for q in QUESTIONS if q not in answers]
    await callback.message.edit_text(QUESTION_TEXT[rest[0]], reply_markup=today_kb(rest[0], back=bool(answers)))


async def _finish_survey(callback: CallbackQuery, state: FSMContext, bot: Bot, app: App, answers: dict,
                         same: bool = False) -> None:
    with app.conn:
        save_daily(app.conn, callback.from_user.id, answers)
    await state.clear()
    await callback.message.edit_text(f"Сегодня: {summary(answers)}" + (" (как в прошлый раз)" if same else "")
                                     + ". Подбираю…")
    await start_recommendations(bot, callback.from_user.id, app, answers=answers)


@router.callback_query(Today.asking, TodayCb.filter())
async def today_answer(callback: CallbackQuery, callback_data: TodayCb, state: FSMContext, bot: Bot,
                       app: App) -> None:
    answers = (await state.get_data()).get("answers", {})
    if callback_data.q in answers:
        await callback.answer("Ответ уже есть")
        return
    answers[callback_data.q] = callback_data.v
    await state.update_data(answers=answers)
    await callback.answer()
    if len(answers) < len(QUESTIONS):
        await _ask(callback, answers)
    else:
        await _finish_survey(callback, state, bot, app, answers)


@router.callback_query(Today.asking, NavCb.filter(F.a == "today_same"))
async def today_same(callback: CallbackQuery, state: FSMContext, bot: Bot, app: App) -> None:
    last = load_last(app.conn, callback.from_user.id)
    await callback.answer()
    if last:
        await _finish_survey(callback, state, bot, app, last, same=True)


@router.callback_query(Today.asking, NavCb.filter(F.a == "today_back"))
async def today_back(callback: CallbackQuery, state: FSMContext) -> None:
    answers = (await state.get_data()).get("answers", {})
    if answers:
        answers.pop(list(answers)[-1])
        await state.update_data(answers=answers)
    await callback.answer()
    await _ask(callback, answers)


@router.callback_query(NavCb.filter(F.a == "today_again"))
async def today_again(callback: CallbackQuery, state: FSMContext, bot: Bot) -> None:
    """«Другое настроение»: опрос заново, не дожидаясь, пока прежние ответы устареют."""
    await callback.answer()
    await state.clear()
    await start_survey(bot, callback.from_user.id, state)


# --- ответы на карточках ---

@router.callback_query(MovieCb.filter(F.a == "op"))
async def open_card(callback: CallbackQuery, callback_data: MovieCb, bot: Bot, app: App) -> None:
    await send_card(bot, callback.from_user.id, app, callback_data.t, callback_data.c)
    await callback.answer()


@router.callback_query(MovieCb.filter(F.c != "cal"))
async def card_answer(callback: CallbackQuery, callback_data: MovieCb, state: FSMContext, bot: Bot,
                      app: App) -> None:
    user, t, a = callback.from_user.id, callback_data.t, callback_data.a
    if a == "r":
        app.rate(user, t, callback_data.v, "recommendation")
    elif a in REACTIONS:
        app.react(user, t, REACTIONS[a])
    elif a in ("hd", "uh"):
        app.hide(user, t, hidden=a == "hd")
    else:
        await callback.answer()
        return
    status = STATUS[a].format(v=callback_data.v)
    await callback.answer("Записал")
    c = _carousel_of(callback, app)
    if c is not None and c.search_id:  # поиск текстом дал результат, на который человек ответил
        app.mark_search_chosen(user, "text", c.search_id)
    if c is not None and c.current == t:
        c.statuses[t] = status
    if a == "r":  # после оценки — вопрос об отзыве, потом следующий фильм
        if c is not None:
            await render(bot, user, app, keyboard_only=True)
        else:
            await callback.message.edit_reply_markup(reply_markup=status_kb(status))
        await state.set_state(Review.text)
        await state.update_data(tconst=t, in_carousel=c is not None)
        await bot.send_message(user, f"Пара слов о «{app.label(t)}»? Напиши отзыв или нажми «Пропустить».",
                               reply_markup=nav_kb(("Пропустить", "review_skip")))
        return
    if c is not None:  # «не интересно», «позже», «скрыть» — сразу следующий фильм
        if not await advance(bot, user, app):
            await render(bot, user, app)
            await bot.send_message(user, nothing_more(c))
    else:
        await callback.message.edit_reply_markup(reply_markup=status_kb(status))
    if a == "ni":
        await ask_reason(bot, user, app, t)


# --- необязательный вопрос «Почему не интересно?» ---

async def ask_reason(bot: Bot, user_id: int, app: App, tconst: str) -> None:
    """Отдельным сообщением под подборкой, чтобы не тормозить листание. Вопрос без ответа убирается,
    когда задаём следующий: в чате не копятся старые вопросы."""
    old = app.reason_msgs.pop(user_id, None)
    if old:
        try:
            await bot.delete_message(chat_id=user_id, message_id=old)
        except TelegramBadRequest:
            pass
    msg = await bot.send_message(user_id, f"Почему «{escape(app.label(tconst))}» не подошёл? Можно не отвечать.",
                                 reply_markup=reason_kb(tconst))
    app.reason_msgs[user_id] = msg.message_id


@router.callback_query(ReasonCb.filter())
async def reason_answer(callback: CallbackQuery, callback_data: ReasonCb, app: App) -> None:
    user = callback.from_user.id
    if callback_data.r in REASONS:
        app.set_reason(user, callback_data.t, callback_data.r)
        if app.reason_msgs.get(user) == callback.message.message_id:
            del app.reason_msgs[user]
        await callback.message.edit_reply_markup(reply_markup=status_kb(f"Не интересно: {REASONS[callback_data.r].lower()}"))
    await callback.answer("Спасибо!")


# --- отзыв, затем продолжение ---

async def after_review(bot: Bot, user_id: int, app: App, in_carousel: bool) -> None:
    """Что дальше после отзыва: предложить любимых, показать следующий фильм подборки и при случае — опрос.
    Опрос последним сообщением: если он выше новой карточки, его не замечают."""
    await offer_favorite(bot, user_id, app)
    if in_carousel and user_id in app.carousels:
        if not await advance(bot, user_id, app, new_message=True):
            await bot.send_message(user_id, nothing_more(app.carousels[user_id]))
    else:
        await bot.send_message(
            user_id, "Подобрать тебе ещё фильмы?",
            reply_markup=nav_kb(("Да, по моим оценкам и любимым фильмам", "menu_recommend"),
                                ("Да, под настроение (спрошу пару вопросов)", "menu_today")),
        )
    await offer_survey(bot, user_id, app)


@router.message(Review.text, F.text, ~F.text.startswith("/"))
async def review(message: Message, state: FSMContext, bot: Bot, app: App) -> None:
    data = await state.get_data()
    if data.get("tconst"):
        app.save_review(message.from_user.id, data["tconst"], message.text)
    await state.clear()
    await message.answer("Спасибо за отзыв!")
    await after_review(bot, message.from_user.id, app, data.get("in_carousel", False))


@router.callback_query(Review.text, NavCb.filter(F.a == "review_skip"))
async def review_skip(callback: CallbackQuery, state: FSMContext, bot: Bot, app: App) -> None:
    data = await state.get_data()
    await state.clear()
    await callback.message.edit_reply_markup(reply_markup=status_kb("Без отзыва"))
    await callback.answer()
    await after_review(bot, callback.from_user.id, app, data.get("in_carousel", False))


@router.callback_query(NavCb.filter(F.a.in_({"menu_recommend", "menu_today"})))
async def menu(callback: CallbackQuery, callback_data: NavCb, state: FSMContext, bot: Bot, app: App) -> None:
    await callback.answer()
    await state.clear()
    if callback_data.a == "menu_recommend":
        await start_recommendations(bot, callback.from_user.id, app)
    else:
        await open_today(bot, callback.from_user.id, state, app)


# --- опрос «насколько подходят подборки» (онлайн-тест, этап 6) ---

SURVEY_WAVES = (3, 20)  # спросить после 3 и после 20 оценок фильмов из подборок
SURVEY_LABELS = {1: "1 — совсем мимо", 2: "2 — скорее мимо", 3: "3 — так себе", 4: "4 — хорошо", 5: "5 — в точку"}


class SurveyCb(CallbackData, prefix="sv"):
    w: int
    s: int


def survey_due(app: App, user_id: int) -> int | None:
    rated = app.conn.execute("SELECT count(*) FROM ratings WHERE user_id = ? AND source = 'recommendation'",
                             (user_id,)).fetchone()[0]
    done = {r[0] for r in app.conn.execute("SELECT wave FROM survey WHERE user_id = ?", (user_id,))}
    done |= {w for u, w in app.survey_offered if u == user_id}  # уже спросили — не повторяем после каждой оценки
    due = [w for w in SURVEY_WAVES if rated >= w and w not in done]
    return due[-1] if due else None


async def offer_survey(bot: Bot, user_id: int, app: App) -> None:
    wave = survey_due(app, user_id)
    if wave is None:
        return
    app.survey_offered.add((user_id, wave))
    kb = InlineKeyboardBuilder()
    for score, label in SURVEY_LABELS.items():
        kb.button(text=label, callback_data=SurveyCb(w=wave, s=score))
    kb.adjust(1)
    await bot.send_message(user_id, f"Уже {wave} оценок фильмов из подборок — спасибо! Насколько подборки тебе подходят?",
                           reply_markup=kb.as_markup())
    app.log_survey_offer(user_id, wave)  # после отправки: если бота заблокировали, предложения не было


@router.callback_query(SurveyCb.filter())
async def survey_answer(callback: CallbackQuery, callback_data: SurveyCb, app: App) -> None:
    with app.conn:
        if not app.conn.execute("SELECT 1 FROM survey WHERE user_id = ? AND wave = ?",
                                (callback.from_user.id, callback_data.w)).fetchone():
            app.conn.execute("INSERT INTO survey (user_id, wave, score) VALUES (?, ?, ?)",
                             (callback.from_user.id, callback_data.w, callback_data.s))
    await callback.message.edit_reply_markup(reply_markup=status_kb(SURVEY_LABELS[callback_data.s]))
    await callback.answer("Спасибо!")


# --- автопредложение любимых ---

async def offer_favorite(bot: Bot, user_id: int, app: App) -> None:
    suggestions = suggest_favorites(app.conn, user_id)
    if not suggestions:
        return
    s = suggestions[0]
    who = "режиссёра" if s["role"] == "director" else "с участием"
    kb = InlineKeyboardBuilder()
    kb.button(text="Да, добавить", callback_data=PersonCb(a="yes", n=s["nconst"]))
    kb.button(text="Нет", callback_data=PersonCb(a="no", n=s["nconst"]))
    await bot.send_message(
        user_id, f"Тебе понравились {s['liked']} {films_word(s['liked'])} {who} {s['name']}. Добавить в любимые?",
        reply_markup=kb.as_markup(),
    )


@router.callback_query(PersonCb.filter(F.a.in_({"yes", "no"})))
async def favorite_answer(callback: CallbackQuery, callback_data: PersonCb, app: App) -> None:
    yes = callback_data.a == "yes"
    with app.conn:
        add_favorite(app.conn, callback.from_user.id, callback_data.n, "suggestion" if yes else "declined")
    await callback.message.edit_reply_markup(reply_markup=status_kb("Добавлен в любимые" if yes else "Не добавлять"))
    await callback.answer()
