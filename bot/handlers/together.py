"""Кино на компанию (/together): компания, ссылка-приглашение, подборка, которая подходит всем."""
from aiogram import Bot, F, Router
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.filters.callback_data import CallbackData
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message, User
from aiogram.utils.keyboard import InlineKeyboardBuilder

from bot.handlers.recommend import start_carousel
from bot.services import App, Carousel
from recsys.profile import ensure_user

router = Router()
GROUP_FILMS = 20
MAX_MEMBERS = 8


class TogetherCb(CallbackData, prefix="tg"):
    a: str  # pick | new | leave
    g: int


def remember_name(app: App, user: User) -> None:
    ensure_user(app.conn, user.id)
    app.conn.execute("INSERT OR REPLACE INTO user_names (user_id, name) VALUES (?, ?)",
                     (user.id, user.first_name or "Без имени"))


def current_group(app: App, user_id: int) -> int | None:
    row = app.conn.execute(
        """SELECT g.group_id FROM group_members m JOIN groups g USING (group_id)
           WHERE m.user_id = ? ORDER BY g.group_id DESC LIMIT 1""", (user_id,)).fetchone()
    return row[0] if row else None


def members_of(app: App, group_id: int) -> list[tuple[int, str]]:
    return [(u, n or "участник") for u, n in app.conn.execute(
        """SELECT m.user_id, n.name FROM group_members m LEFT JOIN user_names n USING (user_id)
           WHERE m.group_id = ? ORDER BY m.rowid""", (group_id,))]


def create_group(app: App, user: User) -> int:
    with app.conn:
        remember_name(app, user)
        gid = app.conn.execute("INSERT INTO groups (owner_id) VALUES (?)", (user.id,)).lastrowid
        app.conn.execute("INSERT INTO group_members (group_id, user_id) VALUES (?, ?)", (gid, user.id))
    return gid


async def show_panel(bot: Bot, user_id: int, app: App, group_id: int) -> None:
    me = await bot.get_me()
    link = f"https://t.me/{me.username}?start=join_{group_id}"
    names = ", ".join(n for _, n in members_of(app, group_id))
    text = (f"<b>Кино на компанию</b>\nСейчас в компании: {names}.\n\n"
            f"Позови друзей — отправь им ссылку, по ней они присоединятся:\n{link}\n\n"
            "Когда все в сборе, жми «Подобрать фильм для всех»: я найду фильмы, которые не разочаруют никого, "
            "и уберу те, что кто-то уже видел.")
    kb = InlineKeyboardBuilder()
    kb.button(text="Подобрать фильм для всех", callback_data=TogetherCb(a="pick", g=group_id))
    kb.button(text="Новая компания", callback_data=TogetherCb(a="new", g=group_id))
    kb.button(text="Выйти из компании", callback_data=TogetherCb(a="leave", g=group_id))
    kb.adjust(1)
    await bot.send_message(user_id, text, reply_markup=kb.as_markup(), disable_web_page_preview=True)


@router.message(Command("together"))
async def together(message: Message, state: FSMContext, bot: Bot, app: App) -> None:
    await state.clear()
    gid = current_group(app, message.from_user.id) or create_group(app, message.from_user)
    await show_panel(bot, message.from_user.id, app, gid)


@router.message(CommandStart(deep_link=True, magic=F.args.regexp(r"^join_\d+$")))
async def join(message: Message, command: CommandObject, state: FSMContext, bot: Bot, app: App) -> None:
    """Переход по ссылке-приглашению t.me/<бот>?start=join_<номер компании>."""
    await state.clear()
    gid = int(command.args.split("_", 1)[1])
    if not app.conn.execute("SELECT 1 FROM groups WHERE group_id = ?", (gid,)).fetchone():
        await message.answer("Эта компания уже не существует. Создай свою: /together")
        return
    user = message.from_user
    members = members_of(app, gid)
    if user.id not in [u for u, _ in members]:
        if len(members) >= MAX_MEMBERS:
            await message.answer(f"В компании уже {MAX_MEMBERS} человек — больше нельзя.")
            return
        with app.conn:
            remember_name(app, user)
            app.conn.execute("INSERT INTO group_members (group_id, user_id) VALUES (?, ?)", (gid, user.id))
        for uid, _ in members:
            await bot.send_message(uid, f"К компании присоединяется {user.first_name}. Подобрать фильм для всех — /together")
    onboarded = app.conn.execute("SELECT onboarding_done FROM users WHERE user_id = ?", (user.id,)).fetchone()[0]
    if not onboarded:
        await message.answer("Ты в компании! Чтобы я учитывал и твой вкус, пройди короткое знакомство — /start. "
                             "Без него буду считать, что тебе нравятся просто хорошие известные фильмы.")
    await show_panel(bot, user.id, app, gid)


@router.callback_query(TogetherCb.filter())
async def panel_action(callback: CallbackQuery, callback_data: TogetherCb, bot: Bot, app: App) -> None:
    user, gid = callback.from_user, callback_data.g
    members = members_of(app, gid)
    if user.id not in [u for u, _ in members]:
        await callback.answer("Ты уже не в этой компании")
        return
    await callback.answer()
    if callback_data.a == "new":
        await show_panel(bot, user.id, app, create_group(app, user))
    elif callback_data.a == "leave":
        with app.conn:
            app.conn.execute("DELETE FROM group_members WHERE group_id = ? AND user_id = ?", (gid, user.id))
        await callback.message.edit_text("Готово, тебя больше нет в этой компании. Создать новую — /together")
    elif len(members) < 2:
        await bot.send_message(user.id, "Пока в компании только ты — отправь друзьям ссылку из сообщения выше.")
    else:
        ids, names = [u for u, _ in members], dict(members)
        df = app.rec.recommend_group(app.conn, ids, GROUP_FILMS)
        why = {t: f"Почему: подходит всем; больше всего — {names[b]}" for t, b in zip(df.tconst, df.best)}
        label = ", ".join(names.values())
        await start_carousel(bot, user.id, app, Carousel(films=df.tconst.tolist(), group=label, why=why))
