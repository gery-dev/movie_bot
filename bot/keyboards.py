"""Инлайн-кнопки. Данные кнопки ограничены 64 байтами, поэтому имена полей короткие."""
from aiogram.filters.callback_data import CallbackData
from aiogram.types import InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder

from bot.texts import GENRES_RU, START_GENRES
from recsys.today import QUESTIONS


class MovieCb(CallbackData, prefix="m"):
    """a — действие, t — фильм, v — оценка, c — где показана карточка:
    cal — калибровка, rec — подборка, fb — вопрос «посмотрел?», lst — из списка, hid — из скрытых."""
    a: str
    t: str
    v: int = 0
    c: str = "rec"


class GenreCb(CallbackData, prefix="g"):
    g: str


class PickCb(CallbackData, prefix="pk"):
    t: str


class PersonCb(CallbackData, prefix="p"):
    a: str  # add | del | films | yes | no
    n: str


class TodayCb(CallbackData, prefix="td"):
    q: str
    v: str


class ListCb(CallbackData, prefix="l"):
    name: str


class NavCb(CallbackData, prefix="n"):
    a: str


class UndoCb(CallbackData, prefix="u"):
    """Отмена выбора: k — что отменяем (film — любимый фильм в знакомстве, person — любимый человек
    в знакомстве, fav — любимый человек в /favorites), i — ID фильма или человека."""
    k: str
    i: str


class ReasonCb(CallbackData, prefix="rs"):
    """Причина отказа «Не интересно»: t — фильм, r — код из REASONS."""
    t: str
    r: str


# «Уже видел» отдельной кнопки нет: оценка и так значит, что фильм просмотрен
ACTIONS = {"ni": "Не интересно", "lt": "Посмотрю позже", "hd": "Скрыть навсегда"}
# необязательный вопрос после «Не интересно» (ТЗ п. 4.1 называет «уже видел» причиной отказа)
REASONS = {"seen": "Уже видел", "genre": "Не мой жанр", "kind": "Не хочу такое кино"}
REASONS_SERIES = {**REASONS, "kind": "Не хочу такой сериал"}  # те же коды, подпись под сериал
STATUS = {"r": "Оценка {v} ★", "ns": "Не смотрел", "sk": "Пропущено", "sn": "В списке «видел»",
          "ni": "Не интересно", "lt": "В планах", "hd": "Скрыт", "uh": "Возвращён в подборки"}


def genres_kb(selected: set[str]) -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    for g in START_GENRES:
        kb.button(text=("✓ " if g in selected else "") + GENRES_RU[g], callback_data=GenreCb(g=g))
    kb.adjust(3)
    kb.row(*InlineKeyboardBuilder().button(text="Готово →", callback_data=NavCb(a="genres_done")).buttons)
    return kb.as_markup()


Nav = list[tuple[str, str]]  # кнопки навигации: (подпись, действие NavCb)


def _links(kb: InlineKeyboardBuilder, trailer: str | None, watch: str | None) -> list[int]:
    links = [(text, url) for text, url in [("▶ Трейлер", trailer), ("Открыть на Кинопоиске", watch)] if url]
    for text, url in links:
        kb.button(text=text, url=url)
    return [len(links)] if links else []


def _nav(kb: InlineKeyboardBuilder, nav: Nav | None, extra: Nav | None = None) -> list[int]:
    """nav — одна строка (← Назад, Далее →); extra — каждая кнопка отдельной строкой ниже."""
    for text, action in (nav or []) + (extra or []):
        kb.button(text=text, callback_data=NavCb(a=action))
    return ([len(nav)] if nav else []) + [1] * len(extra or [])


def movie_kb(tconst: str, ctx: str, trailer: str | None = None, watch: str | None = None,
             nav: Nav | None = None, extra: Nav | None = None) -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    sizes = _links(kb, trailer, watch)
    for v in range(1, 6):
        kb.button(text=f"{v} ★", callback_data=MovieCb(a="r", t=tconst, v=v, c=ctx))
    if ctx == "cal":
        kb.button(text="Не смотрел", callback_data=MovieCb(a="ns", t=tconst, c=ctx))
        kb.button(text="Пропустить", callback_data=MovieCb(a="sk", t=tconst, c=ctx))
        sizes += [5, 2]
    elif ctx == "hid":
        kb.button(text="Вернуть в подборки", callback_data=MovieCb(a="uh", t=tconst, c=ctx))
        sizes += [5, 1]
    else:
        # «Скрыть навсегда» — под навигацией, чтобы не нажать случайно, листая подборку
        *answers, hide = ACTIONS.items()
        for a, label in answers:
            kb.button(text=label, callback_data=MovieCb(a=a, t=tconst, c=ctx))
        sizes += [5, len(answers)] + _nav(kb, nav)
        kb.button(text=hide[1], callback_data=MovieCb(a=hide[0], t=tconst, c=ctx))
        kb.adjust(*sizes, 1, *_nav(kb, None, extra))
        return kb.as_markup()
    kb.adjust(*sizes, *_nav(kb, nav, extra))
    return kb.as_markup()


def status_kb(text: str, trailer: str | None = None, watch: str | None = None,
              nav: Nav | None = None, extra: Nav | None = None) -> InlineKeyboardMarkup:
    """После ответа вместо кнопок ответа — неактивная строка с итогом; ссылки и навигация остаются."""
    kb = InlineKeyboardBuilder()
    sizes = _links(kb, trailer, watch)
    kb.button(text=f"✓ {text}", callback_data=NavCb(a="noop"))
    kb.adjust(*sizes, 1, *_nav(kb, nav, extra))
    return kb.as_markup()


def reason_kb(tconst: str, series: bool = False) -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    for code, label in (REASONS_SERIES if series else REASONS).items():
        kb.button(text=label, callback_data=ReasonCb(t=tconst, r=code))
    kb.adjust(2, 1)  # в одну строку три подписи на телефоне не помещаются
    return kb.as_markup()


def person_button(p: dict, favorites: set[str]) -> str:
    """Подпись человека в результатах поиска; кто уже в любимых — отмечен, чтобы не добавлять второй раз."""
    if p["nconst"] in favorites:
        return f"✓ {p['name']} — уже в любимых"
    return f"{p['name']} ({p['films']} фильм.)"


def undo_kb(text: str, kind: str, item: str) -> InlineKeyboardMarkup:
    """После выбора любимого: строка с итогом и кнопка «Отменить»."""
    kb = InlineKeyboardBuilder()
    kb.button(text=f"✓ {text}", callback_data=NavCb(a="noop"))
    kb.button(text="↩ Отменить", callback_data=UndoCb(k=kind, i=item))
    kb.adjust(1)
    return kb.as_markup()


def today_kb(question: str, back: bool, extra: Nav | None = None) -> InlineKeyboardMarkup:
    """extra — кнопки сверху отдельными строками (например, «Как в прошлый раз»)."""
    kb = InlineKeyboardBuilder()
    for text, action in extra or []:
        kb.button(text=text, callback_data=NavCb(a=action))
    for v, label in QUESTIONS[question].items():
        kb.button(text=label, callback_data=TodayCb(q=question, v=v))
    n = len(QUESTIONS[question])
    sizes = [2] * (n // 2) + ([1] if n % 2 else [])
    kb.adjust(*[1] * len(extra or []), *sizes, *_nav(kb, [("← Назад", "today_back")] if back else None))
    return kb.as_markup()


def nav_kb(*buttons: tuple[str, str]) -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    for text, action in buttons:
        kb.button(text=text, callback_data=NavCb(a=action))
    kb.adjust(1)
    return kb.as_markup()
