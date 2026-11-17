"""Тесты частей бота, которые не требуют Telegram: карточка, кнопки, поиск фильмов."""
import pytest

from bot.cards import CAPTION_LIMIT, card_text, kinopoisk_url, youtube_search_url
from bot.keyboards import MovieCb, PersonCb, UndoCb, movie_kb, person_button, status_kb, today_kb, undo_kb
from bot.texts import films_word
from data_pipeline.tmdb import trailer_keys

MOVIE = {"title": "Interstellar", "title_ru": "Интерстеллар", "year": 2014, "runtime": 169, "rating": 8.7,
         "genres": "Adventure,Sci-Fi", "kp_id": 258687, "overview_ru": "Когда засуха приводит человечество к кризису…"}


def test_card_has_main_facts():
    text = card_text(MOVIE, ["Christopher Nolan"], ["Matthew McConaughey", "Anne Hathaway"], ["Okko"])
    for part in ["<b>Интерстеллар</b> (2014)", "<i>Interstellar</i>", "IMDb 8,7", "169 мин",
                 "Приключения, Фантастика", "Режиссёр: Christopher Nolan", "В ролях: Matthew McConaughey",
                 "Где смотреть: Okko", "Когда засуха"]:
        assert part in text


def test_card_fits_caption_and_escapes_html():
    movie = {**MOVIE, "title_ru": "Tom & Jerry <3", "overview_ru": "слово " * 400}
    text = card_text(movie, [], [], [], note="Фильм 1 из 10")
    assert len(text) <= CAPTION_LIMIT
    assert "Tom &amp; Jerry &lt;3" in text
    assert text.endswith("<i>Фильм 1 из 10</i>")
    assert "…" in text


def test_card_without_optional_fields():
    text = card_text({"title": "Untitled", "year": None}, [], [], [])
    assert text == "<b>Untitled</b>"


def test_callback_data_fits_telegram_limit():
    # Telegram ограничивает данные кнопки 64 байтами
    for cb in [MovieCb(a="r", t="tt12345678", v=5, c="rec"), PersonCb(a="films", n="nm12345678"),
               UndoCb(k="person", i="nm12345678")]:
        assert len(cb.pack().encode()) <= 64


def test_recommendation_card_has_no_seen_button():
    # оценка уже значит «смотрел», отдельной кнопки «Уже видел» нет
    labels = [b.text for row in movie_kb("tt0816692", "rec").inline_keyboard for b in row]
    assert "Уже видел" not in labels
    assert {"Не интересно", "Посмотрю позже", "Скрыть навсегда"} <= set(labels)


def test_calibration_card_buttons():
    kb = movie_kb("tt0816692", "cal", trailer="https://youtu.be/x")
    labels = [b.text for row in kb.inline_keyboard for b in row]
    assert labels[0] == "▶ Трейлер"
    assert "Не смотрел" in labels and "Пропустить" in labels and "Скрыть навсегда" not in labels


def test_carousel_card_hide_below_nav():
    kb = movie_kb("tt0816692", "rec", trailer="https://youtu.be/x", watch=kinopoisk_url(MOVIE),
                  nav=[("← Назад", "prev"), ("Далее →", "next")], extra=[("Другое настроение", "mood")])
    rows = [[b.text for b in row] for row in kb.inline_keyboard]
    assert rows[0] == ["▶ Трейлер", "Открыть на Кинопоиске"]
    assert rows[2:] == [["Не интересно", "Посмотрю позже"], ["← Назад", "Далее →"], ["Скрыть навсегда"],
                        ["Другое настроение"]]


def test_status_keeps_links_and_nav():
    kb = status_kb("Оценка 4 ★", watch="https://kp", nav=[("Далее →", "next")])
    rows = [[b.text for b in row] for row in kb.inline_keyboard]
    assert rows == [["Открыть на Кинопоиске"], ["✓ Оценка 4 ★"], ["Далее →"]]


def test_kinopoisk_url_is_film_page_or_none():
    assert kinopoisk_url(MOVIE) == "https://www.kinopoisk.ru/film/258687/"
    assert kinopoisk_url({**MOVIE, "kp_id": None}) is None


def test_trailers_russian_first_then_english_only():
    videos = [{"site": "YouTube", "type": "Trailer", "iso_639_1": "en", "key": "en1"},
              {"site": "YouTube", "type": "Teaser", "iso_639_1": "ru", "key": "teaser"},
              {"site": "YouTube", "type": "Trailer", "iso_639_1": "de", "key": "de1"},
              {"site": "YouTube", "type": "Trailer", "iso_639_1": "ru", "key": "ru1"}]
    assert trailer_keys({"videos": {"results": videos}}) == ["ru1", "en1"]


def test_youtube_search_fallback():
    assert youtube_search_url(MOVIE) == ("https://www.youtube.com/results?search_query="
                                         "%D0%98%D0%BD%D1%82%D0%B5%D1%80%D1%81%D1%82%D0%B5%D0%BB%D0%BB%D0%B0%D1%80+2014+"
                                         "%D1%82%D1%80%D0%B5%D0%B9%D0%BB%D0%B5%D1%80")


def test_today_keyboard_back_only_after_first_question():
    first = [b.text for row in today_kb("mood", back=False).inline_keyboard for b in row]
    later = [b.text for row in today_kb("time", back=True).inline_keyboard for b in row]
    assert "← Назад" not in first and later[-1] == "← Назад"
    assert "До 3 часов" in later and "Весёлое" in first


def test_text_soon_after_rating_is_review(monkeypatch):
    import bot.services as services

    app = services.App.__new__(services.App)  # без загрузки данных: проверяем только окно отзыва
    app.last_rated = {}
    now = [1000.0]
    monkeypatch.setattr(services.time, "time", lambda: now[0])
    app.last_rated[1] = ("tt01", now[0])
    assert app.awaiting_review(1) == "tt01"
    now[0] += services.REVIEW_WINDOW + 1
    assert app.awaiting_review(1) is None
    assert app.awaiting_review(2) is None


@pytest.mark.parametrize("n, word", [(1, "фильм"), (3, "фильма"), (5, "фильмов"), (11, "фильмов"), (22, "фильма")])
def test_films_word(n, word):
    assert films_word(n) == word


def test_survey_asked_after_3_and_20_ratings_once(tmp_path):
    from types import SimpleNamespace

    from bot.handlers.recommend import survey_due
    from db.database import connect
    from recsys.profile import record_rating

    conn = connect(tmp_path / "t.db")
    conn.executemany("INSERT INTO movies (tconst, title) VALUES (?, ?)", [(f"tt{i:02d}", str(i)) for i in range(25)])
    app = SimpleNamespace(conn=conn, survey_offered=set())
    for i in range(2):
        record_rating(conn, 1, f"tt{i:02d}", 4, "recommendation")
    assert survey_due(app, 1) is None
    record_rating(conn, 1, "tt02", 4, "recommendation")
    assert survey_due(app, 1) == 3
    app.survey_offered.add((1, 3))  # спросили, но не ответил — после следующих оценок не повторяем
    record_rating(conn, 1, "tt03", 4, "recommendation")
    assert survey_due(app, 1) is None
    for i in range(4, 20):
        record_rating(conn, 1, f"tt{i:02d}", 4, "recommendation")
    assert survey_due(app, 1) == 20
    conn.execute("INSERT INTO survey (user_id, wave, score) VALUES (1, 20, 4)")
    assert survey_due(app, 1) is None


def test_old_database_gets_new_column(tmp_path):
    import sqlite3

    from db.database import connect

    old = sqlite3.connect(tmp_path / "old.db")
    old.execute("CREATE TABLE recommendations (id INTEGER PRIMARY KEY, user_id INTEGER, tconst TEXT, "
                "shown_at TEXT, asked_at TEXT, answer TEXT)")
    old.commit()
    old.close()
    conn = connect(tmp_path / "old.db")
    assert "mode" in {r[1] for r in conn.execute("PRAGMA table_info(recommendations)")}


def _update(**kw):
    from datetime import datetime

    from aiogram.types import CallbackQuery, Chat, Document, Message, Update, User
    user = User(id=1, is_bot=False, first_name="T")
    chat = Chat(id=1, type="private")
    if "callback" in kw:
        return Update(update_id=1, callback_query=CallbackQuery(id="1", from_user=user, chat_instance="c",
                                                                data=kw["callback"]))
    doc = Document(file_id="f", file_unique_id="u", mime_type="application/zip") if kw.get("doc") else None
    return Update(update_id=1, message=Message(message_id=1, date=datetime.now(), chat=chat, from_user=user,
                                               text=kw.get("text"), document=doc))


def test_analytics_keeps_no_message_text():
    from bot.analytics import describe
    assert describe(_update(text="/start@movies_selection_bot g_42")) == ("command", "start", "g_42")
    assert describe(_update(text="/recommend")) == ("command", "recommend", None)
    assert describe(_update(text="мой секретный запрос")) == ("text", None, None)
    assert describe(_update(doc=True)) == ("file", "application/zip", None)
    assert describe(_update(callback="mv:rate:tt0816692:4")) == ("callback", "mv", "rate:tt0816692:4")


def test_analytics_writes_event(tmp_path):
    from bot.analytics import log_event
    from db.database import connect
    conn = connect(tmp_path / "t.db")
    log_event(conn, 1, _update(text="/today"), "Onboarding:genres", 12, False)
    row = dict(conn.execute("SELECT user_id, kind, name, state, duration_ms, error FROM events").fetchone())
    assert row == {"user_id": 1, "kind": "command", "name": "today", "state": "Onboarding:genres",
                   "duration_ms": 12, "error": 0}


def test_description_fits_telegram_limits():
    from bot import texts
    assert len(texts.DESCRIPTION) <= 512 and len(texts.SHORT_DESCRIPTION) <= 120
    assert texts.WELCOME.startswith(texts.DESCRIPTION)


def test_user_updates_handled_one_by_one():
    """Быстрые нажатия одного пользователя не затирают состояние друг друга (как было с выбором жанров)."""
    import asyncio
    from datetime import datetime

    from aiogram import Bot, Dispatcher
    from aiogram.fsm.context import FSMContext
    from aiogram.fsm.storage.memory import SimpleEventIsolation
    from aiogram.types import Chat, Message, Update, User

    dp = Dispatcher(events_isolation=SimpleEventIsolation())

    @dp.message()
    async def tap(message: Message, state: FSMContext) -> None:
        n = (await state.get_data()).get("n", 0)
        await asyncio.sleep(0.01)  # как ожидание ответа Telegram
        await state.update_data(n=n + 1)

    async def run() -> int:
        bot = Bot("123:TEST")
        user, chat = User(id=1, is_bot=False, first_name="u"), Chat(id=1, type="private")
        updates = [Update(update_id=i, message=Message(message_id=i, date=datetime.now(), chat=chat, from_user=user,
                                                       text="x")) for i in range(5)]
        await asyncio.gather(*(dp.feed_update(bot, u) for u in updates))
        state = dp.fsm.get_context(bot, chat_id=1, user_id=1)
        return (await state.get_data())["n"]

    assert asyncio.run(run()) == 5


def test_undo_button_after_favorite_pick():
    rows = undo_kb("В любимых: Кристофер Нолан", "person", "nm0634240").inline_keyboard
    assert [b.text for row in rows for b in row] == ["✓ В любимых: Кристофер Нолан", "↩ Отменить"]
    assert UndoCb.unpack(rows[1][0].callback_data) == UndoCb(k="person", i="nm0634240")


def test_person_already_in_favorites_is_marked():
    p = {"nconst": "nm0853380", "name": "Андрей Тарковский", "films": 7}
    assert person_button(p, set()) == "Андрей Тарковский (7 фильм.)"
    assert person_button(p, {"nm0853380"}) == "✓ Андрей Тарковский — уже в любимых"


def _app_with_db(tmp_path):
    import bot.services as services
    from db.database import connect

    app = services.App.__new__(services.App)  # без загрузки моделей: нужна только БД
    app.conn = connect(tmp_path / "t.db")
    app.conn.executemany("INSERT INTO movies (tconst, title) VALUES (?, ?)", [(f"tt{i:02d}", str(i)) for i in range(5)])
    return app


def test_shown_log_keeps_position_ratings_model_and_mood(tmp_path):
    import json

    from recsys.profile import record_rating
    from recsys.recommender import MODEL_VERSION

    app = _app_with_db(tmp_path)
    record_rating(app.conn, 1, "tt00", 5, "calibration")
    record_rating(app.conn, 1, "tt01", 3, "calibration")
    app.log_shown(1, ["tt02", "tt03"], "today", start_position=4, mood={"mood": "fun", "time": "120"})
    app.log_shown(1, ["tt04"])  # позиция и настроение необязательны
    rows = [dict(r) for r in app.conn.execute(
        "SELECT mode, position, n_ratings, model_version, mood FROM recommendations ORDER BY id")]
    assert [(r["position"], r["n_ratings"], r["model_version"]) for r in rows] == [
        (4, 2, MODEL_VERSION), (5, 2, MODEL_VERSION), (None, 2, MODEL_VERSION)]
    assert json.loads(rows[0]["mood"]) == {"mood": "fun", "time": "120"} and rows[2]["mood"] is None
    assert [r["mode"] for r in rows] == ["today", "today", "recommend"]


def test_search_log_keeps_length_not_text(tmp_path):
    app = _app_with_db(tmp_path)
    first = app.log_search(1, "film", "интерстеллар", 3)
    app.log_search(1, "film", "ыыы", 0)  # ничего не нашлось — выбрать нечего
    app.mark_search_chosen(1, "film")      # выбор относится к последнему поиску, где что-то нашлось
    text_id = app.log_search(1, "text", "что-нибудь повеселее", 20)
    app.mark_search_chosen(1, "text", text_id)
    rows = [tuple(r) for r in app.conn.execute("SELECT kind, query_len, found, chosen FROM search_log ORDER BY id")]
    assert rows == [("film", 12, 3, 1), ("film", 3, 0, 0), ("text", 20, 20, 1)] and first == 1
    assert "интерстеллар" not in repr(list(app.conn.iterdump()))


def test_survey_offer_logged_each_time(tmp_path):
    app = _app_with_db(tmp_path)
    app.log_survey_offer(1, 3)
    app.log_survey_offer(1, 3)  # повторное предложение (например, после перезапуска бота) видно отдельной строкой
    assert app.conn.execute("SELECT count(*), min(wave) FROM survey_offers WHERE user_id = 1").fetchone()[:] == (2, 3)


def test_event_knows_whether_a_real_handler_took_it(tmp_path):
    """events.matched: 1 — настоящий обработчик, 0 — заглушка «не понял» или обработчика нет совсем."""
    import asyncio

    from aiogram import Bot, Dispatcher, F
    from aiogram.types import Message

    from bot import analytics
    from db.database import connect

    conn = connect(tmp_path / "t.db")
    dp = Dispatcher()
    dp.update.outer_middleware(analytics.middleware(conn))

    @dp.message(F.text == "ok")
    async def real(message: Message) -> None:
        pass

    @dp.message(F.text == "boom")
    async def broken(message: Message) -> None:
        raise RuntimeError("сбой внутри настоящего обработчика")

    @dp.message()
    async def fallback(message: Message) -> None:
        analytics.mark_fallback()

    async def run() -> None:
        bot = Bot("123:TEST")
        for text in ("ok", "что это", "ok"):  # подряд в одной задаче: флаг прошлого обновления не должен остаться
            await dp.feed_update(bot, _update(text=text))
        with pytest.raises(RuntimeError):  # ошибка в обработчике — он всё же настоящий (в боте её ловит router.errors)
            await dp.feed_update(bot, _update(text="boom"))

    asyncio.run(run())
    rows = [tuple(r) for r in conn.execute("SELECT matched, error FROM events ORDER BY id")]
    assert rows == [(1, 0), (0, 0), (1, 0), (1, 1)]

    conn2 = connect(tmp_path / "t2.db")
    empty = Dispatcher()  # ни одного обработчика → UNHANDLED
    empty.update.outer_middleware(analytics.middleware(conn2))
    asyncio.run(empty.feed_update(Bot("123:TEST"), _update(text="ok")))
    assert conn2.execute("SELECT matched FROM events").fetchone()[0] == 0


def test_migration_is_repeatable_and_keeps_rows(tmp_path):
    import sqlite3

    from db.database import connect

    old = sqlite3.connect(tmp_path / "old.db")
    old.execute("CREATE TABLE recommendations (id INTEGER PRIMARY KEY, user_id INTEGER, tconst TEXT, "
                "shown_at TEXT, asked_at TEXT, answer TEXT, mode TEXT)")
    old.execute("CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, ts TEXT, kind TEXT NOT NULL, "
                "name TEXT, detail TEXT, state TEXT, duration_ms INTEGER, error INTEGER NOT NULL DEFAULT 0)")
    old.execute("INSERT INTO recommendations (user_id, tconst, mode) VALUES (1, 'tt01', 'recommend')")
    old.execute("INSERT INTO events (user_id, kind) VALUES (1, 'command')")
    old.commit()
    old.close()
    for _ in range(2):  # второй запуск ничего не ломает и не дублирует
        conn = connect(tmp_path / "old.db")
        cols = [r[1] for r in conn.execute("PRAGMA table_info(recommendations)")]
        assert {"position", "n_ratings", "model_version", "mood"} <= set(cols) and len(cols) == len(set(cols))
        assert "matched" in {r[1] for r in conn.execute("PRAGMA table_info(events)")}
        assert conn.execute("SELECT position, n_ratings, model_version, mood FROM recommendations").fetchone()[:] == (
            None, None, None, None)
        assert conn.execute("SELECT matched FROM events").fetchone()[0] is None
        assert conn.execute("SELECT count(*) FROM search_log").fetchone()[0] == 0
        conn.close()


def test_reason_callback_fits_telegram_limit():
    from bot.keyboards import ReasonCb
    assert len(ReasonCb(t="tt12345678", r="genre").pack().encode()) <= 64


def test_set_reason_goes_to_latest_not_interested_row_only(tmp_path):
    from recsys.profile import ensure_user

    app = _app_with_db(tmp_path)
    ensure_user(app.conn, 1)
    app.conn.executemany("INSERT INTO recommendations (user_id, tconst, answer) VALUES (1, ?, ?)",
                         [("tt01", "not_interested"), ("tt01", "not_interested"), ("tt02", "rated"),
                          ("tt03", None)])
    app.set_reason(1, "tt01", "genre")
    app.set_reason(1, "tt02", "seen")   # на оцененный фильм причина отказа не ставится
    app.set_reason(1, "tt03", "seen")   # и на фильм без ответа тоже
    rows = [r[0] for r in app.conn.execute("SELECT reason FROM recommendations ORDER BY id")]
    assert rows == [None, "genre", None, None]


def _reason_fakes():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    sent, deleted = [], []

    class FakeBot:
        async def send_message(self, chat_id, text, reply_markup=None):
            sent.append((chat_id, text))
            return SimpleNamespace(message_id=100 + len(sent))

        async def delete_message(self, chat_id, message_id):
            deleted.append(message_id)

    def callback(user, message_id=7):
        return SimpleNamespace(from_user=SimpleNamespace(id=user), answer=AsyncMock(),
                               message=SimpleNamespace(message_id=message_id, edit_reply_markup=AsyncMock()))
    return FakeBot(), sent, deleted, callback


def test_not_interested_asks_why_and_replaces_unanswered_question(monkeypatch):
    import asyncio
    from types import SimpleNamespace

    from bot.handlers import recommend
    from bot.keyboards import MovieCb
    from bot.services import Carousel

    async def fake_advance(bot, user, app, new_message=False):
        return True
    monkeypatch.setattr(recommend, "advance", fake_advance)
    bot, sent, deleted, callback = _reason_fakes()
    app = SimpleNamespace(carousels={1: Carousel(films=["tt01", "tt02"], message_id=7)}, reason_msgs={},
                          react=lambda *a, **k: None, label=lambda t: f"Фильм <{t}>")

    async def press(user, action, film):
        await recommend.card_answer(callback(user), MovieCb(a=action, t=film, c="rec"), None, bot, app)

    async def run():
        await press(1, "ni", "tt01")
        assert len(sent) == 1 and "&lt;tt01&gt;" in sent[0][1] and app.reason_msgs == {1: 101}
        await press(1, "lt", "tt02")    # «позже» причину не спрашивает
        assert len(sent) == 1
        await press(1, "ni", "tt02")    # новый вопрос заменяет прежний без ответа
        assert len(sent) == 2 and deleted == [101] and app.reason_msgs == {1: 102}
        await press(2, "ni", "tt01")    # карточка не из подборки: статус на месте, вопрос тоже
        assert len(sent) == 3 and app.reason_msgs == {1: 102, 2: 103}
    asyncio.run(run())


def test_answered_reason_is_saved_and_not_deleted_later(tmp_path):
    import asyncio

    from bot.handlers import recommend
    from bot.keyboards import ReasonCb
    from recsys.profile import ensure_user

    app = _app_with_db(tmp_path)
    ensure_user(app.conn, 1)
    app.conn.execute("INSERT INTO recommendations (user_id, tconst, answer) VALUES (1, 'tt01', 'not_interested')")
    app.reason_msgs = {1: 55}
    _, _, _, callback = _reason_fakes()
    asyncio.run(recommend.reason_answer(callback(1, 55), ReasonCb(t="tt01", r="kind"), app))
    asyncio.run(recommend.reason_answer(callback(1, 56), ReasonCb(t="tt01", r="нет-такой"), app))  # чужой код игнорируется
    assert app.conn.execute("SELECT reason FROM recommendations").fetchone()[0] == "kind"
    assert app.reason_msgs == {}  # отвеченный вопрос из списка «ждут ответа» убран


def test_series_card_shows_years_seasons_and_episode():
    from bot.services import years_label
    show = {"title": "Breaking Bad", "title_ru": "Во все тяжкие", "year": 2008, "end_year": 2013, "kind": "series",
            "seasons": 5, "episode_runtime": 48, "rating": 9.5, "genres": "Crime,Drama", "runtime": None}
    text = card_text(show, ["A", "B", "C"], [], [])
    assert "<b>Во все тяжкие</b> (2008–2013)" in text and "5 сезонов" in text and "серия ~48 мин" in text
    assert "Режиссёры: A, B" in text and "C" not in text.split("Режиссёры: ")[1].split("\n")[0]
    assert years_label(2019, None, "series") == " (2019–…)" and years_label(2016, 2016, "series") == " (2016)"
    assert years_label(2010, None, "movie") == " (2010)"
