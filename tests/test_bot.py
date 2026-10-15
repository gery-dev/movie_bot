"""Тесты частей бота, которые не требуют Telegram: карточка, кнопки, поиск фильмов."""
import pytest

from bot.cards import CAPTION_LIMIT, card_text, kinopoisk_url, youtube_search_url
from bot.keyboards import MovieCb, PersonCb, movie_kb, status_kb, today_kb
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
    for cb in [MovieCb(a="r", t="tt12345678", v=5, c="rec"), PersonCb(a="films", n="nm12345678")]:
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


def test_carousel_card_has_kinopoisk_and_nav_row_last():
    kb = movie_kb("tt0816692", "rec", trailer="https://youtu.be/x", watch=kinopoisk_url(MOVIE),
                  nav=[("← Назад", "prev"), ("Далее →", "next")])
    rows = [[b.text for b in row] for row in kb.inline_keyboard]
    assert rows[0] == ["▶ Трейлер", "Открыть на Кинопоиске"]
    assert rows[-1] == ["← Назад", "Далее →"]


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


def test_survey_asked_after_5_and_20_ratings(tmp_path):
    from types import SimpleNamespace

    from bot.handlers.recommend import survey_due
    from db.database import connect
    from recsys.profile import record_rating

    conn = connect(tmp_path / "t.db")
    conn.executemany("INSERT INTO movies (tconst, title) VALUES (?, ?)", [(f"tt{i:02d}", str(i)) for i in range(25)])
    app = SimpleNamespace(conn=conn)
    for i in range(4):
        record_rating(conn, 1, f"tt{i:02d}", 4, "recommendation")
    assert survey_due(app, 1) is None
    record_rating(conn, 1, "tt04", 4, "recommendation")
    assert survey_due(app, 1) == 5
    conn.execute("INSERT INTO survey (user_id, wave, score) VALUES (1, 5, 4)")
    assert survey_due(app, 1) is None
    for i in range(5, 20):
        record_rating(conn, 1, f"tt{i:02d}", 4, "recommendation")
    assert survey_due(app, 1) == 20


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
