"""Тесты профиля, профиля на сегодня, избранного и калибровки на маленькой тестовой БД."""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from db.database import connect
from recsys import calibration
from recsys.explain import explain
from recsys.favorites import add_favorite, person_unseen, suggest_favorites
from recsys.profile import (DAY, Profiler, decay, excluded_films, rating_signal, rebuild_profile, record_rating,
                            record_reaction, load_profile, refused_films, user_events)
from recsys.recommender import Recommender
from recsys.today import (extend_daily, load_daily, load_last, save_daily, summary, today_filter,
                          today_scores)

USER = 1
MOVIES = [
    # tconst, title, year, runtime, genres, language, rating, votes
    ("tt01", "Space One", 2014, 169, "Sci-Fi,Drama", "en", 8.7, 900_000),
    ("tt02", "Space Two", 2010, 148, "Sci-Fi,Action", "en", 8.8, 800_000),
    ("tt03", "Space Three", 2020, 150, "Sci-Fi,Adventure", "en", 8.0, 700_000),
    ("tt04", "Space Four", 1979, 117, "Sci-Fi,Horror", "en", 8.5, 600_000),
    ("tt05", "Love Story", 2004, 95, "Romance,Drama", "en", 7.5, 300_000),
    ("tt06", "Funny", 2009, 85, "Comedy", "en", 7.0, 200_000),
    ("tt07", "Русская комедия", 1975, 184, "Comedy,Romance", "ru", 8.2, 30_000),
    ("tt08", "Cartoon", 2001, 92, "Animation,Family", "ja", 8.6, 500_000),
    ("tt09", "Space Five", 2023, 105, "Sci-Fi,Thriller", "en", 7.2, 100_000),
]


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "test.db")
    c.executemany(
        "INSERT INTO movies (tconst, title, year, runtime, genres, language, rating, votes) VALUES (?,?,?,?,?,?,?,?)",
        MOVIES,
    )
    c.executemany("INSERT INTO movie_genres VALUES (?, ?)",
                  [(m[0], g) for m in MOVIES for g in m[4].split(",")])
    c.executemany("INSERT INTO people (nconst, name) VALUES (?, ?)",
                  [("nm1", "Christopher Nolan"), ("nm2", "Other Director"), ("nm3", "Matthew McConaughey")])
    c.executemany("INSERT INTO movie_people VALUES (?, ?, ?, ?)", [
        ("tt01", "nm1", "director", None), ("tt02", "nm1", "director", None), ("tt03", "nm1", "director", None),
        ("tt09", "nm1", "director", None), ("tt04", "nm2", "director", None), ("tt01", "nm3", "actor", 1),
    ])
    yield c
    c.close()


@pytest.fixture
def profiler(conn):
    return Profiler([m[0] for m in MOVIES], conn=conn)


def weight(profiler, profile, feature):
    return profile[profiler.names.index(feature)]


# --- долгосрочный профиль ---

def test_rating_signal_range():
    assert rating_signal(5) == 1 and rating_signal(1) == -1 and rating_signal(3) == 0


def test_decay_halves_after_half_life():
    assert decay(365 * DAY, 365) == pytest.approx(0.5)
    assert decay(10 * DAY, None) == 1


def test_high_ratings_raise_genre_weight_low_lower(conn, profiler):
    for t in ["tt01", "tt02", "tt03"]:
        record_rating(conn, USER, t, 5)
    high = rebuild_profile(conn, profiler, USER)
    assert weight(profiler, high, "genre:Sci-Fi") > 0

    for t in ["tt01", "tt02", "tt03"]:
        record_rating(conn, USER, t, 1)
    low = rebuild_profile(conn, profiler, USER)
    assert weight(profiler, low, "genre:Sci-Fi") < 0


def test_old_rating_weighs_less(profiler):
    now = 1_000 * DAY
    fresh = profiler.vector([("tt05", 1.0, now)], now=now)
    old = profiler.vector([("tt05", 1.0, now - 365 * DAY)], now=now, half_life_days=365)
    assert weight(profiler, old, "genre:Romance") == pytest.approx(weight(profiler, fresh, "genre:Romance") / 2)


def test_not_seen_is_weak_negative(conn, profiler):
    record_rating(conn, USER, "tt05", 5)
    record_reaction(conn, USER, "tt06", "not_seen")
    p = rebuild_profile(conn, profiler, USER)
    assert -weight(profiler, p, "genre:Comedy") < weight(profiler, p, "genre:Romance") / 5
    assert weight(profiler, p, "genre:Comedy") < 0


def test_seen_goes_to_list_without_taste_change(conn, profiler):
    record_reaction(conn, USER, "tt06", "seen")
    p = rebuild_profile(conn, profiler, USER)
    assert not p.any()
    assert "tt06" in excluded_films(conn, USER)


def test_later_goes_to_planned(conn):
    record_reaction(conn, USER, "tt05", "later")
    lists = conn.execute("SELECT list FROM user_lists WHERE user_id = ? AND tconst = 'tt05'", (USER,)).fetchall()
    assert [tuple(r) for r in lists] == [("planned",)]


def test_rating_replaces_earlier_reaction(conn):
    record_reaction(conn, USER, "tt06", "not_seen")
    record_rating(conn, USER, "tt06", 4)
    assert [e[0] for e in user_events(conn, USER)] == ["tt06"]


def test_review_tone_refines_rating(conn, profiler):
    record_rating(conn, USER, "tt05", 4)
    base = weight(profiler, rebuild_profile(conn, profiler, USER), "genre:Romance")
    record_reaction(conn, USER, "tt05", "review", review="обожаю", sentiment=0.9)
    loved = weight(profiler, rebuild_profile(conn, profiler, USER), "genre:Romance")
    assert loved > base
    record_reaction(conn, USER, "tt05", "review", review="на самом деле так себе", sentiment=-0.9)
    assert weight(profiler, rebuild_profile(conn, profiler, USER), "genre:Romance") == pytest.approx(base)


def test_profile_saved_and_loaded(conn, profiler):
    record_rating(conn, USER, "tt01", 5)
    saved = rebuild_profile(conn, profiler, USER)
    assert np.allclose(load_profile(conn, profiler, USER), saved)


# --- профиль на сегодня ---

def movies_df():
    return pd.DataFrame(MOVIES, columns=["tconst", "title", "year", "runtime", "genres", "language", "rating", "votes"])


def test_today_time_filter():
    m = movies_df()
    ok = today_filter(m, {"time": "90"})
    assert set(m.tconst[ok]) == {"tt06"}
    assert (m.runtime[ok] <= 90).all()


def test_today_language_filter():
    m = movies_df()
    assert set(m.tconst[today_filter(m, {"language": "ru"})]) == {"tt07"}


def test_today_family_prefers_animation_and_avoids_horror():
    m = movies_df().set_index("tconst")
    s = pd.Series(today_scores(m.reset_index(), {"company": "family"}), index=m.index)
    assert s["tt08"] == s.max()
    assert s["tt04"] < 0


def test_daily_session_saved_and_expires(conn):
    save_daily(conn, USER, {"mood": "fun"})
    assert load_daily(conn, USER) == {"mood": "fun"}
    save_daily(conn, USER, {"mood": "fun"}, hours=-1)
    assert load_daily(conn, USER) is None
    assert load_last(conn, USER) == {"mood": "fun"}  # для кнопки «Как в прошлый раз»


def test_activity_extends_only_live_answers(conn):
    save_daily(conn, USER, {"mood": "fun"}, hours=1)
    extend_daily(conn, USER)
    expires = conn.execute("SELECT expires_at FROM daily_session WHERE user_id = ?", (USER,)).fetchone()[0]
    assert datetime.fromisoformat(expires) > datetime.now() + timedelta(hours=21)
    save_daily(conn, USER, {"mood": "fun"}, hours=-1)
    extend_daily(conn, USER)  # устаревшие ответы не оживают
    assert load_daily(conn, USER) is None


def test_summary_skips_any():
    assert summary({"mood": "fun", "time": "180", "era": "any"}) == "Весёлое · До 3 часов"
    assert summary({"mood": "any"}) == "без особых пожеланий"


# --- любимые люди ---

def test_suggest_after_three_liked_films(conn):
    for t in ["tt01", "tt02"]:
        record_rating(conn, USER, t, 5)
    assert suggest_favorites(conn, USER) == []
    record_rating(conn, USER, "tt03", 4)
    assert [s["nconst"] for s in suggest_favorites(conn, USER)] == ["nm1"]
    add_favorite(conn, USER, "nm1", "declined")
    assert suggest_favorites(conn, USER) == []  # отказ не предлагаем повторно


def test_person_unseen_sorted_by_rating(conn):
    record_rating(conn, USER, "tt02", 5)
    films = person_unseen(conn, USER, "nm1")
    assert [f["tconst"] for f in films] == ["tt01", "tt03", "tt09"]


# --- итоговая формула ---

def test_recommender_excludes_seen_and_applies_filters(conn):
    rec = Recommender(conn)
    record_rating(conn, USER, "tt01", 5)
    record_reaction(conn, USER, "tt02", "seen")
    out = rec.recommend(conn, USER, k=9, answers={"time": "120"})
    assert not {"tt01", "tt02"} & set(out.tconst)
    assert (out.runtime <= 120).all()


def test_favorite_bonus_raises_film(conn):
    rec = Recommender(conn)
    record_rating(conn, USER, "tt05", 5)
    score = lambda: rec.recommend(conn, USER, k=9, answers={}).set_index("tconst").score  # noqa: E731
    before = score()
    add_favorite(conn, USER, "nm2")
    after = score()
    assert after["tt04"] > before["tt04"]                      # фильм любимого режиссёра поднялся
    assert after.drop("tt04").equals(before.drop("tt04"))      # остальные не изменились


def test_recommend_ignores_saved_mood(conn):
    """/recommend — только вкус: сохранённые ответы /today на неё не влияют."""
    rec = Recommender(conn)
    record_rating(conn, USER, "tt05", 5)
    plain = rec.recommend(conn, USER, k=9).tconst.tolist()
    save_daily(conn, USER, {"time": "90", "company": "family"})
    assert rec.recommend(conn, USER, k=9).tconst.tolist() == plain
    assert rec.recommend(conn, USER, k=9, answers=load_daily(conn, USER)).tconst.tolist() != plain


def test_explanation_names_director_and_genre(conn):
    rec = Recommender(conn)
    for t in ["tt01", "tt02"]:
        record_rating(conn, USER, t, 5)
    out = rec.recommend(conn, USER, k=9, answers={})
    why = out.set_index("tconst").why
    assert why["tt03"].startswith("Почему: тебе нравятся")
    assert "Christopher Nolan" in why["tt03"] or "фантастика" in why["tt03"]


def test_explanation_mentions_favorite():
    text = explain(taste=["фантастика"], favorites=["Christopher Nolan"], mood=False, cf_driven=False, popular=False)
    assert text == "Почему: здесь Christopher Nolan — из твоих любимых; тебе нравятся фантастика"


def test_mmr_breaks_same_director_streak(conn):
    """Без MMR первыми идут фильмы Нолана подряд; с MMR между ними попадает другое."""
    rec = Recommender(conn)
    for t in ["tt01", "tt02"]:
        record_rating(conn, USER, t, 5)
    plain = rec.recommend(conn, USER, k=3, answers={}, diverse=False).tconst.tolist()
    diverse = rec.recommend(conn, USER, k=3, answers={}).tconst.tolist()
    nolan = {"tt01", "tt02", "tt03", "tt09"}
    assert sum(t in nolan for t in diverse) <= sum(t in nolan for t in plain)
    assert len(set(diverse)) == 3


def test_group_skips_films_seen_by_anyone_and_fits_both(conn):
    rec = Recommender(conn)
    record_rating(conn, 1, "tt01", 5)   # первому нравится фантастика Нолана
    record_rating(conn, 2, "tt08", 5)   # второму — мультфильмы
    record_reaction(conn, 2, "tt02", "seen")
    out = rec.recommend_group(conn, [1, 2], k=9)
    assert not {"tt01", "tt08", "tt02"} & set(out.tconst)
    assert out.min_fit.iloc[0] >= out.min_fit.min()
    assert set(out.best) <= {1, 2}


def test_new_user_gets_popular_films(conn):
    out = Recommender(conn).recommend(conn, 99, k=3, answers={})
    assert out.tconst.iloc[0] in {"tt01", "tt02"}


# --- калибровка ---

def test_calibration_is_diverse(conn):
    pool = calibration.calibration_pool(movies_df().set_index("tconst").assign(country="US"))
    rng = np.random.default_rng(0)
    shown = []
    for _ in range(4):
        shown.append(calibration.next_film(pool, shown, rng=rng))
    assert len(set(shown)) == 4
    genres = set().union(*pool.loc[shown].genre_set)
    assert len(genres) >= 5  # четыре фильма — не меньше пяти разных жанров


def test_excluded_films_permanent_and_paused(conn):
    record_reaction(conn, USER, "tt01", "not_interested")
    record_reaction(conn, USER, "tt02", "not_seen")
    record_reaction(conn, USER, "tt03", "not_seen")
    conn.execute("UPDATE reactions SET created_at = datetime('now', '-40 days') WHERE tconst = 'tt03'")
    conn.execute("INSERT INTO recommendations (user_id, tconst) VALUES (?, 'tt04')", (USER,))
    conn.execute("INSERT INTO recommendations (user_id, tconst, shown_at) VALUES (?, 'tt05', '2000-01-01 00:00:00')",
                 (USER,))
    skip = excluded_films(conn, USER)
    assert {"tt01", "tt02", "tt04"} <= skip          # «не интересно» навсегда, свежие «не смотрел» и показы — на паузе
    assert not {"tt03", "tt05"} & skip               # через 30 дней снова можно


def test_recency_and_new_only(conn):
    from recsys.recommender import recency
    assert list(recency(np.array([1990, 2026, np.nan]), now_year=2026)) == [0.0, 1.0, 0.0]
    rec = Recommender(conn)
    assert set(rec.recommend(conn, USER, k=5, min_year=2020).tconst) == {"tt03", "tt09"}


def test_chosen_genres_saved_and_raise_their_films(conn):
    from recsys.profile import load_genres, save_genres
    save_genres(conn, USER, ["Comedy", "Romance"])
    save_genres(conn, USER, ["Comedy"])                     # новый выбор заменяет прежний
    assert load_genres(conn, USER) == {"Comedy"}
    record_rating(conn, USER, "tt08", 3)
    rec = Recommender(conn)
    with_genre = rec.recommend(conn, USER, k=9, diverse=False).tconst.tolist()
    save_genres(conn, USER, [])
    assert load_genres(conn, USER) == set()
    without = rec.recommend(conn, USER, k=9, diverse=False).tconst.tolist()
    assert with_genre.index("tt06") < without.index("tt06")  # комедия поднялась
    assert "tt07" not in with_genre                          # 30 тыс. голосов — ниже порога известности


def test_taste_fit_follows_liked_years_and_language(conn):
    rec = Recommender(conn)
    assert rec.taste_fit(conn, USER) is None  # понравившихся пока мало
    for t in ["tt03", "tt09", "tt01"]:  # 2020, 2023, 2014, все на английском
        record_rating(conn, USER, t, 5, "onboarding")
    era, lang = rec.taste_fit(conn, USER)
    pos = rec.catalog.index
    assert era[pos["tt03"]] > era[pos["tt07"]]   # 2020 ближе к понравившемуся, чем 1975
    assert lang[pos["tt01"]] == 1.0 and lang[pos["tt07"]] == 0.0  # русского среди понравившегося нет


def test_refused_years_push_era_away_but_seen_reason_does_not(conn):
    rec = Recommender(conn)
    pos = rec.catalog.index
    for t in ["tt03", "tt09", "tt01"]:  # понравились 2020, 2023, 2014
        record_rating(conn, USER, t, 5, "onboarding")
    before, lang_before = rec.taste_fit(conn, USER)
    record_reaction(conn, USER, "tt04", "not_interested")  # отказ от фильма 1979 года
    record_reaction(conn, USER, "tt05", "not_interested")  # отказ, а потом оценка — это уже не отказ
    record_rating(conn, USER, "tt05", 3)
    assert refused_films(conn, USER) == {"tt04"}
    after, lang_after = rec.taste_fit(conn, USER)
    # фильм 1975 года (рядом с отвергнутым) опустился и относительно 2009-го, и ниже нуля
    assert after[pos["tt07"]] - after[pos["tt06"]] < before[pos["tt07"]] - before[pos["tt06"]]
    assert after[pos["tt07"]] < 0
    assert np.array_equal(lang_after, lang_before)  # язык по отказам не меняется
    # на «Почему?» ответили «Уже видел»: это не нелюбовь и не отказ по эпохе
    conn.execute("INSERT INTO recommendations (user_id, tconst, answer, reason) VALUES (?, 'tt04', 'not_interested', 'seen')",
                 (USER,))
    assert refused_films(conn, USER) == set()
    assert {t: s for t, s, _ in user_events(conn, USER)}["tt04"] == 0.0
    seen_era, _ = rec.taste_fit(conn, USER)
    liked_only = rec._year_closeness(["tt03", "tt09", "tt01"])[0] / 3
    assert np.allclose(seen_era, liked_only)


def test_als_file_and_only_liked_ratings(conn, tmp_path):
    from recsys.models import ALS, Catalog
    ids = [m[0] for m in MOVIES]
    train = pd.DataFrame({"userId": [1, 1, 2, 2, 3, 3], "tconst": ["tt01", "tt02", "tt01", "tt02", "tt05", "tt06"],
                          "rating": [5.0, 5.0, 4.0, 5.0, 5.0, 4.0]})
    als = ALS(Catalog(ids), factors=2, iterations=3).fit(train)
    als.save(tmp_path / "als.npz")
    other = ALS.load(tmp_path / "als.npz", Catalog(ids[::-1] + ["tt99"]))  # другой порядок каталога и новый фильм
    assert other.known[other.catalog.index["tt01"]] and not other.known[other.catalog.index["tt03"]]
    assert not other.known[other.catalog.index["tt99"]]
    assert np.allclose(other.model.item_factors[other.catalog.index["tt05"]], als.model.item_factors[als.catalog.index["tt05"]])
    rec = Recommender(conn, als=ALS.load(tmp_path / "als.npz", Catalog(ids)))
    record_rating(conn, USER, "tt05", 1)
    assert rec.long_term_parts(conn, USER)["cf"] is None  # «1★» в ALS не идёт — нелюбимое учитывает профиль
    record_rating(conn, USER, "tt01", 5)
    assert rec.long_term_parts(conn, USER)["cf"] is not None


def test_undo_onboarding_film_keeps_other_ratings(conn, profiler):
    from types import SimpleNamespace

    from bot.services import App
    a, b = MOVIES[0][0], MOVIES[1][0]
    record_rating(conn, USER, a, 5, "onboarding")
    record_rating(conn, USER, b, 4, "calibration")
    App.undo_answer(SimpleNamespace(conn=conn, rec=SimpleNamespace(profiler=profiler)), USER, a, "onboarding")
    assert [r[0] for r in conn.execute("SELECT tconst FROM ratings WHERE user_id = ?", (USER,))] == [b]
    assert a not in excluded_films(conn, USER)  # фильм снова может попасть в подборки


def test_seen_and_planned_do_not_overlap(conn):
    record_rating(conn, USER, "tt01", 5)  # оценил — в «видел»
    record_reaction(conn, USER, "tt01", "later")  # потом «посмотрю позже» из карточки списка «видел»
    lists = conn.execute("SELECT list FROM user_lists WHERE user_id = ? AND tconst = 'tt01'", (USER,)).fetchall()
    assert [r[0] for r in lists] == ["planned"]
    record_rating(conn, USER, "tt01", 4)
    lists = conn.execute("SELECT list FROM user_lists WHERE user_id = ? AND tconst = 'tt01'", (USER,)).fetchall()
    assert [r[0] for r in lists] == ["seen"]
