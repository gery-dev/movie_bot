"""Тесты профиля, профиля на сегодня, избранного и калибровки на маленькой тестовой БД."""
import numpy as np
import pandas as pd
import pytest

from db.database import connect
from recsys import calibration
from recsys.favorites import add_favorite, person_unseen, suggest_favorites
from recsys.profile import (DAY, Profiler, decay, excluded_films, rating_signal, rebuild_profile, record_rating,
                            record_reaction, load_profile, user_events)
from recsys.recommender import Recommender
from recsys.today import load_daily, save_daily, today_filter, today_scores

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
    before = rec.recommend(conn, USER, k=9, answers={}).tconst.tolist()
    add_favorite(conn, USER, "nm2")
    after = rec.recommend(conn, USER, k=9, answers={}).tconst.tolist()
    assert after.index("tt04") < before.index("tt04")


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
