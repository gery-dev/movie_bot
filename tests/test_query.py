"""Разбор запросов обычным текстом."""
import numpy as np
import pandas as pd

from recsys.query import describe, parse, query_scores, understood

MOVIES = pd.DataFrame([
    ("tt01", "Interstellar", "Интерстеллар", 2014, 169, "Adventure,Drama,Sci-Fi", "en", 900_000),
    ("tt02", "Brother 2", "Брат 2", 2000, 122, "Action,Crime", "ru", 50_000),
    ("tt03", "Brother", "Брат", 1997, 99, "Crime,Drama", "ru", 40_000),
    ("tt04", "Funny", "Смешной", 2023, 90, "Comedy", "en", 10_000),
    ("tt05", "Old Horror", "Вий", 1967, 77, "Horror", "ru", 20_000),
], columns=["tconst", "title", "title_ru", "year", "runtime", "genres", "language", "votes"])


def test_reference_in_quotes_and_modifier():
    p = parse(MOVIES, "хочу что-то как «Интерстеллар», но повеселее")
    assert p.reference == "tt01" and p.genres == {"Comedy": 1.0}


def test_reference_after_kak_without_quotes():
    assert parse(MOVIES, "похоже на брат 2").reference == "tt02"
    assert parse(MOVIES, "как брат").reference == "tt03"  # точное совпадение важнее начала названия


def test_filters_and_understood():
    p = parse(MOVIES, "страшное советское")
    assert p.genres == {"Horror": 1.0} and p.russian and understood(p)
    assert not understood(parse(MOVIES, "фильм про путешествия во времени"))


def test_short_filter_applies():
    p = parse(MOVIES, "что-нибудь смешное и короткое")
    score = query_scores(p, None, None, MOVIES, np.zeros(len(MOVIES)), 2026)
    allowed = MOVIES.tconst[np.isfinite(score)].tolist()
    assert "tt01" not in allowed and "tt04" in allowed
    assert MOVIES.tconst[np.argmax(score)] == "tt04"


def test_describe():
    p = parse(MOVIES, "как «Интерстеллар», но повеселее")
    assert describe(p, MOVIES, "tt04") == "Почему: похож на «Интерстеллар»; комедии"
