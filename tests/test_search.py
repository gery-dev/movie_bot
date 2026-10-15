"""Нечёткий поиск: опечатки, «ё», русский и латиница, известность при равном сходстве."""
from recsys.search import NameIndex, normalize, translit

FILMS = NameIndex(
    ["tt0816692", "tt0068646", "tt0118767", "tt0118768"],
    [["Интерстеллар", "Interstellar", "Interstellar"], ["Крёстный отец", "The Godfather", None],
     ["Брат", "Brother", "Brat"], ["Братство", "Fellowship", None]],
    [2_000_000, 2_000_000, 100_000, 5_000_000],
)
PEOPLE = NameIndex(
    ["nm0634240", "nm0634300", "nm0000138"],
    [["Christopher Nolan", "Кристофер Нолан"], ["John Nolan", None], ["Leonardo DiCaprio", "Леонардо Ди Каприо", "Дикаприо"]],
    [20_000_000, 50_000, 20_000_000],
)


def test_normalize_and_translit():
    assert normalize("  Крёстный  отец!") == "крестный отец"
    assert translit("нолан") == "nolan"


def test_typos_and_yo():
    assert FILMS.search("интерстелар")[0] == "tt0816692"
    assert FILMS.search("Interstelar")[0] == "tt0816692"
    assert FILMS.search("крестный атец")[0] == "tt0068646"


def test_exact_title_beats_popular_longer_one():
    assert FILMS.search("брат")[0] == "tt0118767"


def test_people_russian_latin_and_translit():
    assert PEOPLE.search("Кристофер Нолон")[0] == "nm0634240"
    assert PEOPLE.search("дикаприо")[0] == "nm0000138"
    assert PEOPLE.search("Нолан")[0] == "nm0634240"   # известнее однофамильца
    assert "nm0634300" in PEOPLE.search("Джон Нолан")  # русского имени нет — находим транслитерацией


def test_junk_finds_nothing():
    assert FILMS.search("ывапролдж") == [] and PEOPLE.search("qwertyuiop") == []
