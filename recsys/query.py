"""Запрос обычным текстом: «хочу что-то как „Интерстеллар“, но повеселее», «фильм про путешествия во времени».

1. Ищем фильм-образец: название в кавычках или после «как», «типа», «похож на». Найден — берём фильмы,
   близкие к нему по описанию (эмбеддинги) и признакам (жанры, режиссёр, актёры).
2. Слова-пожелания («повеселее», «пострашнее», «покороче», «советское»…) поднимают жанры или фильтруют.
3. Если образца нет — сравниваем смысл всего запроса с описаниями фильмов (та же модель эмбеддингов).
4. Немного учитываем вкус пользователя, чтобы из похожих выбрать подходящие ему.
Языковую модель не подключаем: на этапе 2 решили начать с эмбеддингов (бесплатно, без внешнего API).
"""
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from recsys.explain import GENRE_TASTE
from recsys.models import _unit

# основа слова → (жанр, вес); ищется как подстрока в нижнем регистре
MODIFIERS = {
    "весел": ("Comedy", 1.0), "весёл": ("Comedy", 1.0), "смешн": ("Comedy", 1.0), "комеди": ("Comedy", 1.0),
    "легк": ("Comedy", 0.5), "лёгк": ("Comedy", 0.5), "поржать": ("Comedy", 1.0),
    "страшн": ("Horror", 1.0), "ужас": ("Horror", 1.0), "жутк": ("Horror", 1.0), "хоррор": ("Horror", 1.0),
    "напряж": ("Thriller", 1.0), "триллер": ("Thriller", 1.0), "остросюжет": ("Thriller", 1.0),
    "романт": ("Romance", 1.0), "любов": ("Romance", 1.0), "мелодрам": ("Romance", 1.0),
    "грустн": ("Drama", 1.0), "драм": ("Drama", 1.0), "трогательн": ("Drama", 1.0), "глубок": ("Drama", 0.5),
    "боевик": ("Action", 1.0), "экшн": ("Action", 1.0), "динамичн": ("Action", 1.0), "драк": ("Action", 0.5),
    "фантаст": ("Sci-Fi", 1.0), "космос": ("Sci-Fi", 1.0), "фэнтези": ("Fantasy", 1.0), "волшебн": ("Fantasy", 1.0),
    "мульт": ("Animation", 1.0), "анимац": ("Animation", 1.0), "семейн": ("Family", 1.0), "детск": ("Family", 1.0),
    "детектив": ("Mystery", 1.0), "загадк": ("Mystery", 1.0), "расследован": ("Crime", 0.5),
    "криминал": ("Crime", 1.0), "мафи": ("Crime", 1.0), "приключен": ("Adventure", 1.0),
    "военн": ("War", 1.0), "войн": ("War", 1.0), "историческ": ("History", 1.0), "биограф": ("Biography", 1.0),
    "документальн": ("Documentary", 1.0), "мюзикл": ("Musical", 1.0), "вестерн": ("Western", 1.0),
    "спорт": ("Sport", 1.0),
}
SHORT = ("коротк", "недлинн", "на вечер")       # ≤ 100 минут
RUSSIAN = ("русск", "советск", "отечествен", "российск")
NEW = ("нов", "свеж", "недавн")
OLD = ("стар", "классик", "классическ")
# откуда берётся образец: «как X», «типа X», «похож… на X», «вроде X»
REF_PATTERN = re.compile(r"(?:как|типа|вроде|похож\w*\s+на|наподобие)\s+(.+?)(?:,|\s+но\s|\s+только\s|$)", re.I)
QUOTED = re.compile(r"[«\"“'](.+?)[»\"”']")

# веса подобраны вручную на примерах из docstring: при W_MOD < 1 «повеселее» почти не меняло выдачу,
# без известности (W_POP) запросы из одних пожеланий выдавали забытые фильмы
# W_POP — для запросов из одних пожеланий; с образцом или поиском по смыслу известность почти не нужна,
# иначе к «Брату 2» примешиваются «Крёстный отец» и «Шоушенк»
W_REF, W_TEXT, W_MOD, W_TASTE, W_POP, W_POP_SIM = 1.0, 1.0, 1.5, 0.3, 0.4, 0.1


@dataclass
class Parsed:
    reference: str | None = None              # tconst фильма-образца
    genres: dict[str, float] = field(default_factory=dict)
    short: bool = False
    russian: bool = False
    era: str | None = None                    # new | old


def find_title(movies: pd.DataFrame, text: str) -> str | None:
    """Фильм по названию: точное совпадение, затем начало, затем вхождение; среди равных — известнее."""
    q = text.strip().lower().strip(" .!?")
    if len(q) < 2:
        return None
    ru, en = movies.title_ru.fillna("").str.lower(), movies.title.fillna("").str.lower()
    for mask in [(ru == q) | (en == q), ru.str.startswith(q) | en.str.startswith(q)]:
        hits = movies[mask]
        if len(hits):
            return hits.sort_values("votes", ascending=False).tconst.iloc[0]
    return None


def parse(movies: pd.DataFrame, text: str) -> Parsed:
    low = text.lower()
    p = Parsed()
    candidates = QUOTED.findall(text) + REF_PATTERN.findall(text)
    for c in candidates:
        if (t := find_title(movies, c)) is not None:
            p.reference = t
            break
    # слова из названия образца не должны считаться пожеланиями («Весёлые ребята»)
    rest = low
    for c in candidates:
        rest = rest.replace(c.lower(), " ")
    for stem, (genre, w) in MODIFIERS.items():
        if stem in rest:
            p.genres[genre] = max(p.genres.get(genre, 0), w)
    p.short = any(s in rest for s in SHORT)
    p.russian = any(s in rest for s in RUSSIAN)
    p.era = "new" if any(re.search(rf"\b{s}", rest) for s in NEW) else "old" if any(s in rest for s in OLD) else None
    return p


def describe(p: Parsed, movies: pd.DataFrame, tconst: str) -> str:
    """«Почему: похож на «Интерстеллар»; комедии» для карточки."""
    parts = []
    if p.reference:
        ref = movies.set_index("tconst").loc[p.reference]
        parts.append(f"похож на «{ref.title_ru or ref.title}»")
    film_genres = set((movies.set_index("tconst").loc[tconst].genres or "").split(","))
    matched = [GENRE_TASTE.get(g, g) for g in p.genres if g in film_genres]
    if matched:
        parts.append(", ".join(matched))
    if not parts:
        parts.append("по смыслу запроса")
    return "Почему: " + "; ".join(parts)


def understood(p: Parsed) -> bool:
    """Удалось ли что-то разобрать; если нет — ищем по смыслу всего текста."""
    return bool(p.reference or p.genres or p.short or p.russian or p.era)


def query_scores(p: Parsed, text_sim: np.ndarray | None, ref_sim: np.ndarray | None, movies: pd.DataFrame,
                 taste: np.ndarray, now_year: int) -> np.ndarray:
    """Оценка каждого фильма каталога; -inf — не проходит жёсткие условия запроса."""
    fame = np.log10(np.maximum(movies.votes.fillna(1).to_numpy(float), 1))
    w_pop = W_POP_SIM if (ref_sim is not None or text_sim is not None) else W_POP
    # вкус — через ранги: у любимых фильмов оценка вкуса уходит в длинный хвост и иначе перебивает запрос
    taste_rank = np.argsort(np.argsort(np.nan_to_num(taste, neginf=-1e9))).astype(float)
    score = W_TASTE * _unit(taste_rank) + w_pop * _unit(fame)
    if ref_sim is not None:
        score = score + W_REF * _unit(ref_sim)
    elif text_sim is not None:
        score = score + W_TEXT * _unit(text_sim)
    if p.genres:
        genres = movies.genres.fillna("").str.split(",")
        boost = np.array([sum(p.genres.get(g, 0) for g in gs) for gs in genres])
        score = score + W_MOD * _unit(boost)
    ok = np.ones(len(movies), bool)
    if p.short:
        ok &= movies.runtime.fillna(999).to_numpy() <= 100
    if p.russian:
        ok &= movies.language.fillna("").to_numpy() == "ru"
    year = movies.year.fillna(0).to_numpy()
    if p.era == "new":
        ok &= year >= now_year - 5
    elif p.era == "old":
        ok &= year < 1990
    return np.where(ok, score, -np.inf)
