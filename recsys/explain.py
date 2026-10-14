"""Объяснение рекомендации: «тебе нравятся Christopher Nolan и фантастика».

Берём признаки фильма с наибольшим вкладом в оценку профиля (вес признака в профиле × его вес
в фильме) и подставляем в шаблон. Плюс любимые люди, совпадение с настроением и коллаборативная часть.
"""
import numpy as np

GENRES_RU = {
    "Drama": "Драма", "Comedy": "Комедия", "Action": "Боевик", "Crime": "Криминал", "Romance": "Мелодрама",
    "Thriller": "Триллер", "Adventure": "Приключения", "Horror": "Ужасы", "Mystery": "Детектив",
    "Fantasy": "Фэнтези", "Biography": "Биография", "Sci-Fi": "Фантастика", "Family": "Семейный",
    "Animation": "Мультфильм", "History": "История", "Documentary": "Документальный", "Music": "Музыка",
    "War": "Военный", "Sport": "Спорт", "Musical": "Мюзикл", "Western": "Вестерн", "Film-Noir": "Нуар",
    "News": "Новости",
}
# как жанр звучит во фразе «тебе нравятся …»
GENRE_TASTE = {
    "Drama": "драмы", "Comedy": "комедии", "Action": "боевики", "Crime": "криминальные фильмы",
    "Romance": "мелодрамы", "Thriller": "триллеры", "Adventure": "приключения", "Horror": "ужасы",
    "Mystery": "детективы", "Fantasy": "фэнтези", "Biography": "биографические фильмы", "Sci-Fi": "фантастика",
    "Family": "семейные фильмы", "Animation": "мультфильмы", "History": "исторические фильмы",
    "Documentary": "документальное кино", "Music": "музыкальные фильмы", "War": "военные фильмы",
    "Sport": "фильмы о спорте", "Musical": "мюзиклы", "Western": "вестерны", "Film-Noir": "нуар",
}
EXPLAINED = ("director:", "actor:", "genre:", "decade:")  # страна, язык и длительность звучат неестественно
CF_STRONG = 1.0  # насколько ALS должна «тянуть» фильм сильнее профиля, чтобы сказать о похожих людях


def feature_label(feature: str, names: dict[str, str]) -> str | None:
    kind, value = feature.split(":", 1)
    if kind in ("director", "actor"):
        return names.get(value)
    if kind == "genre":
        return GENRE_TASTE.get(value)
    if kind == "decade":
        return f"фильмы {value[2:]}-х" if value < "2000" else f"фильмы {value}-х"
    return None


def top_features(profile: np.ndarray, row, feature_names: list[str], n: int = 2) -> list[str]:
    """Признаки фильма (строка разреженной матрицы) с наибольшим положительным вкладом в оценку."""
    row = row.tocsr()
    contrib = profile[row.indices] * row.data
    order = np.argsort(-contrib)
    picked = []
    for i in order:
        if contrib[i] <= 0:
            break
        name = feature_names[row.indices[i]]
        if name.startswith(EXPLAINED):
            picked.append(name)
        if len(picked) == n:
            break
    return picked


def explain(*, taste: list[str], favorites: list[str], mood: bool, cf_driven: bool, popular: bool) -> str:
    """Собрать фразу из готовых частей; не больше двух причин, чтобы читалось легко."""
    reasons = []
    if favorites:
        reasons.append(f"здесь {', '.join(favorites[:2])} — из твоих любимых")
    if taste:
        reasons.append("тебе нравятся " + " и ".join(taste))
    elif cf_driven:
        reasons.append("его высоко ценят люди с похожим на твой вкусом")
    if mood:
        reasons.append("подходит под настроение")
    if not reasons:
        reasons.append("один из самых любимых зрителями фильмов" if popular else "может тебе подойти")
    return "Почему: " + "; ".join(reasons[:2])
