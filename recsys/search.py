"""Нечёткий поиск по названиям фильмов и именам людей: находит и с опечатками, «е» вместо «ё»,
по-русски и латиницей.

Каждое написание (русское, английское, оригинальное название; имя латиницей, по-русски и другие
русские варианты) разбивается на куски по 2–3 буквы, запрос сравнивается с ними по TF-IDF —
одна-две опечатки меняют лишь несколько кусков. Запрос кириллицей дополнительно сравнивается
в транслитерации: так находятся люди, у которых нет русского имени в Wikidata.
"""
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

MIN_SIM = 0.45      # ниже — уже не опечатка, а другое название
W_POP = 0.5         # известность поднимает при близком сходстве («Нолан» → Кристофер, а не однофамилец)
EXACT_BONUS = 0.3   # написание совпало целиком
PREFIX_BONUS = 0.1  # написание начинается с запроса
CANDIDATES = 200    # бонусы считаем только у самых похожих

CYRILLIC = re.compile("[а-я]")
TRANSLIT = dict(zip("абвгдезийклмнопрстуфхцчшщъыьэюя",
                    ["a", "b", "v", "g", "d", "e", "z", "i", "y", "k", "l", "m", "n", "o", "p", "r", "s", "t",
                     "u", "f", "h", "ts", "ch", "sh", "sch", "", "y", "", "e", "yu", "ya"]))
TRANSLIT["ж"] = "zh"


def normalize(text: str) -> str:
    text = text.lower().replace("ё", "е")
    return " ".join(re.sub(r"[^\w+]+", " ", text).split())


def translit(text: str) -> str:
    return "".join(TRANSLIT.get(c, c) for c in text)


class NameIndex:
    def __init__(self, ids: list, variants: list[list[str | None]], popularity: list[float]):
        names, owners = [], []
        for i, group in enumerate(variants):
            for name in {normalize(v) for v in group if isinstance(v, str)} - {""}:
                names.append(name)
                owners.append(i)
        self.ids, self.names, self.owner = list(ids), names, np.array(owners)
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3))
        self.matrix = self.vectorizer.fit_transform(names)
        pop = np.log1p(np.maximum(np.asarray(popularity, dtype=float), 0))
        self.pop = pop / pop.max() if pop.max() > 0 else pop

    def search(self, query: str, limit: int = 5) -> list:
        q = normalize(query)
        if len(q) < 2:
            return []
        queries = [q] + ([translit(q)] if CYRILLIC.search(q) else [])
        sims = np.max([(self.matrix @ self.vectorizer.transform([x]).T).toarray().ravel() for x in queries], axis=0)
        rows = np.argsort(-sims)[:CANDIDATES]
        best: dict[int, float] = {}
        for r in rows:
            if sims[r] < MIN_SIM:
                break
            name = self.names[r]
            bonus = EXACT_BONUS if name == q else PREFIX_BONUS if name.startswith(q) else 0.0
            owner = self.owner[r]
            best[owner] = max(best.get(owner, 0.0), sims[r] + bonus)
        order = sorted(best, key=lambda o: -(best[o] + W_POP * self.pop[o]))
        return [self.ids[o] for o in order[:limit]]
