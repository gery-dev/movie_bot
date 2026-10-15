"""ID Кинопоиска для фильмов — чтобы кнопка в карточке вела прямо на страницу фильма.

    python -m data_pipeline.kinopoisk   # записать movies.kp_id

Соответствие IMDb ID → ID Кинопоиска берётся из Wikidata (свойства P345 и P2603), лицензия CC0:
https://www.wikidata.org. Поиск Кинопоиска по названию для этого не годится: при расхождении года
или названия он ничего не находит.
"""
import httpx

from db.database import connect

SPARQL_URL = "https://query.wikidata.org/sparql"
# весь список (~250 тыс. пар) Wikidata отдаёт не всегда целиком, поэтому спрашиваем только свои фильмы пачками
QUERY = "SELECT ?imdb ?kp WHERE {{ VALUES ?imdb {{ {} }} ?film wdt:P345 ?imdb; wdt:P2603 ?kp . }}"
BATCH = 2000
# Wikidata отклоняет запросы без понятного User-Agent
USER_AGENT = "movie_bot/0.1 (https://github.com/gery-dev/movie_bot)"


def fetch(tconsts: list[str]) -> dict[str, int]:
    ids: dict[str, int] = {}
    for start in range(0, len(tconsts), BATCH):
        values = " ".join(f'"{t}"' for t in tconsts[start:start + BATCH])
        r = httpx.post(SPARQL_URL, data={"query": QUERY.format(values)}, timeout=120,
                       headers={"Accept": "application/sparql-results+json", "User-Agent": USER_AGENT})
        r.raise_for_status()
        for b in r.json()["results"]["bindings"]:
            if b["kp"]["value"].isdigit():
                ids.setdefault(b["imdb"]["value"], int(b["kp"]["value"]))  # бывает несколько ID — берём первый
    return ids


def main() -> None:
    conn = connect()
    tconsts = [t for (t,) in conn.execute("SELECT tconst FROM movies")]
    ids = fetch(tconsts)
    with conn:
        conn.executemany("UPDATE movies SET kp_id = ? WHERE tconst = ?", [(k, t) for t, k in ids.items()])
    print(f"Фильмов в БД: {len(tconsts)}, с ID Кинопоиска: {len(ids)} ({len(ids) / len(tconsts):.1%})")


if __name__ == "__main__":
    main()
