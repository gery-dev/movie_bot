"""Русские имена актёров и режиссёров — для поиска по-русски и подписей в боте.

    python -m data_pipeline.people_ru   # записать people.name_ru и people.aka_ru

Имена в IMDb только латиницей. Русское имя и другие написания (например, «Ди Каприо»)
берутся из Wikidata по IMDb ID (свойство P345), лицензия CC0: https://www.wikidata.org
"""
import httpx

from data_pipeline.kinopoisk import BATCH, SPARQL_URL, USER_AGENT
from db.database import connect

QUERY = """SELECT ?imdb ?label (GROUP_CONCAT(DISTINCT ?alias; separator="|") AS ?aliases) WHERE {{
  VALUES ?imdb {{ {} }}
  ?p wdt:P345 ?imdb; rdfs:label ?label . FILTER(LANG(?label) = "ru")
  OPTIONAL {{ ?p skos:altLabel ?alias . FILTER(LANG(?alias) = "ru") }}
}} GROUP BY ?imdb ?label"""


def fetch(nconsts: list[str]) -> dict[str, tuple[str, str | None]]:
    names: dict[str, tuple[str, str | None]] = {}
    for start in range(0, len(nconsts), BATCH):
        values = " ".join(f'"{n}"' for n in nconsts[start:start + BATCH])
        r = httpx.post(SPARQL_URL, data={"query": QUERY.format(values)}, timeout=120,
                       headers={"Accept": "application/sparql-results+json", "User-Agent": USER_AGENT})
        r.raise_for_status()
        for b in r.json()["results"]["bindings"]:
            names.setdefault(b["imdb"]["value"], (b["label"]["value"], b["aliases"]["value"] or None))
    return names


def main() -> None:
    conn = connect()
    nconsts = [n for (n,) in conn.execute("SELECT nconst FROM people")]
    names = fetch(nconsts)
    with conn:
        conn.executemany("UPDATE people SET name_ru = ?, aka_ru = ? WHERE nconst = ?",
                         [(ru, aka, n) for n, (ru, aka) in names.items()])
    print(f"Людей в БД: {len(nconsts)}, с русским именем: {len(names)} ({len(names) / len(nconsts):.1%})")


if __name__ == "__main__":
    main()
