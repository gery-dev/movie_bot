"""Клиент TMDB с кэшем ответов в таблице api_cache.

    python -m data_pipeline.tmdb enrich --limit 100  # дополнить фильмы из TMDB (без --limit — все)

По IMDb ID находит фильм в TMDB и записывает в movies: tmdb_id, страны, язык,
описание на русском, постер и русское название, если его не было.
Повторный запрос того же адреса берётся из кэша, в TMDB не уходит.
Источник данных указывается в боте и отчёте: https://www.themoviedb.org
"""
import argparse
import asyncio
import json
import os
import re
import sqlite3
import time

import httpx
from dotenv import load_dotenv

from db.database import connect

API_URL = "https://api.themoviedb.org/3"
IMAGE_URL = "https://image.tmdb.org/t/p/w500"
CONCURRENCY = 20  # TMDB допускает около 50 запросов в секунду
BATCH = 500
REGION = "RU"  # для какой страны храним сервисы просмотра
NOW_PLAYING_PAGES = 2     # по 20 фильмов: сейчас в прокате, самые обсуждаемые первыми
NOW_PLAYING_HOURS = 12    # как часто обновлять список проката
# жанры TMDB → названия IMDb, как в таблице movies (у «TV Movie» пары нет)
GENRES = {28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy", 80: "Crime", 99: "Documentary",
          18: "Drama", 10751: "Family", 14: "Fantasy", 36: "History", 27: "Horror", 10402: "Music",
          9648: "Mystery", 10749: "Romance", 878: "Sci-Fi", 53: "Thriller", 10752: "War", 37: "Western"}
YOUTUBE_RECHECK_DAYS = 30  # через сколько дней заново проверять, что ролик на YouTube ещё доступен
CYRILLIC = re.compile("[а-яё]", re.IGNORECASE)


class TMDB:
    def __init__(self, conn: sqlite3.Connection):
        load_dotenv()
        key = os.getenv("TMDB_API_KEY", "").strip()
        if not key:
            raise RuntimeError("TMDB_API_KEY пуст в .env")
        # длинный Read Access Token передаётся заголовком, короткий API Key — параметром
        self.auth_headers = {"Authorization": f"Bearer {key}"} if key.startswith("eyJ") else {}
        self.auth_params = {} if key.startswith("eyJ") else {"api_key": key}
        self.conn = conn
        self.client = httpx.AsyncClient(base_url=API_URL, headers=self.auth_headers, timeout=20)
        self.web = httpx.AsyncClient(timeout=5)  # для YouTube: без ключа TMDB в заголовках
        self.limit = asyncio.Semaphore(CONCURRENCY)
        self.requests = 0

    async def close(self) -> None:
        await self.client.aclose()
        await self.web.aclose()

    async def get(self, path: str, shrink=None, max_age_hours: float | None = None, **params) -> dict:
        """GET с кэшем; shrink(data) убирает из ответа лишнее перед сохранением.
        max_age_hours — для меняющихся списков: кэш старше этого запрашивается заново."""
        key = path + "?" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        fresh = "" if max_age_hours is None else f" AND fetched_at > datetime('now', '-{float(max_age_hours)} hours')"
        row = self.conn.execute("SELECT response FROM api_cache WHERE source = 'tmdb' AND key = ?" + fresh,
                                (key,)).fetchone()
        if row:
            return json.loads(row[0])

        async with self.limit:
            for attempt in range(5):
                r = await self.client.get(path, params={**params, **self.auth_params})
                self.requests += 1
                if r.status_code == 429:  # превышен лимит: ждём, сколько попросил сервер
                    await asyncio.sleep(float(r.headers.get("Retry-After", 1 + attempt)))
                    continue
                break
        if r.status_code == 404:
            data = {}
        else:
            r.raise_for_status()
            data = shrink(r.json()) if shrink else r.json()
        self.conn.execute(
            "INSERT OR REPLACE INTO api_cache (source, key, response) VALUES ('tmdb', ?, ?)",
            (key, json.dumps(data, ensure_ascii=False)),
        )
        return data

    async def find_movie_id(self, tconst: str) -> int | None:
        data = await self.get(f"/find/{tconst}", external_source="imdb_id")
        results = data.get("movie_results", [])
        return results[0]["id"] if results else None

    async def movie(self, tmdb_id: int) -> dict:
        """Карточка фильма на русском с трейлерами (рус., затем англ.) и сервисами просмотра."""
        return await self.get(
            f"/movie/{tmdb_id}", shrink=shrink_movie, language="ru-RU",
            append_to_response="videos,watch/providers", include_video_language="ru,en",
        )

    async def now_playing(self) -> list[dict]:
        """Фильмы в прокате по всему миру (без привязки к стране — в российском списке почти одни местные
        премьеры), отсортированные по популярности TMDB: просмотры страниц, поиски, добавления в списки."""
        pages = [await self.get("/movie/now_playing", max_age_hours=NOW_PLAYING_HOURS, language="ru-RU", page=p)
                 for p in range(1, NOW_PLAYING_PAGES + 1)]
        films = {m["id"]: m for page in pages for m in page.get("results", [])}
        return sorted(films.values(), key=lambda m: -m.get("popularity", 0))

    async def trailer_url(self, movie: dict) -> str | None:
        """Первый доступный трейлер: сначала русские, потом английские. None — рабочего нет."""
        for key in trailer_keys(movie):
            if await self.youtube_ok(key):
                return f"https://www.youtube.com/watch?v={key}"
        return None

    async def youtube_ok(self, key: str) -> bool:
        """Ролик существует и открыт: oEmbed YouTube отвечает 404 на удалённые и 403 на закрытые
        (401 — запрещено встраивание, но смотреть можно). Ответ кэшируется на YOUTUBE_RECHECK_DAYS."""
        row = self.conn.execute(
            "SELECT response FROM api_cache WHERE source = 'youtube' AND key = ? AND fetched_at > datetime('now', ?)",
            (key, f"-{YOUTUBE_RECHECK_DAYS} days")).fetchone()
        if row:
            return json.loads(row[0])
        try:
            r = await self.web.get("https://www.youtube.com/oembed",
                                   params={"url": f"https://www.youtube.com/watch?v={key}", "format": "json"})
        except httpx.HTTPError:
            return True  # YouTube не ответил — не прячем трейлер и не кэшируем
        if r.status_code not in (200, 401, 403, 404):
            return True
        ok = r.status_code in (200, 401)
        self.conn.execute(
            "INSERT OR REPLACE INTO api_cache (source, key, response, fetched_at) VALUES ('youtube', ?, ?, CURRENT_TIMESTAMP)",
            (key, json.dumps(ok)))
        return ok

    async def search_person(self, query: str, limit: int = 5) -> list[tuple[str, str]]:
        """Поиск человека по имени на любом языке (например, «Нолан») → [(IMDb ID, имя)]."""
        data = await self.get("/search/person", query=query.strip(), language="ru-RU")
        people = data.get("results", [])[:limit]
        ids = await asyncio.gather(*(self.get(f"/person/{p['id']}/external_ids") for p in people))
        return [(x["imdb_id"], p["name"]) for p, x in zip(people, ids) if x.get("imdb_id")]

    async def overview_en(self, tmdb_id: int) -> str | None:
        """Описание на английском — для фильмов, у которых нет русского."""
        data = await self.get(f"/movie/{tmdb_id}", shrink=lambda m: {"overview": m.get("overview")}, language="en-US")
        return data.get("overview") or None


def shrink_movie(movie: dict) -> dict:
    # сервисы просмотра TMDB отдаёт сразу для всех стран (~85 КБ на фильм): оставляем Россию,
    # из видео — только трейлеры с YouTube
    providers = movie.get("watch/providers", {}).get("results", {})
    movie["watch/providers"] = {"results": {r: providers[r] for r in [REGION] if r in providers}}
    videos = movie.get("videos", {}).get("results", [])
    movie["videos"] = {"results": [v for v in videos if v.get("site") == "YouTube" and v.get("type") == "Trailer"]}
    return movie


def poster_url(movie: dict) -> str | None:
    return IMAGE_URL + movie["poster_path"] if movie.get("poster_path") else None


def trailer_keys(movie: dict) -> list[str]:
    """Ключи трейлеров на YouTube: сначала русские, потом английские."""
    videos = movie.get("videos", {}).get("results", [])
    trailers = [v for v in videos if v.get("site") == "YouTube" and v.get("type") == "Trailer"
                and v.get("iso_639_1") in ("ru", "en")]
    trailers.sort(key=lambda v: v.get("iso_639_1") != "ru")
    return [v["key"] for v in trailers]


def watch_providers(movie: dict) -> list[str]:
    region_data = movie.get("watch/providers", {}).get("results", {}).get(REGION, {})
    names = [p["provider_name"] for kind in ("flatrate", "rent", "buy") for p in region_data.get(kind, [])]
    return list(dict.fromkeys(names))


def movie_row(m: dict) -> tuple | None:
    """Карточка TMDB → строка movies (для новинки, которой ещё нет в выгрузке IMDb). Рейтинга IMDb
    у неё ещё нет — rating пустой, votes = 0."""
    if not m.get("imdb_id"):
        return None
    title = m.get("title") or ""
    return (
        m["imdb_id"], m.get("original_title") or title, title if CYRILLIC.search(title) else None,
        int(m["release_date"][:4]) if m.get("release_date") else None, m.get("runtime") or None,
        ",".join(GENRES[g["id"]] for g in m.get("genres", []) if g["id"] in GENRES) or None,
        m["id"], ",".join(m.get("origin_country") or []) or None, m.get("original_language") or None,
        m.get("overview") or None, m.get("poster_path") or None,
    )


async def enrich_one(tmdb: TMDB, tconst: str, title_ru: str | None) -> tuple | None:
    tmdb_id = await tmdb.find_movie_id(tconst)
    if tmdb_id is None:
        return None
    m = await tmdb.movie(tmdb_id)
    title = m.get("title") or ""
    new_title_ru = title_ru or (title if CYRILLIC.search(title) else None)
    return (
        tmdb_id,
        ",".join(m.get("origin_country") or []) or None,
        m.get("original_language") or None,
        m.get("overview") or None,
        m.get("poster_path") or None,
        new_title_ru,
        tconst,
    )


async def enrich(limit: int | None) -> None:
    conn = connect()
    tmdb = TMDB(conn)
    sql = "SELECT tconst, title_ru FROM movies WHERE tmdb_id IS NULL ORDER BY votes DESC"
    todo = conn.execute(sql + (f" LIMIT {int(limit)}" if limit else "")).fetchall()
    print(f"Фильмов без данных TMDB: {len(todo)}")

    t0, found, missing = time.time(), 0, []
    try:
        for start in range(0, len(todo), BATCH):
            batch = todo[start:start + BATCH]
            results = await asyncio.gather(*(enrich_one(tmdb, r["tconst"], r["title_ru"]) for r in batch))
            rows = [r for r in results if r]
            missing += [b["tconst"] for b, r in zip(batch, results) if r is None]
            with conn:
                conn.executemany(
                    """UPDATE movies SET tmdb_id = ?, country = ?, language = ?, overview_ru = ?,
                       poster_path = ?, title_ru = ? WHERE tconst = ?""",
                    rows,
                )
            found += len(rows)
            print(f"  {start + len(batch)}/{len(todo)}, запросов к TMDB: {tmdb.requests}, {time.time() - t0:.0f} с")
    finally:
        conn.commit()  # кэш сохраняется, даже если прервали
        await tmdb.close()
        conn.close()

    print(f"Найдено в TMDB: {found}, не найдено: {len(missing)}")
    if missing:
        print("  не найдены, например:", ", ".join(missing[:10]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["enrich"])
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    asyncio.run(enrich(args.limit))
