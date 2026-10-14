"""Проверка окружения (этап 0): ключи из .env, запрос к TMDB, эхо-бот.

    python check_setup.py tmdb   # тестовый запрос к TMDB
    python check_setup.py bot    # эхо-бот, остановка Ctrl+C
"""
import asyncio
import os
import sys

import httpx
from dotenv import load_dotenv

load_dotenv()

INTERSTELLAR_IMDB_ID = "tt0816692"


def check_tmdb() -> None:
    key = os.getenv("TMDB_API_KEY", "").strip()
    if not key:
        sys.exit("TMDB_API_KEY пуст в .env")

    # TMDB выдаёт два вида ключа: короткий API Key (v3) и длинный Read Access Token (JWT)
    if key.startswith("eyJ"):
        auth = {"headers": {"Authorization": f"Bearer {key}"}, "params": {}}
    else:
        auth = {"headers": {}, "params": {"api_key": key}}

    try:
        r = httpx.get(
            f"https://api.themoviedb.org/3/find/{INTERSTELLAR_IMDB_ID}",
            params={"external_source": "imdb_id", "language": "ru-RU", **auth["params"]},
            headers=auth["headers"],
            timeout=15,
        )
    except httpx.HTTPError as e:
        sys.exit(f"TMDB недоступен: {e!r}")

    if r.status_code == 401:
        sys.exit("TMDB: ключ не принят (401)")
    r.raise_for_status()
    movies = r.json().get("movie_results", [])
    if not movies:
        sys.exit("TMDB ответил, но фильм не найден")
    m = movies[0]
    print(f"TMDB ok: {m['title']} ({m['release_date'][:4]}), постер: {bool(m.get('poster_path'))}")


async def run_echo_bot() -> None:
    from aiogram import Bot, Dispatcher
    from aiogram.filters import CommandStart
    from aiogram.types import Message

    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        sys.exit("TELEGRAM_BOT_TOKEN пуст в .env")

    bot = Bot(token)
    dp = Dispatcher()

    @dp.message(CommandStart())
    async def start(message: Message) -> None:
        await message.answer("Бот запущен. Напиши что-нибудь, я повторю.")

    @dp.message()
    async def echo(message: Message) -> None:
        await message.answer(message.text or "(не текст)")

    me = await bot.get_me()
    print(f"Бот @{me.username} запущен, напиши ему /start. Остановка: Ctrl+C")
    await dp.start_polling(bot)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "tmdb":
        check_tmdb()
    elif mode == "bot":
        asyncio.run(run_echo_bot())
    else:
        sys.exit(__doc__)
