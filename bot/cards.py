"""Карточка фильма: постер, название, год, режиссёр, актёры, рейтинг, описание и кнопки."""
import html
import logging
from dataclasses import dataclass
from urllib.parse import quote_plus

import pandas as pd
from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import InlineKeyboardMarkup, InputMediaPhoto

from bot.keyboards import movie_kb
from bot.services import App
from bot.texts import GENRES_RU
from data_pipeline.tmdb import IMAGE_URL, watch_providers

log = logging.getLogger(__name__)
CAPTION_LIMIT = 1024  # ограничение Telegram на подпись к фото


def card_text(movie: dict, directors: list[str], actors: list[str], providers: list[str],
              note: str | None = None) -> str:
    """Текст карточки в HTML; описание обрезается, чтобы уложиться в подпись к фото."""
    e = html.escape
    title = movie.get("title_ru") or movie.get("title")
    year = f" ({int(movie['year'])})" if pd.notna(movie.get("year")) else ""
    lines = [f"<b>{e(title)}</b>{year}"]
    if movie.get("title_ru") and movie.get("title") and movie["title"] != movie["title_ru"]:
        lines.append(f"<i>{e(movie['title'])}</i>")
    facts = []
    if pd.notna(movie.get("rating")):
        facts.append(f"IMDb {movie['rating']:.1f}".replace(".", ","))
    if pd.notna(movie.get("runtime")):
        facts.append(f"{int(movie['runtime'])} мин")
    genres = [GENRES_RU.get(g, g) for g in (movie.get("genres") or "").split(",") if g]
    if genres:
        facts.append(", ".join(genres))
    if facts:
        lines.append(" · ".join(facts))
    if directors:
        lines.append(f"Режиссёр: {e(', '.join(directors))}")
    if actors:
        lines.append(f"В ролях: {e(', '.join(actors))}")
    if providers:
        lines.append(f"Где смотреть: {e(', '.join(providers[:4]))}")
    head = "\n".join(lines)
    tail = f"\n\n<i>{e(note)}</i>" if note else ""
    overview = e(movie.get("overview_ru") or "")
    room = CAPTION_LIMIT - len(head) - len(tail) - 2
    if len(overview) > room:  # обрезаем по слову, чтобы не порвать HTML-сущность вроде &amp;
        overview = overview[: max(room - 1, 0)].rsplit(" ", 1)[0] + "…" if room > 50 else ""
    return head + (f"\n\n{overview}" if overview else "") + tail


@dataclass
class Card:
    tconst: str
    text: str
    photo: str | None
    trailer: str | None
    watch: str | None


def kinopoisk_url(movie: dict) -> str | None:
    """Страница фильма на Кинопоиске (там и кнопка «Смотреть»). ID Кинопоиска берётся из Wikidata;
    без него кнопки нет: поиск по названию находит нужный фильм слишком редко."""
    return f"https://www.kinopoisk.ru/film/{int(movie['kp_id'])}/" if movie.get("kp_id") else None


def youtube_search_url(movie: dict) -> str:
    """Запасная ссылка, если рабочего трейлера нет: поиск трейлера на YouTube."""
    title = movie.get("title_ru") or movie.get("title") or ""
    year = f" {int(movie['year'])}" if pd.notna(movie.get("year")) else ""
    return "https://www.youtube.com/results?search_query=" + quote_plus(f"{title}{year} трейлер")


async def build_card(app: App, tconst: str, note: str | None = None) -> Card:
    movie = dict(app.conn.execute("SELECT * FROM movies WHERE tconst = ?", (tconst,)).fetchone())
    trailer, providers = None, []
    if movie.get("tmdb_id"):
        details = await app.tmdb.movie(movie["tmdb_id"])
        app.conn.commit()  # кэш TMDB
        trailer, providers = await app.tmdb.trailer_url(details), watch_providers(details)
        app.conn.commit()  # кэш проверки YouTube
    directors, actors = app.people_of(tconst)
    photo = IMAGE_URL + movie["poster_path"] if movie.get("poster_path") else None
    return Card(tconst, card_text(movie, directors, actors, providers, note), photo,
                trailer or youtube_search_url(movie), kinopoisk_url(movie))


async def show(bot: Bot, chat_id: int, card: Card, kb: InlineKeyboardMarkup, message_id: int | None = None) -> int:
    """Показать карточку: заменить сообщение message_id на месте или отправить новое. Возвращает id сообщения."""
    if message_id:
        try:
            if card.photo:
                await bot.edit_message_media(InputMediaPhoto(media=card.photo, caption=card.text),
                                             chat_id=chat_id, message_id=message_id, reply_markup=kb)
            else:
                await bot.edit_message_text(card.text, chat_id=chat_id, message_id=message_id, reply_markup=kb)
            return message_id
        except TelegramBadRequest as e:
            # например, сообщение было фото, а у нового фильма нет постера — тогда заменяем сообщение
            log.info("edit failed (%s), sending new message", e)
            try:
                await bot.delete_message(chat_id, message_id)
            except TelegramBadRequest:
                pass
    if card.photo:
        try:
            return (await bot.send_photo(chat_id, card.photo, caption=card.text, reply_markup=kb)).message_id
        except TelegramBadRequest as e:  # Telegram не смог скачать постер — шлём без него
            log.warning("poster failed for %s: %s", card.tconst, e)
    return (await bot.send_message(chat_id, card.text, reply_markup=kb)).message_id


async def send_card(bot: Bot, chat_id: int, app: App, tconst: str, ctx: str, note: str | None = None) -> int:
    """Отдельная карточка (из списка, вопрос «посмотрел?») — новым сообщением."""
    card = await build_card(app, tconst, note)
    return await show(bot, chat_id, card, movie_kb(tconst, ctx, card.trailer, card.watch))
