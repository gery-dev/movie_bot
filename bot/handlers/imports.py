"""Импорт оценок из Letterboxd (/import): пользователь присылает zip или csv из выгрузки."""
import logging
import tempfile
from pathlib import Path

import pandas as pd

from aiogram import Bot, F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from bot.keyboards import nav_kb
from bot.services import App
from bot.states import LetterboxdImport
from data_pipeline.letterboxd import ExportError, import_export
from recsys.profile import rebuild_profile

router = Router()
log = logging.getLogger(__name__)
MAX_SIZE = 20 * 1024 * 1024  # Telegram отдаёт ботам файлы до 20 МБ
HOW_TO = (
    "Перенесу твои оценки из Letterboxd — так подборки сразу станут точнее.\n\n"
    "1. На сайте letterboxd.com открой Settings → Data → Export your data.\n"
    "2. Скачается zip-архив — пришли его сюда файлом (можно и отдельный ratings.csv).\n\n"
    "Оценки, просмотренное и «хочу посмотреть» перенесутся; то, что ты уже оценил в боте, не изменится."
)
# что не так с файлом (код причины — из ExportError.reason и ImportResult.reason)
PROBLEMS = {
    "empty": "Файл пустой.",
    "not_csv": "Не получилось разобрать таблицу.",
    "encoding": "Не смог определить кодировку.",
    "no_name_column": "В нём нет столбца Name — это не файл из выгрузки Letterboxd.",
    "no_known_files": "В архиве нет ratings.csv, watched.csv и watchlist.csv.",
    "bad_zip": "Архив повреждён или защищён паролем.",
    "too_big": "Файл слишком большой для выгрузки Letterboxd: больше 10 МБ, 1000 файлов в архиве или 20 000 фильмов.",
}
NEED_FILES = (
    "Нужна выгрузка Letterboxd: весь zip-архив или один из файлов ratings.csv (оценки), watched.csv (просмотренное), "
    "watchlist.csv («хочу посмотреть») со столбцами Name и Year. Пришли файл ещё раз; передумал — любая команда, "
    "например /help."
)


@router.message(Command("import"))
async def start_import(message: Message, state: FSMContext) -> None:
    await state.clear()
    await state.set_state(LetterboxdImport.file)
    await message.answer(HOW_TO)


@router.message(LetterboxdImport.file, F.document)
async def receive(message: Message, state: FSMContext, bot: Bot, app: App) -> None:
    doc = message.document
    name = (doc.file_name or "").lower()
    if not name.endswith((".zip", ".csv")):
        await message.answer("Нужен zip-архив или csv-файл из выгрузки Letterboxd.")
        return
    if doc.file_size and doc.file_size > MAX_SIZE:
        await message.answer("Файл больше 20 МБ — Telegram не даёт боту его скачать. Пришли только ratings.csv.")
        return
    await message.answer("Импортирую…")
    # файл с чужими данными держим только во временной папке и удаляем сразу после импорта
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / Path(name).name
        await bot.download(doc, destination=path)
        try:
            with app.conn:
                res = await import_export(app.conn, message.from_user.id, path,
                                          app.rec.movies.merge(_extra(app), on="tconst"), app.tmdb)
                rebuild_profile(app.conn, app.rec.profiler, message.from_user.id)
        except ExportError as e:  # в лог — только причина, без содержимого файла
            log.warning("letterboxd import: user %s, file not read, reason=%s", message.from_user.id, e.reason)
            await message.answer(f"{PROBLEMS[e.reason]}\n\n{NEED_FILES}")
            return
        except Exception:
            log.exception("letterboxd import failed")
            await message.answer("Не получилось прочитать файл. Проверь, что это выгрузка Letterboxd.")
            return
    if res.reason:  # файл прочитали, но переносить нечего
        log.warning("letterboxd import: user %s, nothing to import, reason=%s, encoding=%s, rows=%s",
                    message.from_user.id, res.reason, res.encoding, res.rows)
        await message.answer(f"{PROBLEMS[res.reason]}\n\n{NEED_FILES}")
        return
    await state.clear()
    log.info("user %s imported letterboxd: %s rated, %s watched, %s planned, %s unmatched of %s, %.1f s, "
             "encoding=%s, rows=%s", message.from_user.id, res.rated, res.watched, res.planned, len(res.unmatched),
             res.total, res.seconds, res.encoding, res.rows)
    text = (f"Готово за {res.seconds:.0f} с. Нашёл {res.total - len(res.unmatched)} из {res.total} фильмов "
            f"({res.matched_share:.0%}).\n"
            f"• оценок перенесено: {res.rated}\n• в «видел»: {res.watched}\n• в «в планах»: {res.planned}")
    if res.skipped_existing:
        text += f"\n• уже были оценены в боте (оставил как есть): {res.skipped_existing}"
    if res.unmatched:
        sample = ", ".join(res.unmatched[:8]) + ("…" if len(res.unmatched) > 8 else "")
        text += f"\n\nНе нашёл в базе ({len(res.unmatched)}): {sample}"
    await message.answer(text, reply_markup=nav_kb(("Подобрать фильмы по моим оценкам", "menu_recommend")))


@router.message(LetterboxdImport.file)
async def not_a_file(message: Message) -> None:
    await message.answer("Жду файл из выгрузки Letterboxd (zip или csv). Передумал — любая команда, например /help.")


def _extra(app: App):
    """Колонки, которых нет в данных рекомендателя, но нужны для сопоставления."""
    return pd.read_sql("SELECT tconst, original_title, tmdb_id, kind FROM movies", app.conn)
