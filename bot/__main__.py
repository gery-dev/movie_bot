"""Запуск бота: python -m bot (остановка — Ctrl+C)."""
import asyncio
import logging
import os
import sys

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import BotCommand
from dotenv import load_dotenv

from bot.feedback import feedback_loop
from bot.handlers import common, lists, onboarding, recommend
from bot.services import App
from db.database import ROOT
from recsys.today import extend_daily

COMMANDS = [
    BotCommand(command="recommend", description="Подборка по вкусу"),
    BotCommand(command="today", description="Подборка под настроение"),
    BotCommand(command="lists", description="Видел / в планах / скрыто"),
    BotCommand(command="favorites", description="Любимые актёры и режиссёры"),
    BotCommand(command="restart", description="Пройти знакомство заново"),
    BotCommand(command="help", description="Справка"),
]


def setup_logging() -> None:
    # в лог пишем только ID пользователей и действия, без текстов сообщений
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(ROOT / "logs" / "bot.log", encoding="utf-8"), logging.StreamHandler()],
    )
    # httpx пишет адреса запросов целиком, а в них ключ TMDB — такие строки в лог не пускаем
    logging.getLogger("httpx").setLevel(logging.WARNING)


async def main() -> None:
    load_dotenv()
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        sys.exit("TELEGRAM_BOT_TOKEN пуст в .env")
    setup_logging()
    log = logging.getLogger("bot")

    log.info("загрузка данных и моделей…")
    app = App()
    bot = Bot(token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher()
    dp["app"] = app

    @dp.update.outer_middleware()
    async def keep_today_alive(handler, event, data):
        """Любое действие в боте продлевает ответы /today: они сгорают только после 22 часов тишины."""
        user = data.get("event_from_user")
        if user:
            with app.conn:
                extend_daily(app.conn, user.id)
        return await handler(event, data)

    # порядок важен: обработчики шагов знакомства раньше общих, «всё остальное» — последним
    dp.include_routers(onboarding.router, recommend.router, lists.router, common.router)
    await bot.set_my_commands(COMMANDS)

    feedback = asyncio.create_task(feedback_loop(bot, app))
    me = await bot.get_me()
    log.info("бот @%s запущен", me.username)
    try:
        await dp.start_polling(bot)
    finally:
        feedback.cancel()
        await app.tmdb.close()
        app.conn.close()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
