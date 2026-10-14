# Movie Bot

Telegram-бот для персональных рекомендаций фильмов. Замысел и ТЗ лежат на уровень выше: `../movie_bot_plan.pdf`, `../ТЗ_бот_рекомендаций_фильмов.pdf`.

## Структура

| Папка | Что внутри |
|-------|------------|
| `bot/` | Бот на aiogram: команды, клавиатуры, состояния диалога |
| `recsys/` | Модели рекомендаций, профиль пользователя, MMR, объяснения |
| `data_pipeline/` | Загрузка IMDb и MovieLens, клиенты TMDB и OMDb, импорт Letterboxd |
| `db/` | Схема БД и доступ к данным |
| `notebooks/` | Анализ данных и офлайн-эксперименты |
| `tests/` | Тесты pytest |
| `data/` | Сырые (`raw/`) и обработанные (`processed/`) данные, БД — в git не попадает |

## Запуск окружения

```bash
python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # и вписать ключи
```

## Данные

```bash
python -m data_pipeline.imdb download              # датасеты IMDb в data/raw/ (~1,9 ГБ)
python -m data_pipeline.imdb load --min-votes 5000 # фильмы, люди, жанры в data/movies.db
python -m data_pipeline.movielens download ml-latest-small  # оценки для опытов (1 МБ)
python -m data_pipeline.movielens download ml-32m           # оценки для итоговых замеров (239 МБ)
python -m data_pipeline.movielens link ml-latest-small      # сколько фильмов и оценок связалось с IMDb
python -m data_pipeline.tmdb enrich                         # страна, язык, описание, постер из TMDB (~15 мин)
```

Ответы TMDB кэшируются в таблице `api_cache`: повторный запуск не обращается к API.

Анализ данных: `python notebooks/eda.py` строит графики в `reports/figures/`, выводы — в [reports/eda.md](reports/eda.md).

## Модели

```bash
python -m recsys.features embeddings     # эмбеддинги описаний всех фильмов (один раз, ~5 мин)
python -m recsys.evaluate ml-latest-small # сравнение моделей на MovieLens
pytest                                   # тесты метрик и разбиения
```

Результаты: [reports/metrics_ml-latest-small.md](reports/metrics_ml-latest-small.md).

## Профиль пользователя

| Модуль | Что делает |
|--------|------------|
| `recsys/profile.py` | Профиль — веса признаков по оценкам и реакциям, затухание старых оценок, списки «видел / в планах / скрыто» |
| `recsys/today.py` | Опрос «на сегодня»: длительность и язык — фильтры, настроение, компания, новизна, эпоха — приоритеты |
| `recsys/favorites.py` | Любимые актёры и режиссёры: поиск, автопредложение, непросмотренные фильмы человека |
| `recsys/calibration.py` | Подбор 10 разнообразных фильмов для калибровки вкуса |
| `recsys/recommender.py` | Итоговая выдача: кандидаты по долгосрочному профилю и любимым, перестановка с учётом «сегодня» |

```bash
python -m recsys.tune   # подбор параметров профиля на пользователях MovieLens
```

Результаты: [reports/profile_tuning.md](reports/profile_tuning.md).

## Бот

```bash
python -m bot   # нужны TELEGRAM_BOT_TOKEN и TMDB_API_KEY в .env и загруженные данные (см. выше)
```

Команды: `/start` — знакомство (у знакомого пользователя ничего не сбрасывает), `/recommend` — подборка по вкусу,
`/today` — подборка под настроение, `/lists`, `/favorites`, `/together` — кино на компанию,
`/import` — оценки из Letterboxd, `/restart` — знакомство заново, `/help`. Обычный текст («как „Интерстеллар“,
но повеселее») — поиск фильмов.

| Модуль | Что делает |
|--------|------------|
| `recsys/explain.py` | Объяснение «Почему: тебе нравятся … и …» |
| `recsys/recommender.py` | Выдача, разнообразие (MMR), поиск текстом, подборка для компании |
| `recsys/query.py` | Разбор запроса: фильм-образец, пожелания, фильтры |
| `recsys/sentiment.py` | Тональность отзыва (модель seara/rubert-tiny2-russian-sentiment) |
| `data_pipeline/letterboxd.py` | Импорт выгрузки Letterboxd |
Подборки листаются в одном сообщении; логи — в `logs/bot.log` (без текстов сообщений и ключей).

Интервалы вопроса «посмотрел?» задаются в `.env` в минутах: `FEEDBACK_DELAY_MIN` (по умолчанию сутки после подборки),
`FEEDBACK_ASK_GAP_MIN` (не чаще раза в 6 часов), `FEEDBACK_IGNORE_AFTER_MIN` (через неделю без ответа — «проигнорировано»).

## Развёртывание на сервер

Ubuntu/Debian с sudo. С Mac, из папки проекта (перед первым разом остановить бота на Mac — один токен, один бот):

```bash
SSH_KEY=~/.ssh/ключ ./deploy/deploy.sh user@host --first   # первый раз: код, БД без пользователей, эмбеддинги, .env
SSH_KEY=~/.ssh/ключ ./deploy/deploy.sh user@host           # обновление кода, данные на сервере не трогаются
```

На сервере бот работает службой `movie-bot` (перезапуск при сбое), копия БД — таймер `movie-bot-backup`
каждый день в 04:00, хранятся 14 последних (`data/backups/`). Библиотеки для сервера — `requirements-bot.txt`.

```bash
sudo systemctl status movie-bot                 # состояние
journalctl -u movie-bot -f                      # журнал
python -m db.backup                             # копия вручную (с проверкой)
python -m db.backup --restore data/backups/<файл>.db.gz   # восстановление, бот остановлен
```

## Тестирование

```bash
pytest                                   # юнит-тесты
python -m bot.benchmark                  # время ответа (ТЗ п. 4.3) → reports/timing.md
python -m recsys.cold_start              # холодный старт → reports/cold_start.md
python -m recsys.evaluate ml-32m --sample-users 20000 --eval-users 3000   # большой MovieLens
python -m recsys.online_stats --exclude <свои ID>                         # онлайн-тест → reports/online_test.md
```

Сводка: [reports/testing.md](reports/testing.md).

Данные IMDb и MovieLens используются только в некоммерческих целях и в репозиторий не попадают.
