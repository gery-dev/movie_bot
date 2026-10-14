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

Данные IMDb и MovieLens используются только в некоммерческих целях и в репозиторий не попадают.
