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
