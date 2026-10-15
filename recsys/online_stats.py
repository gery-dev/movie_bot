"""Статистика онлайн-теста (этап 6): как живые пользователи принимают рекомендации.

    python -m recsys.online_stats --exclude 914757854   # исключить свои тестовые аккаунты

В отчёт попадают только сводные числа — без ID, имён и текстов отзывов (репозиторий публичный).
Результат — reports/online_test.md.
"""
import argparse

import pandas as pd

from db.database import ROOT, connect

MODES = {"recommend": "/recommend (по вкусу)", "today": "/today (под настроение)", "query": "поиск текстом",
         "new": "/new (новинки)", "cinema": "/cinema (прокат)"}
ANSWERS = {"rated": "оценил", "not_interested": "не интересно", "later": "позже", "hidden": "скрыл",
           "seen": "уже видел", "ignored": "без ответа (неделя)", None: "ещё без ответа"}


def main(exclude: set[int]) -> None:
    conn = connect()
    q = lambda sql: pd.read_sql(sql, conn)  # noqa: E731
    users = q("SELECT user_id, onboarding_done, calibration_done FROM users")
    users = users[~users.user_id.isin(exclude)]
    ids = set(users.user_id)
    rec = q("SELECT user_id, tconst, mode, answer FROM recommendations")
    rec = rec[rec.user_id.isin(ids)].drop_duplicates(["user_id", "tconst", "mode"])
    ratings = q("SELECT user_id, tconst, rating, source FROM ratings")
    ratings = ratings[ratings.user_id.isin(ids)]
    reviews = q("SELECT user_id, sentiment FROM reactions WHERE reaction = 'review' AND sentiment IS NOT NULL")
    reviews = reviews[reviews.user_id.isin(ids)]
    survey = q("SELECT user_id, wave, score FROM survey")
    survey = survey[survey.user_id.isin(ids)]

    lines = ["# Онлайн-тест\n",
             f"Пользователей: {len(ids)}; прошли знакомство: {int(users.onboarding_done.sum())}; "
             f"калибровку: {int(users.calibration_done.sum())}; импортировали Letterboxd: "
             f"{ratings[ratings.source == 'letterboxd'].user_id.nunique()}. "
             f"Свои тестовые аккаунты исключены: {len(exclude)}.\n"]

    lines.append("## Ответы на показанные фильмы\n")
    lines.append("| Режим | показано | " + " | ".join(ANSWERS.values()) + " | средняя оценка рекомендованных |")
    lines.append("|---|---|" + "---|" * len(ANSWERS) + "---|")
    rated = ratings[ratings.source == "recommendation"]
    for mode, label in MODES.items():
        part = rec[rec["mode"] == mode]
        if part.empty:
            continue
        shares = [f"{(part.answer.isna() if a is None else part.answer == a).mean():.0%}" for a in ANSWERS]
        mean = part.merge(rated, on=["user_id", "tconst"]).rating.mean()
        lines.append(f"| {label} | {len(part)} | " + " | ".join(shares) + f" | {mean:.2f} |" if pd.notna(mean)
                     else f"| {label} | {len(part)} | " + " | ".join(shares) + " | — |")
    lines.append("\nСредняя оценка — по фильмам, которые пользователь оценил прямо из подборки (1–5).\n")

    lines.append("## Опрос «насколько подборки подходят» (1–5)\n")
    if survey.empty:
        lines.append("Ответов пока нет.\n")
    else:
        for wave, g in survey.groupby("wave"):
            dist = g.score.value_counts().reindex(range(1, 6), fill_value=0)
            lines.append(f"- после {wave} оценок: ответов {len(g)}, среднее {g.score.mean():.2f}, "
                         f"распределение 1–5: {' / '.join(map(str, dist))}")
        lines.append("")
    if not reviews.empty:
        lines.append(f"## Отзывы\n\nТекстовых отзывов: {len(reviews)}, средняя тональность {reviews.sentiment.mean():+.2f} "
                     f"(−1 — негатив, +1 — позитив), позитивных (> 0,3): {(reviews.sentiment > 0.3).mean():.0%}.\n")
    text = "\n".join(lines)
    (ROOT / "reports" / "online_test.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude", type=int, nargs="*", default=[], help="ID своих тестовых аккаунтов")
    main(set(parser.parse_args().exclude))
