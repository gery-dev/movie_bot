# %% [markdown]
# # Анализ данных о фильмах (этап 1)
# Запуск: `python notebooks/eda.py` — графики сохраняются в reports/figures/.
# В VS Code ячейки `# %%` можно выполнять по одной (нужен пакет ipykernel).

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_pipeline.movielens import load_ratings  # noqa: E402
from bot.texts import GENRES_RU  # noqa: E402
from db.database import ROOT, connect  # noqa: E402

FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({
    "figure.figsize": (8, 4.5), "figure.dpi": 150, "font.size": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "xtick.color": INK_2, "ytick.color": INK_2, "xtick.major.size": 0, "ytick.major.size": 0,
    "legend.frameon": False,
})


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", facecolor="white")
    plt.close(fig)
    print(f"  сохранён {name}.png")


def thousands(x, _=None) -> str:
    return f"{x:,.0f}".replace(",", " ")


def median_line(ax, value: float, label: str) -> None:
    """Пунктир медианы с подписью над столбцами (ось y растягивается, чтобы подпись не налезала)."""
    top = ax.get_ylim()[1]
    ax.set_ylim(0, top * 1.12)
    ax.axvline(value, color=INK_2, linewidth=1, linestyle="--")
    ax.annotate(label, (value, top * 1.06), xytext=(5, 0), textcoords="offset points", va="center", color=INK)


# %%
conn = connect()
movies = pd.read_sql("SELECT tconst, year, runtime, rating, votes, country FROM movies", conn)
genres = pd.read_sql("SELECT genre FROM movie_genres", conn)
movies["country1"] = movies["country"].str.split(",").str[0]
print(f"Фильмов: {len(movies)}")

# %% Фильмы по десятилетиям
decades = (movies["year"] // 10 * 10).value_counts().sort_index()
fig, ax = plt.subplots()
ax.bar(decades.index.astype(str) + "-е", decades.values, color=BLUE, width=0.75)
ax.grid(axis="x", visible=False)
ax.yaxis.set_major_formatter(thousands)
ax.set_title("Фильмы по десятилетиям выхода")
ax.set_ylabel("Число фильмов")
for x, v in zip(range(len(decades)), decades.values):
    if v == decades.max():
        ax.annotate(thousands(v), (x, v), ha="center", va="bottom", color=INK, xytext=(0, 3), textcoords="offset points")
ax.tick_params(axis="x", labelrotation=45)
save(fig, "01_decades")

# %% Жанры
genre_counts = genres["genre"].map(GENRES_RU).fillna(genres["genre"]).value_counts().sort_values()
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(genre_counts.index, genre_counts.values, color=BLUE, height=0.7)
ax.grid(axis="y", visible=False)
ax.xaxis.set_major_formatter(thousands)
ax.set_title("Жанры (у фильма может быть до трёх жанров)")
ax.set_xlabel("Число фильмов")
for y, v in enumerate(genre_counts.values):
    ax.annotate(thousands(v), (v, y), va="center", color=INK_2, xytext=(4, 0), textcoords="offset points", fontsize=8)
save(fig, "02_genres")

# %% Рейтинг IMDb
fig, ax = plt.subplots()
ax.hist(movies["rating"], bins=np.arange(1, 10.01, 0.2), color=BLUE, edgecolor="white", linewidth=0.5)
median = movies["rating"].median()
median_line(ax, median, f"медиана {median:.1f}".replace(".", ","))
ax.grid(axis="x", visible=False)
ax.set_title("Распределение рейтинга IMDb")
ax.set_xlabel("Рейтинг")
ax.set_ylabel("Число фильмов")
save(fig, "03_rating")

# %% Число голосов (логарифмическая шкала)
fig, ax = plt.subplots()
bins = np.logspace(np.log10(5000), np.log10(movies["votes"].max()), 40)
ax.hist(movies["votes"], bins=bins, color=BLUE, edgecolor="white", linewidth=0.5)
ax.set_xscale("log")
ax.xaxis.set_major_formatter(thousands)
ax.grid(axis="x", visible=False)
median = movies["votes"].median()
median_line(ax, median, f"медиана {thousands(median)}")
ax.set_title("Число голосов на IMDb (шкала логарифмическая)")
ax.set_xlabel("Голосов у фильма")
ax.set_ylabel("Число фильмов")
save(fig, "04_votes")

# %% Страны: вся база против топ-100 по рейтингу
top100 = movies.nlargest(100, "rating")
shown = ["US", "IN", "GB", "FR", "JP", "TR"]
names = {"US": "США", "IN": "Индия", "GB": "Велико-\nбритания", "FR": "Франция", "JP": "Япония", "TR": "Турция"}

def shares(df):
    s = df["country1"].where(df["country1"].isin(shown), "other").value_counts(normalize=True) * 100
    return s.reindex(shown + ["other"]).fillna(0)

all_s, top_s = shares(movies), shares(top100)
labels = [names[c] for c in shown] + ["Другие"]
x = np.arange(len(labels))
fig, ax = plt.subplots()
ax.bar(x - 0.2, all_s.values, width=0.38, color=BLUE, label="Вся база (19 171 фильм)")
ax.bar(x + 0.2, top_s.values, width=0.38, color=ORANGE, label="100 фильмов с высшим рейтингом")
ax.set_xticks(x, labels)
ax.grid(axis="x", visible=False)
ax.set_ylabel("Доля фильмов, %")
ax.set_title("Страна производства: вся база и топ-100 по рейтингу IMDb")
ax.legend(loc="upper right")
for i in (1, 5):  # Индия и Турция: доля в топе намного больше доли в базе
    ax.annotate(f"{top_s.iloc[i]:.0f}%", (i + 0.2, top_s.iloc[i]), ha="center", va="bottom", color=INK, xytext=(0, 3), textcoords="offset points")
save(fig, "05_countries_top100")

# %% MovieLens: длинный хвост
ratings = load_ratings("ml-32m")
per_movie = ratings.groupby("tconst").size().sort_values(ascending=False).values
films_share = np.arange(1, len(per_movie) + 1) / len(per_movie) * 100
ratings_share = per_movie.cumsum() / per_movie.sum() * 100
fig, ax = plt.subplots()
ax.plot(films_share, ratings_share, color=BLUE, linewidth=2)
i10 = int(len(per_movie) * 0.1) - 1
ax.plot(films_share[i10], ratings_share[i10], "o", color=BLUE, markersize=8, markeredgecolor="white", markeredgewidth=2)
ax.annotate(f"10% самых популярных фильмов\nсобирают {ratings_share[i10]:.0f}% оценок", (films_share[i10], ratings_share[i10]),
            xytext=(15, -35), textcoords="offset points", color=INK)
ax.set_xlim(0, 100)
ax.set_ylim(0, 101)
ax.set_title("MovieLens 32M: на какие фильмы приходятся оценки")
ax.set_xlabel("Фильмы, отсортированные по числу оценок, % от всех")
ax.set_ylabel("Накопленная доля оценок, %")
save(fig, "06_movielens_long_tail")

# %% Сводка для отчёта
print("\nСводка:")
print(f"  годы {movies.year.min()}–{movies.year.max()}, медиана {movies.year.median():.0f}; с 2000 года {(movies.year >= 2000).mean():.0%}")
print(f"  длительность: медиана {movies.runtime.median():.0f} мин, 80% фильмов от {movies.runtime.quantile(.1):.0f} до {movies.runtime.quantile(.9):.0f} мин")
print(f"  рейтинг: медиана {movies.rating.median():.1f}; голоса: медиана {thousands(movies.votes.median())}")
print(f"  США {(movies.country1 == 'US').mean():.0%} базы; Индия {(movies.country1 == 'IN').mean():.0%} базы, но {top_s['IN']:.0f}% топ-100")
print(f"  MovieLens: {len(ratings)} оценок, {len(per_movie)} фильмов, медиана {np.median(per_movie):.0f} оценок на фильм")
