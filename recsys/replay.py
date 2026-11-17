"""Переигрывание журнала онлайн-теста: как вариант рекомендателя ранжирует фильмы, на которые живые люди
ответили хорошо, относительно фильмов, на которые ответили плохо.

    SSH_KEY=~/.ssh/<ключ> python -m recsys.replay export <user@сервер>   # выгрузка → data/online/export.json.gz
    python -m recsys.replay                                              # метрики нынешних коэффициентов

Выгрузка обезличена: Telegram ID заменены порядковыми номерами, тексты отзывов, имена и параметры команд
не берутся. На сервере только читается БД (sqlite mode=ro), файлы там не создаются.

Хорошо — оценка 4–5 или «позже»; плохо — «не интересно» (кроме причины «уже видел»), «скрыть», оценка 1–2.
Берутся показы подборок фильмов (/recommend, /today, /new); поиск текстом, прокат и сериалы — другие выдачи.
Для каждого показа восстанавливается состояние человека на момент показа: только события раньше shown_at,
время сдвинуто так, что «сейчас» для рекомендателя — момент показа (затухание, пауза 30 дней). Настоящий
рекомендатель ранжирует каталог как подборку по вкусу (/recommend, без MMR; настроение /today не учитывается)
и смотрится место показанного фильма; вне CANDIDATES лучших — одно общее последнее место.
Метрики по людям, разброс — бутстреп по людям:
- AUC — доля пар «хороший, плохой» одного человека, где хороший стоит выше (среднее по людям, у кого есть оба);
- топ-10 — доля хороших среди размеченных фильмов, попавших в десятку; хороших и плохих, попавших в десятку.
Холодный старт — показы при 0–2 оценках на момент показа, остальные — от 3 оценок.
"""
import gzip
import json
import os
import sqlite3
import subprocess
import sys
import time
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from unittest import mock

import numpy as np
import pandas as pd

from db.database import ROOT, connect
from recsys import recommender as R
from recsys.models import ALS, Catalog
from recsys.train_als import ALS_FILE

EXPORT_FILE = ROOT / "data" / "online" / "export.json.gz"
TOP = 10
COLD_MAX = 2          # холодный старт — не больше стольких оценок на момент показа
MODES = ("recommend", "today", "new")
USER_TABLES = ("users", "ratings", "reactions", "recommendations", "user_lists", "favorites", "user_profile")
BOOT = 2000

# Выполняется на сервере (только стандартная библиотека): читает БД и печатает JSON.
EXPORT_SCRIPT = r'''
import json, sqlite3, sys
c = sqlite3.connect("file:/opt/movie_bot/data/movies.db?mode=ro", uri=True)
users = [r[0] for r in c.execute("SELECT user_id FROM users ORDER BY created_at, rowid")]
active = dict(c.execute("SELECT user_id, count(*) FROM events WHERE user_id IS NOT NULL GROUP BY user_id"))
ids = users + sorted(set(active) - set(users))
num = {u: i + 1 for i, u in enumerate(ids)}
TABLES = {
    "users": "user_id, created_at, onboarding_done, calibration_done, genres",
    "ratings": "user_id, tconst, rating, source, created_at",
    "reactions": "id, user_id, tconst, reaction, sentiment, created_at",
    "recommendations": "id, user_id, tconst, shown_at, answer, mode, position, n_ratings, model_version, mood, reason",
    "user_lists": "user_id, tconst, list, created_at",
    "favorites": "user_id, nconst, added_by, created_at",
    "survey": "user_id, wave, score, created_at",
    "survey_offers": "user_id, wave, offered_at",
    "events": "user_id, ts, kind, name, state, duration_ms, error, matched",
}
out = {"owner": num[max(active, key=active.get)], "exported_at": c.execute("SELECT datetime('now')").fetchone()[0]}
TABLES["movies"] = "NULL AS user_id, tconst, title, year, votes, rating, kind"  # каталог сервера: сверить с локальным
for table, cols in TABLES.items():
    cur = c.execute(f"SELECT {cols} FROM {table}")
    names = [d[0] for d in cur.description]
    out[table] = [dict(zip(names, r), user_id=num.get(r[names.index("user_id")])) for r in cur]
json.dump(out, sys.stdout, ensure_ascii=False)
'''


def export(host: str) -> None:
    key = os.path.expanduser(os.getenv("SSH_KEY", "~/.ssh/id_rsa"))
    res = subprocess.run(["ssh", "-i", key, "-o", "ConnectTimeout=20", "-o", "BatchMode=yes", host, "python3", "-"],
                         input=EXPORT_SCRIPT.encode(), capture_output=True, check=True)
    data = json.loads(res.stdout)
    EXPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(EXPORT_FILE, "wt", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"{EXPORT_FILE}: " + ", ".join(f"{t} {len(v)}" for t, v in data.items() if isinstance(v, list)))


def load_export() -> dict:
    with gzip.open(EXPORT_FILE, "rt", encoding="utf-8") as f:
        return json.load(f)


def _dt(text: str) -> datetime:
    return datetime.fromisoformat(text[:19])


# --- показы с ответом ---------------------------------------------------------

def label(answer, rating, reason) -> str | None:
    if answer == "rated":
        return "good" if rating >= 4 else "bad" if rating <= 2 else None
    if answer == "later":
        return "good"
    if answer == "hidden" or (answer == "not_interested" and reason != "seen"):
        return "bad"
    return None


def labeled_shows(data: dict, owner: bool | None = False, ignored: bool = False) -> pd.DataFrame:
    """Показы подборок фильмов с хорошим или плохим ответом. owner: False — без владельца, True — только
    владелец, None — все. ignored=True — добавить пролистанные без ответа (метка «ignored»)."""
    rec = pd.DataFrame(data["recommendations"])
    rat = pd.DataFrame(data["ratings"])
    rec = rec[rec["mode"].isin(MODES)].merge(rat[["user_id", "tconst", "rating"]], how="left")
    rec["label"] = [label(a, r, s) for a, r, s in zip(rec.answer, rec.rating, rec.reason)]
    if ignored:
        rec.loc[rec.answer.isin(["ignored"]) | rec.answer.isna(), "label"] = "ignored"
    rec = rec[rec.label.notna()]
    if owner is not None:
        rec = rec[(rec.user_id == data["owner"]) == owner]
    # ответ записывается во все неотвеченные показы фильма — оставляем последний показ
    rec = rec.sort_values("shown_at").drop_duplicates(["user_id", "tconst", "label"], keep="last")
    times = rat.assign(t=rat.created_at.map(_dt)).groupby("user_id").t.apply(lambda s: np.sort(s.to_numpy()))
    rec["t"] = rec.shown_at.map(_dt)
    rec["n_before"] = [int(np.searchsorted(times.get(u, np.array([], "datetime64[us]")), np.datetime64(t)))
                       for u, t in zip(rec.user_id, rec.t)]
    return rec.reset_index(drop=True)


# --- переигрывание -------------------------------------------------------------

class Replay:
    """Каталог — копия локальной БД в памяти без пользовательских таблиц; ALS — векторы бота.
    server_catalog=True — добавить фильмы, которые есть только в БД сервера (как каталог на сервере)."""

    def __init__(self, data: dict, server_catalog: bool = False):
        src = connect()
        self.conn = sqlite3.connect(":memory:")
        src.backup(self.conn)
        src.close()
        self.conn.row_factory = sqlite3.Row
        with self.conn:
            for t in USER_TABLES:
                self.conn.execute(f"DELETE FROM {t}")
            if server_catalog:
                self.conn.executemany(
                    """INSERT OR IGNORE INTO movies (tconst, title, year, votes, rating, kind)
                       VALUES (:tconst, :title, :year, :votes, :rating, :kind)""", data["movies"])
        self.ids = [r[0] for r in self.conn.execute("SELECT tconst FROM movies WHERE kind = 'movie' ORDER BY tconst")]
        self.als = ALS.load(ALS_FILE, Catalog(self.ids))
        self.data = data
        self.frames = {t: pd.DataFrame(data[t]) for t in ("users", "ratings", "reactions", "recommendations",
                                                          "user_lists", "favorites")}
        self.now = datetime.now(timezone.utc).replace(tzinfo=None)
        self.rec = self.recommender()

    def recommender(self) -> R.Recommender:
        """Новый рекомендатель — после изменения констант, которые он считает при создании."""
        rec = R.Recommender(self.conn, als=self.als)
        rec._why = lambda *a, **k: ""
        return rec

    def _snapshot(self, uid: int, user: int, t: datetime) -> None:
        """Записать события человека раньше t под uid; время сдвинуто на (сейчас − t)."""
        shift = self.now - t

        def before(df, col):
            if df.empty:
                return df
            df = df[(df.user_id == user) & (df[col].map(_dt) < t)].copy()
            df[col] = (df[col].map(_dt) + shift).dt.strftime("%Y-%m-%d %H:%M:%S") if len(df) else df[col]
            return df.assign(user_id=uid)

        f = self.frames
        u = f["users"][f["users"].user_id == user]
        genres = u.genres.iloc[0] if len(u) else None
        rat, rea = before(f["ratings"], "created_at"), before(f["reactions"], "created_at")
        lists, fav = before(f["user_lists"], "created_at"), before(f["favorites"], "created_at")
        recs = before(f["recommendations"], "shown_at")
        # ответ на прошлый показ известен, только если он дан раньше t
        known = set(rat.tconst) | set(rea.tconst) | set(lists.tconst)
        if len(recs):
            recs.loc[~recs.tconst.isin(known), ["answer", "reason"]] = None
        c = self.conn
        c.execute("INSERT INTO users (user_id, genres) VALUES (?, ?)", (uid, genres))
        c.executemany("INSERT INTO ratings (user_id, tconst, rating, source, created_at) VALUES (?, ?, ?, ?, ?)",
                      rat[["user_id", "tconst", "rating", "source", "created_at"]].itertuples(index=False))
        c.executemany("INSERT INTO reactions (user_id, tconst, reaction, sentiment, created_at) VALUES (?, ?, ?, ?, ?)",
                      rea[["user_id", "tconst", "reaction", "sentiment", "created_at"]].itertuples(index=False))
        c.executemany("INSERT OR IGNORE INTO user_lists (user_id, tconst, list, created_at) VALUES (?, ?, ?, ?)",
                      lists[["user_id", "tconst", "list", "created_at"]].itertuples(index=False))
        c.executemany("INSERT OR IGNORE INTO favorites (user_id, nconst, added_by, created_at) VALUES (?, ?, ?, ?)",
                      fav[["user_id", "nconst", "added_by", "created_at"]].itertuples(index=False))
        c.executemany("INSERT INTO recommendations (user_id, tconst, shown_at, answer, mode, reason) VALUES (?, ?, ?, ?, ?, ?)",
                      recs[["user_id", "tconst", "shown_at", "answer", "mode", "reason"]].itertuples(index=False))

    def _clear(self, uid: int) -> None:
        for t in USER_TABLES:
            self.conn.execute(f"DELETE FROM {t} WHERE user_id = ?", (uid,))

    def ranks(self, shows: pd.DataFrame, rec: R.Recommender | None = None) -> np.ndarray:
        """Место каждого показанного фильма (1 — первый) в подборке на момент показа; CANDIDATES + 1 — вне
        кандидатов; NaN — фильма нет в пуле подборок (каталог, порог голосов)."""
        rec = rec or self.rec
        out = np.full(len(shows), np.nan)
        orig = R.excluded_films
        for i, s in enumerate(shows.itertuples()):
            if s.tconst not in rec.catalog.index:
                continue
            uid = 10 ** 9 + i
            self._snapshot(uid, s.user_id, s.t)
            # показанный фильм ранжируется, даже если к этому моменту он уже исключён (повторный показ)
            with mock.patch.object(R, "excluded_films", lambda conn, u: orig(conn, u) - {s.tconst}):
                pos = rec.catalog.index[s.tconst]
                ok = (rec.movies.votes.iloc[pos] or 0) >= R.MIN_VOTES
                if ok:
                    top = rec.recommend(self.conn, uid, k=R.CANDIDATES, diverse=False).tconst.tolist()
                    out[i] = top.index(s.tconst) + 1 if s.tconst in top else R.CANDIDATES + 1
            self._clear(uid)
        return out


def patched(changes: dict) -> ExitStack:
    """Временные значения констант: {(модуль, имя): значение}; словари (SIGNALS) — patch.dict."""
    stack = ExitStack()
    for (module, name), value in changes.items():
        target = getattr(module, name)
        if isinstance(target, dict):
            stack.enter_context(mock.patch.dict(target, value))
        else:
            stack.enter_context(mock.patch.object(module, name, value))
    return stack


# --- метрики --------------------------------------------------------------------

def by_user(shows: pd.DataFrame, ranks: np.ndarray, mask=None) -> list[tuple[np.ndarray, np.ndarray]]:
    """По каждому человеку — места хороших и плохих показов (без фильмов вне пула)."""
    df = shows.assign(rank=ranks)
    if mask is not None:
        df = df[np.asarray(mask)]
    df = df[df["rank"].notna()]
    return [(g[g.label == "good"]["rank"].to_numpy(), g[g.label == "bad"]["rank"].to_numpy())
            for _, g in df.groupby("user_id")]


def _auc(good: np.ndarray, bad: np.ndarray) -> float:
    if not len(good) or not len(bad):
        return np.nan
    d = good[:, None] - bad[None, :]
    return float(((d < 0) + 0.5 * (d == 0)).mean())


def stats(users: list, top: int = TOP) -> dict:
    good = np.concatenate([g for g, _ in users] or [[]])
    bad = np.concatenate([b for _, b in users] or [[]])
    per = [a for g, b in users if np.isfinite(a := _auc(g, b))]
    g10, b10 = (good <= top).sum(), (bad <= top).sum()
    return {
        "AUC": np.mean(per) if per else np.nan,                    # среднее по людям, у кого есть оба
        "AUC общий": _auc(good, bad),                               # все пары, и разных людей
        "хороших среди десятки": g10 / (g10 + b10) if g10 + b10 else np.nan,
        "хороших в десятке": g10 / len(good) if len(good) else np.nan,
        "плохих в десятке": b10 / len(bad) if len(bad) else np.nan,
    }


def _resamples(n_users: int, n: int = BOOT, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n_users, (n, n_users))


def summary(users: list) -> dict:
    """Метрики и 95% интервал бутстрепа по людям."""
    est = stats(users)
    boots = [stats([users[i] for i in idx]) for idx in _resamples(len(users))] if len(users) > 1 else []
    out = {"людей": len(users), "с обоими": sum(bool(len(g) and len(b)) for g, b in users),
           "хороших": sum(len(g) for g, _ in users), "плохих": sum(len(b) for _, b in users)}
    for k, v in est.items():
        vals = [b[k] for b in boots]
        lo, hi = np.nanpercentile(vals, [2.5, 97.5]) if boots else (np.nan, np.nan)
        out[k] = (v, lo, hi)
    return out


def compare(a: list, b: list) -> dict:
    """Разница «b − a» на тех же людях и тех же выборках бутстрепа; для AUC — у скольких лучше и хуже."""
    est_a, est_b = stats(a), stats(b)
    diffs = []
    for idx in _resamples(len(a)):
        sa, sb = stats([a[i] for i in idx]), stats([b[i] for i in idx])
        diffs.append({k: sb[k] - sa[k] for k in sa})
    out = {}
    for k in est_a:
        lo, hi = np.nanpercentile([d[k] for d in diffs], [2.5, 97.5])
        out[k] = (est_b[k] - est_a[k], lo, hi)
    d = [_auc(*ub) - _auc(*ua) for ua, ub in zip(a, b)]
    out["лучше / хуже"] = f"{sum(x > 1e-9 for x in d)} / {sum(x < -1e-9 for x in d)}"
    return out


def fmt(x) -> str:
    if isinstance(x, tuple):
        est, lo, hi = x
        return f"{est:.3f} [{lo:.3f}; {hi:.3f}]" if np.isfinite(lo) else f"{est:.3f}"
    return str(x)


SEGMENTS = {"все": lambda s: np.ones(len(s), bool), f"0–{COLD_MAX} оценки": lambda s: s.n_before <= COLD_MAX,
            f"от {COLD_MAX + 1} оценок": lambda s: s.n_before > COLD_MAX}


def table(rows: dict) -> str:
    """rows: {название строки: словарь метрик} → таблица Markdown."""
    cols = list(next(iter(rows.values())))
    lines = ["| | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    lines += [f"| {name} | " + " | ".join(fmt(r[c]) for c in cols) + " |" for name, r in rows.items()]
    return "\n".join(lines)


def main() -> None:
    t0 = time.time()
    data = load_export()
    for server, title in [(True, "каталог сервера (12 фильмов без рейтинга)"), (False, "локальный каталог")]:
        rp = Replay(data, server_catalog=server)
        rows = {}
        for owner, who in [(False, "без владельца"), (None, "с владельцем")]:
            shows = labeled_shows(data, owner)
            ranks = rp.ranks(shows)
            rows |= {f"{who}, {seg}": summary(by_user(shows, ranks, fn(shows))) for seg, fn in SEGMENTS.items()}
        print(f"\n{title}\n{table(rows)}")
    print(f"\n{time.time() - t0:.0f} с")


if __name__ == "__main__":
    if sys.argv[1:2] == ["export"]:
        export(sys.argv[2])
    else:
        main()
