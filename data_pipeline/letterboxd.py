"""Импорт выгрузки Letterboxd (Settings → Data → Export your data): zip или отдельные csv.

ratings.csv (Name, Year, Rating 0,5–5) → оценки 1–5; watched.csv → список «видел»;
watchlist.csv → «в планах». Фильм ищем в БД по названию и году (допуск ±1 год), не нашли — через
поиск TMDB. Уже поставленные в боте оценки не перезаписываем. Несопоставленные фильмы возвращаем,
чтобы показать пользователю (и учесть в отчёте).

Кодировка csv — UTF-8 (с BOM или без) или cp1251 (так сохраняет Excel). Если переносить нечего, причина
остаётся в ImportResult.reason (или в ExportError), чтобы бот объяснил её человеку и записал в лог.
"""
import asyncio
import io
import math
import re
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from recsys.profile import ensure_user, record_rating, set_list

FILES = {"ratings.csv": "ratings", "watched.csv": "watched", "watchlist.csv": "watchlist"}
# Защита от архивов-бомб и огромных файлов (на сервере 1,6 ГБ памяти, а распаковка целиком в память раньше
# могла занять гигабайты): настоящая выгрузка на 10 тыс. фильмов — около 1 МБ и пара десятков файлов.
MAX_FILE_BYTES = 10 * 1024 * 1024   # распакованный csv; проверяется и по заголовку zip, и при чтении
MAX_ZIP_ENTRIES = 1000              # файлов в архиве
MAX_ROWS = 20_000                   # строк во всех файлах вместе
MAX_TMDB_LOOKUPS = 300              # не найденных в БД ищем через TMDB не больше стольких


@dataclass
class ImportResult:
    rated: int = 0
    watched: int = 0
    planned: int = 0
    skipped_existing: int = 0
    total: int = 0
    unmatched: list[str] = field(default_factory=list)
    seconds: float = 0.0
    rows: dict[str, int] = field(default_factory=dict)  # строк в файлах: ratings / watched / watchlist
    encoding: str = ""                                  # utf-8 или cp1251 (через запятую, если файлов несколько)
    reason: str | None = None                           # почему переносить нечего: no_known_files | no_name_column | empty

    @property
    def matched_share(self) -> float:
        return 1 - len(self.unmatched) / self.total if self.total else 0.0


class ExportError(Exception):
    """Файл не разобрать как csv. reason — короткий код для лога и ответа бота:
    empty | not_csv | encoding | bad_zip | too_big (превышены лимиты размера, файлов или строк)."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def read_csv_bytes(raw: bytes) -> pd.DataFrame:
    """csv из байтов: UTF-8 (BOM снимается сам), иначе cp1251. Кодировка остаётся в df.attrs['encoding']."""
    for encoding, label in (("utf-8-sig", "utf-8"), ("cp1251", "cp1251")):
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        try:
            df = pd.read_csv(io.StringIO(text))
        except pd.errors.EmptyDataError:
            raise ExportError("empty") from None
        except pd.errors.ParserError:
            raise ExportError("not_csv") from None
        df.attrs["encoding"] = label
        return df
    raise ExportError("encoding")


def read_export(path: Path) -> dict[str, pd.DataFrame]:
    """{'ratings': df, 'watched': df, 'watchlist': df} из zip или одного csv."""
    path = Path(path)
    if zipfile.is_zipfile(path):
        out = {}
        with zipfile.ZipFile(path) as z:
            infos = z.infolist()
            if len(infos) > MAX_ZIP_ENTRIES:
                raise ExportError("too_big")
            for info in infos:
                base = info.filename.rsplit("/", 1)[-1].lower()
                # в выгрузке есть и папки (например, deleted/ratings.csv) — берём только верхний уровень
                if base in FILES and "/" not in info.filename.strip("/"):
                    out[FILES[base]] = read_csv_bytes(_read_limited(z, info))
        return _check_rows(out)
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ExportError("too_big")
    df = read_csv_bytes(path.read_bytes())
    kind = FILES.get(path.name.lower()) or ("ratings" if "Rating" in df.columns else "watched")
    return _check_rows({kind: df})


def _read_limited(z: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    """Читает файл из архива, но не больше MAX_FILE_BYTES. Размер в заголовке можно подделать,
    поэтому кроме него ограничено и само чтение."""
    if info.file_size > MAX_FILE_BYTES:
        raise ExportError("too_big")
    try:
        with z.open(info) as f:
            raw = f.read(MAX_FILE_BYTES + 1)
    except (zipfile.BadZipFile, zipfile.LargeZipFile, NotImplementedError, RuntimeError):
        # повреждённый архив, подделанный размер (не сходится контрольная сумма), шифрование
        raise ExportError("bad_zip") from None
    if len(raw) > MAX_FILE_BYTES:
        raise ExportError("too_big")
    return raw


def _check_rows(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    if sum(len(df) for df in data.values()) > MAX_ROWS:
        raise ExportError("too_big")
    return data


def normalize(title: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(title).lower())).strip()


def letterboxd_rating(r: float) -> int:
    """0,5–5 со звёздочками-половинками → 1–5: половинки округляем вверх (2,5 → 3), минимум 1."""
    return int(min(5, max(1, math.floor(r + 0.5))))


class Matcher:
    def __init__(self, movies: pd.DataFrame):
        if "kind" in movies.columns:  # Letterboxd — про фильмы: сериал с тем же названием не подставляем
            movies = movies[movies["kind"].fillna("movie") == "movie"]
        m = movies[["tconst", "title", "original_title", "year", "votes", "tmdb_id"]].copy()
        rows = []
        for col in ["title", "original_title"]:
            part = m.dropna(subset=[col]).assign(key=lambda d: d[col].map(normalize))
            rows.append(part[["key", "tconst", "year", "votes"]])
        self.index = pd.concat(rows).drop_duplicates(["key", "tconst"]).groupby("key")
        self.by_tmdb = dict(zip(m.tmdb_id.dropna().astype(int), m.tconst))

    def find(self, name: str, year) -> str | None:
        key = normalize(name)
        if key not in self.index.groups:
            return None
        cands = self.index.get_group(key)
        if pd.notna(year):
            exact = cands[cands.year == int(year)]
            near = cands[(cands.year - int(year)).abs() <= 1]
            cands = exact if len(exact) else near
        return None if cands.empty else cands.sort_values("votes", ascending=False).tconst.iloc[0]


async def tmdb_find(tmdb, matcher: Matcher, name: str, year) -> str | None:
    params = {"query": name}
    if pd.notna(year):
        params["year"] = int(year)
    data = await tmdb.get("/search/movie", **params)
    for r in data.get("results", [])[:3]:
        if r["id"] in matcher.by_tmdb:
            return matcher.by_tmdb[r["id"]]
    return None


async def import_export(conn, user_id: int, path: Path, movies: pd.DataFrame, tmdb=None) -> ImportResult:
    t0 = time.time()
    # распаковка и разбор csv — в отдельном потоке: иначе на это время бот не отвечает никому
    data = await asyncio.to_thread(read_export, path)
    matcher = Matcher(movies)
    res = ImportResult()
    items = []  # (вид, название, год, оценка)
    for kind in ["ratings", "watched", "watchlist"]:
        df = data.get(kind)
        if df is None or "Name" not in df.columns:
            continue
        res.rows[kind] = len(df)
        for row in df.itertuples(index=False):
            items.append((kind, row.Name, getattr(row, "Year", None), getattr(row, "Rating", None)))
    res.total = len({(n, y) for _, n, y, _ in items})
    res.encoding = ",".join(sorted({df.attrs.get("encoding", "") for df in data.values()} - {""}))
    if not items:
        res.reason = "no_known_files" if not data else "no_name_column" if not res.rows else "empty"
        res.seconds = time.time() - t0
        return res

    found = {(n, y): matcher.find(n, y) for _, n, y, _ in items}
    missing = [k for k, v in found.items() if v is None]
    if tmdb is not None and missing:
        lookup = missing[:MAX_TMDB_LOOKUPS]  # остальные останутся среди ненайденных
        results = await asyncio.gather(*(tmdb_find(tmdb, matcher, n, y) for n, y in lookup))
        found.update(dict(zip(lookup, results)))

    ensure_user(conn, user_id)
    rated_before = {r[0] for r in conn.execute("SELECT tconst FROM ratings WHERE user_id = ?", (user_id,))}
    for kind, name, year, rating in items:
        tconst = found[(name, year)]
        if tconst is None:
            continue
        if kind == "ratings" and pd.notna(rating):
            if tconst in rated_before:
                res.skipped_existing += 1
                continue
            record_rating(conn, user_id, tconst, letterboxd_rating(float(rating)), "letterboxd")
            res.rated += 1
        elif kind == "watched":
            set_list(conn, user_id, tconst, "seen")
            res.watched += 1
        elif kind == "watchlist":
            set_list(conn, user_id, tconst, "planned")
            res.planned += 1
    res.unmatched = sorted({f"{n} ({int(y)})" if pd.notna(y) else str(n) for (n, y), t in found.items() if t is None})
    res.seconds = time.time() - t0
    return res
