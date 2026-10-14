"""Базовые модели рекомендаций.

У всех моделей один интерфейс: fit(train) обучает на оценках, scores(user_id) возвращает
оценку пригодности для каждого фильма каталога (чем больше, тем выше в выдаче).
Каталог — список tconst, общий для всех моделей.
"""
import numpy as np
import pandas as pd
from scipy import sparse

from db.database import connect
from recsys.data import RELEVANT
from recsys.features import load_embeddings, meta_features


class Catalog:
    def __init__(self, tconsts: list[str]):
        self.ids = list(tconsts)
        self.index = {t: i for i, t in enumerate(self.ids)}

    def __len__(self) -> int:
        return len(self.ids)

    def positions(self, tconsts) -> np.ndarray:
        return np.array([self.index[t] for t in tconsts], dtype=int)


class Popularity:
    """Одна и та же выдача для всех. kind:
    imdb — сырой рейтинг IMDb;
    imdb_weighted — рейтинг IMDb с поправкой на число голосов: (v·R + m·C) / (v + m);
    ml — сколько раз фильм понравился пользователям MovieLens в обучающей выборке.
    """

    def __init__(self, catalog: Catalog, kind: str = "imdb_weighted", min_votes: int = 50_000):
        self.catalog, self.kind, self.min_votes = catalog, kind, min_votes

    def fit(self, train: pd.DataFrame) -> "Popularity":
        if self.kind == "ml":
            liked = train[train.rating >= RELEVANT].tconst.value_counts()
            self._scores = liked.reindex(self.catalog.ids).fillna(0).to_numpy(float)
            return self
        conn = connect()
        m = pd.read_sql("SELECT tconst, rating, votes FROM movies", conn).set_index("tconst").reindex(self.catalog.ids)
        conn.close()
        if self.kind == "imdb":
            self._scores = m.rating.to_numpy(float)
        else:
            v, R, C = m.votes.to_numpy(float), m.rating.to_numpy(float), m.rating.mean()
            self._scores = (v * R + self.min_votes * C) / (v + self.min_votes)
        return self

    def scores(self, user_id) -> np.ndarray:
        return self._scores


class Content:
    """Профиль пользователя — сумма векторов оценённых фильмов с весом (оценка − средняя оценка
    пользователя). Пригодность фильма — близость его вектора к профилю.
    emb_weight — доля эмбеддингов описаний, остальное — мета-признаки (жанры, люди, страна…).
    """

    def __init__(self, catalog: Catalog, emb_weight: float = 0.5):
        self.catalog, self.emb_weight = catalog, emb_weight
        self.meta = meta_features(catalog.ids)
        self.emb = load_embeddings(catalog.ids)

    def fit(self, train: pd.DataFrame) -> "Content":
        r = train[["userId", "tconst", "rating"]].copy()
        r["w"] = r.rating - r.groupby("userId").rating.transform("mean")
        r.loc[r.groupby("userId").w.transform(lambda w: (w == 0).all()), "w"] = 1.0
        users = r.userId.unique()
        self.user_index = {u: i for i, u in enumerate(users)}
        W = sparse.csr_matrix(
            (r.w.to_numpy(float), (r.userId.map(self.user_index), self.catalog.positions(r.tconst))),
            shape=(len(users), len(self.catalog)),
        )
        self.profile_meta = (W @ self.meta).tocsr()
        self.profile_emb = W @ self.emb
        return self

    def scores(self, user_id) -> np.ndarray:
        u = self.user_index.get(user_id)
        if u is None:
            return np.zeros(len(self.catalog))
        s_meta = (self.meta @ self.profile_meta[u].T).toarray().ravel()
        s_emb = self.emb @ self.profile_emb[u]
        # приводим к одному масштабу, чтобы вес emb_weight значил одно и то же у всех пользователей
        return (1 - self.emb_weight) * _unit(s_meta) + self.emb_weight * _unit(s_emb)


class SVD:
    """Матричное разложение из surprise: оценка = μ + b_u + b_i + p_u·q_i."""

    def __init__(self, catalog: Catalog, n_factors: int = 100, n_epochs: int = 20, lr_all: float = 0.005,
                 reg_all: float = 0.02):
        self.catalog = catalog
        self.params = dict(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all)

    def fit(self, train: pd.DataFrame) -> "SVD":
        from surprise import SVD as SurpriseSVD, Dataset, Reader

        data = Dataset.load_from_df(train[["userId", "tconst", "rating"]], Reader(rating_scale=(0.5, 5)))
        self.trainset = data.build_full_trainset()
        self.algo = SurpriseSVD(random_state=42, **self.params).fit(self.trainset)
        known = np.array([self.trainset.knows_item(_inner(self.trainset.to_inner_iid, t)) for t in self.catalog.ids])
        inner = [self.trainset.to_inner_iid(t) if k else 0 for t, k in zip(self.catalog.ids, known)]
        self.bi = np.where(known, self.algo.bi[inner], -np.inf)  # фильм без оценок в обучении — в конец
        self.qi = self.algo.qi[inner]
        return self

    def scores(self, user_id) -> np.ndarray:
        if not self.trainset.knows_user(_inner(self.trainset.to_inner_uid, user_id)):
            return np.zeros(len(self.catalog))
        u = self.trainset.to_inner_uid(user_id)
        return self.algo.bu[u] + self.bi + self.qi @ self.algo.pu[u]

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return np.array([self.algo.predict(u, t).est for u, t in zip(test.userId, test.tconst)])


class ALS:
    """Неявная обратная связь (implicit): факт оценки с уверенностью, растущей с оценкой."""

    def __init__(self, catalog: Catalog, factors: int = 64, regularization: float = 0.05, alpha: float = 2.0,
                 iterations: int = 20):
        self.catalog = catalog
        self.params = dict(factors=factors, regularization=regularization, alpha=alpha, iterations=iterations)

    def fit(self, train: pd.DataFrame) -> "ALS":
        from implicit.als import AlternatingLeastSquares

        users = train.userId.unique()
        self.user_index = {u: i for i, u in enumerate(users)}
        items = sparse.csr_matrix(
            (train.rating.to_numpy(np.float32), (train.userId.map(self.user_index), self.catalog.positions(train.tconst))),
            shape=(len(users), len(self.catalog)),
        )
        self.model = AlternatingLeastSquares(random_state=42, **self.params)
        self.model.fit(items, show_progress=False)
        return self

    def scores(self, user_id) -> np.ndarray:
        u = self.user_index.get(user_id)
        if u is None:
            return np.zeros(len(self.catalog))
        return self.model.item_factors @ self.model.user_factors[u]


class Hybrid:
    """Смесь коллаборативной и контентной моделей: cf_weight·cf + (1 − cf_weight)·content
    после приведения оценок обеих моделей к одному масштабу у каждого пользователя."""

    def __init__(self, cf, content: Content, cf_weight: float = 0.5):
        self.cf, self.content, self.cf_weight = cf, content, cf_weight

    def fit(self, train: pd.DataFrame) -> "Hybrid":
        return self  # части обучаются отдельно и передаются уже готовыми

    def scores(self, user_id) -> np.ndarray:
        return self.cf_weight * _unit(self.cf.scores(user_id)) + (1 - self.cf_weight) * _unit(self.content.scores(user_id))


def _unit(s: np.ndarray) -> np.ndarray:
    """Стандартизация по конечным значениям; -inf (фильм неизвестен модели) становится минимумом."""
    s = np.asarray(s, float)
    finite = np.isfinite(s)
    if not finite.any():
        return np.zeros_like(s)
    mu, sd = s[finite].mean(), s[finite].std() or 1.0
    z = (s - mu) / sd
    z[~finite] = z[finite].min()
    return z


def _inner(to_inner, raw_id) -> int | None:
    try:
        return to_inner(raw_id)
    except ValueError:
        return None
