"""Метрики качества рекомендаций.

recommended — список фильмов в порядке выдачи, relevant — множество понравившихся фильмов из теста.
"""
import numpy as np


def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    return len(set(recommended[:k]) & relevant) / k


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    return len(set(recommended[:k]) & relevant) / len(relevant) if relevant else 0.0


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    """Нормированный DCG: попадание на позиции i (с нуля) весит 1 / log2(i + 2)."""
    dcg = sum(1 / np.log2(i + 2) for i, item in enumerate(recommended[:k]) if item in relevant)
    ideal = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


def rmse(predicted, actual) -> float:
    predicted, actual = np.asarray(predicted, float), np.asarray(actual, float)
    return float(np.sqrt(np.mean((predicted - actual) ** 2)))
