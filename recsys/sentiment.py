"""Тональность текстового отзыва: от −1 (негатив) до +1 (позитив).

Готовая русскоязычная модель seara/rubert-tiny2-russian-sentiment (117 МБ, лицензия MIT) даёт
вероятности neutral / positive / negative; тональность = P(positive) − P(negative).
Примеры: «круто» → +0,99, «ужасная концовка, жаль потраченного времени» → −0,79, «ну такое» → −0,10.
"""
MODEL = "seara/rubert-tiny2-russian-sentiment"
_pipeline = None


def load():
    """Загрузить модель (при первом вызове скачивается, дальше ~2 с из кэша)."""
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        _pipeline = pipeline("text-classification", model=MODEL, top_k=None)
    return _pipeline


def sentiment(text: str) -> float:
    scores = {d["label"]: d["score"] for d in load()(text[:512])[0]}
    return round(scores.get("positive", 0.0) - scores.get("negative", 0.0), 3)
