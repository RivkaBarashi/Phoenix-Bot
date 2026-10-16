"""Rerank עם גיבוי: מסדר מחדש את הקטעים שנמצאו לפי כמה הם עונים על השאלה.

הסדר: Cohere -> Voyage -> בלי Rerank (הסדר המקורי). הבוט עובד גם בלי, רק פחות מדויק.
שימו לב: הציונים של Cohere ושל Voyage בסקאלות שונות - סף "יש תשובה" תקף רק ל-Cohere.

מטמון: הציון של כל זוג (שאלה, קטע) נשמר ב-data/cache/rerank.json. Rerank מדרג כל קטע
מול השאלה בנפרד, ולכן בהרצה חוזרת שולחים ל-API רק קטעים שעוד לא דורגו - בדיקות
חוזרות לא שורפות את המכסה של Cohere (1,000 קריאות בחודש).
"""
import collections
import hashlib
import json
import logging

import httpx

from . import config
from .fallback import AllProvidersFailed, Provider, run_chain

log = logging.getLogger("phoenix")

CACHE_FILE = config.ROOT / "data" / "cache" / "rerank.json"
# Voyage בלי כרטיס אשראי: ~10K טוקנים לדקה. כגיבוי שולחים רק את המועמדים הראשונים, מקוצרים.
VOYAGE_MAX_DOCS, VOYAGE_DOC_CHARS = 12, 1200

API_REQUESTS: collections.Counter = collections.Counter()  # קריאות HTTP אמיתיות לכל ספק
CACHE_HITS: collections.Counter = collections.Counter()    # ציונים שהגיעו מהמטמון
_cache: dict[str, float] | None = None


def _load_cache() -> dict[str, float]:
    global _cache
    if _cache is None:
        _cache = json.loads(CACHE_FILE.read_text(encoding="utf-8")) if CACHE_FILE.exists() else {}
    return _cache


def _save_cache():
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(_cache), encoding="utf-8")


def _cached(name: str, call, max_docs: int | None = None):
    """עוטף קריאת Rerank: ציונים קיימים מהמטמון, ל-API נשלחים רק הקטעים החסרים."""
    def cached_call(query, documents, top_n):
        docs = documents[:max_docs] if max_docs else documents
        cache = _load_cache()
        keys = [hashlib.sha1(f"{name}\n{query}\n{d}".encode()).hexdigest() for d in docs]
        missing = [i for i, key in enumerate(keys) if key not in cache]
        CACHE_HITS[name] += len(docs) - len(missing)
        if missing:
            API_REQUESTS[name] += 1
            for j, score in call(query, [docs[i] for i in missing], len(missing)):
                cache[keys[missing[j]]] = score
            _save_cache()
        scored = [(i, cache[key]) for i, key in enumerate(keys) if key in cache]
        return sorted(scored, key=lambda pair: -pair[1])[:top_n]
    return cached_call


def _cohere(model):
    def call(query, documents, top_n):
        r = httpx.post(
            "https://api.cohere.com/v2/rerank",
            headers={"Authorization": f"Bearer {config.API_KEYS['cohere']}"},
            json={"model": model, "query": query, "documents": documents, "top_n": top_n},
            timeout=config.HTTP_TIMEOUT,
        )
        r.raise_for_status()
        return [(x["index"], x["relevance_score"]) for x in r.json()["results"]]
    return call


def _voyage(model):
    def call(query, documents, top_n):
        r = httpx.post(
            "https://api.voyageai.com/v1/rerank",
            headers={"Authorization": f"Bearer {config.API_KEYS['voyage']}"},
            json={"model": model, "query": query, "documents": [d[:VOYAGE_DOC_CHARS] for d in documents],
                  "top_k": top_n},
            timeout=config.HTTP_TIMEOUT,
        )
        r.raise_for_status()
        return [(x["index"], x["relevance_score"]) for x in r.json()["data"]]
    return call


_FACTORIES = {"cohere": (_cohere, None), "voyage": (_voyage, VOYAGE_MAX_DOCS)}

PROVIDERS = [
    Provider(f"{p}:{m}", _cached(f"{p}:{m}", _FACTORIES[p][0](m), _FACTORIES[p][1]), enabled=bool(config.API_KEYS[p]))
    for p, m in config.RERANK_CHAIN
]


def rerank(query: str, documents: list[str], top_n: int = 5,
           only: str | None = None) -> tuple[list[tuple[int, float]], str]:
    """מחזיר [(מיקום במקור, ציון)] מהרלוונטי ביותר, ואת שם הספק ("none" = סדר מקורי).

    only: רק ספק אחד (למשל "cohere") - כדי שציונים יהיו ניתנים להשוואה בבדיקות.
    """
    top_n = min(top_n, len(documents))
    if not top_n:
        return [], "none"
    providers = [p for p in PROVIDERS if only is None or p.name.startswith(only + ":")]
    try:
        return run_chain("rerank", providers, query, documents, top_n, retries=1, max_wait=3)
    except AllProvidersFailed as exc:
        log.warning("rerank unavailable, keeping original order: %s", exc)
        return [(i, 0.0) for i in range(top_n)], "none"
