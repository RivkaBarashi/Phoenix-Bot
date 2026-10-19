"""חיפוש הקטעים הרלוונטיים לשאלה: חיפוש משולב ואז דירוג מחדש.

1. וקטורים: CANDIDATES קטעים מהאינדקס של EMBEDDER (gemini/voyage). אם הספק לא זמין -
   מהאינדקס של השני; אם שניהם לא זמינים - ממשיכים רק עם מילות מפתח.
2. מילות מפתח (BM25 עם טיפול בתחיליות): CANDIDATES קטעים. רץ מקומית, בלי API.
3. איחוד המועמדים וסידור ראשוני לפי Reciprocal Rank Fusion (מי שגבוה בשתי השיטות - ראשון).
4. Rerank (Cohere -> Voyage -> בלי) מדרג את כל המועמדים לפי כמה הם עונים על השאלה.
   ציון ה-Rerank של הקטע הראשון הוא מה שיקבע אם בכלל יש תשובה באתר.
"""
import logging
import time
from dataclasses import dataclass, field

from . import bm25, config, embeddings, rerank
from .fallback import AllProvidersFailed
from .index import get_collection

log = logging.getLogger("phoenix")

CANDIDATES = 20  # מכל שיטה
RRF_K = 60       # הקבוע המקובל ב-Reciprocal Rank Fusion


@dataclass
class Hit:
    id: str
    url: str
    title: str
    heading: str
    text: str
    vector_rank: int | None = None  # המקום בחיפוש הווקטורי (None = לא היה בין המועמדים)
    bm25_rank: int | None = None    # המקום בחיפוש מילות המפתח
    score: float = 0.0              # ציון ה-Rerank (0-1); בלי Rerank - ציון RRF


@dataclass
class SearchResult:
    hits: list[Hit]
    embedder: str | None  # באיזה אינדקס וקטורי חיפשנו (None = רק מילות מפתח)
    reranker: str         # מי דירג ("none" = סדר RRF, בלי ציון מכויל)
    timings: dict[str, float] = field(default_factory=dict)  # שניות לכל שלב: vector, bm25, rerank


def vector_search(query: str, n: int, embedder: str | None = None, fallback: bool = True) -> tuple[list[Hit], str]:
    preferred = embedder or config.EMBEDDER
    order = sorted((p.name for p in embeddings.PROVIDERS if p.enabled), key=lambda name: not name.startswith(preferred + ":"))
    if not fallback:  # רק האינדקס שביקשו - אבל בכל המפתחות שלו
        order = [name for name in order if name.startswith(preferred + ":")]
    opened: dict[str, object] = {}  # אינדקס אחד לכל ספק, גם כשיש לו כמה מפתחות
    for name in order:
        if name.split(":")[0] not in opened:
            opened[name.split(":")[0]] = get_collection(name)
    collections = {name: col for name in order if (col := opened[name.split(":")[0]]) is not None and col.count()}
    if not collections:
        raise RuntimeError(f"אין אינדקס עבור {order} - צריך להריץ: python -m phoenix.index")
    vector, used = embeddings.embed_query(query, list(collections))
    res = collections[used].query(query_embeddings=[vector], n_results=n, include=["documents", "metadatas"])
    hits = [Hit(i, m["url"], m["title"], m["heading"], doc)
            for i, m, doc in zip(res["ids"][0], res["metadatas"][0], res["documents"][0])]
    return hits, used


def candidates(query: str, n: int = CANDIDATES, embedder: str | None = None, fallback: bool = True,
               timings: dict | None = None) -> tuple[list[Hit], str | None]:
    """המועמדים משני החיפושים, מאוחדים ומסודרים לפי RRF. מחזיר (מועמדים, האינדקס הווקטורי).
    timings: מילון שמקבל את זמני השלבים (vector = וקטור לשאלה + חיפוש באינדקס, bm25)."""
    timings = {} if timings is None else timings
    start = time.perf_counter()
    try:
        vector_hits, used = vector_search(query, n, embedder, fallback)
    except (AllProvidersFailed, RuntimeError) as exc:
        log.warning("vector search unavailable, using keywords only: %s", exc)
        vector_hits, used = [], None
    timings["vector"] = time.perf_counter() - start
    start = time.perf_counter()
    merged: dict[str, Hit] = {}
    for rank, hit in enumerate(vector_hits, 1):
        hit.vector_rank = rank
        merged[hit.id] = hit
    for rank, (chunk, _) in enumerate(bm25.search(query, n), 1):
        hit = merged.setdefault(chunk["id"], Hit(chunk["id"], chunk["url"], chunk["title"], chunk["heading"], chunk["text"]))
        hit.bm25_rank = rank
    timings["bm25"] = time.perf_counter() - start
    for hit in merged.values():
        hit.score = sum(1 / (RRF_K + r) for r in (hit.vector_rank, hit.bm25_rank) if r)
    return sorted(merged.values(), key=lambda h: -h.score), used


def rerank_hits(query: str, hits: list[Hit], reranker: str | None = None,
                timings: dict | None = None) -> tuple[list[Hit], str]:
    """מדרג את המועמדים מחדש. אם אין Rerank זמין - נשאר סדר ה-RRF."""
    start = time.perf_counter()
    order, reranked_by = rerank.rerank(query, [h.text for h in hits], top_n=len(hits), only=reranker)
    if timings is not None:
        timings["rerank"] = time.perf_counter() - start
    if reranked_by == "none":
        return hits, reranked_by
    for i, score in order:
        hits[i].score = score
    return [hits[i] for i, _ in order], reranked_by


def search(query: str, k: int = 5, embedder: str | None = None, fallback: bool = True,
           reranker: str | None = None) -> SearchResult:
    """k הקטעים הרלוונטיים ביותר. reranker: רק ספק Rerank אחד (לבדיקות)."""
    timings: dict[str, float] = {}
    hits, used = candidates(query, embedder=embedder, fallback=fallback, timings=timings)
    hits, reranked_by = rerank_hits(query, hits, reranker, timings=timings)
    return SearchResult(hits[:k], used, reranked_by, timings)
