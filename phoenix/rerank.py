"""Rerank עם גיבוי: מסדר מחדש את הקטעים שנמצאו לפי כמה הם עונים על השאלה.

הסדר (config.RERANK_CHAIN): Cohere -> מדרג מקומי -> בלי Rerank (הסדר המקורי, "מצב זהיר").
המדרג המקומי הוא מודל ONNX שמורד פעם אחת ל-data/models/<שם> (לא ב-Git) ורץ על המעבד: בלי רשת, בלי מכסה
ובלי מפתח - כך הסף לא נכבה כש-Cohere לא זמין. לכל מדרג סקאלה משלו, ולכן סף משלו (config.RERANK_THRESHOLDS).

מטמון: הציון של כל זוג (שאלה, קטע) נשמר ב-data/cache/rerank.json. Rerank מדרג כל קטע
מול השאלה בנפרד, ולכן בהרצה חוזרת שולחים ל-API רק קטעים שעוד לא דורגו - בדיקות
חוזרות לא שורפות את המכסה של Cohere (1,000 קריאות בחודש).
"""
import collections
import hashlib
import json
import logging
import math
import threading

import httpx
import numpy as np

from . import config
from .fallback import AllProvidersFailed, Provider, run_chain

log = logging.getLogger("phoenix")

CACHE_FILE = config.ROOT / "data" / "cache" / "rerank.json"
# Voyage בלי כרטיס אשראי: ~10K טוקנים לדקה. כגיבוי שולחים רק את המועמדים הראשונים, מקוצרים.
VOYAGE_MAX_DOCS, VOYAGE_DOC_CHARS = 12, 1200
LOCAL_DIR = config.ROOT / "data" / "models"
LOCAL_FILE = "model_quantized.onnx"   # int8 - פי 4 קטן ומהיר מהמקורי, כמעט בלי הבדל בדירוג
# על המעבד: 34 מועמדים x 512 טוקנים = 14 שניות. העמוד הנכון כמעט תמיד בין 20 הראשונים בסדר ה-RRF (נבדק: מקום
# 1-6, ופעם אחת 11 ו-19), ולכן מדרגים רק אותם, עד 384 טוקנים לזוג - כ-6.5 שניות, בתוך 20 השניות לשאלה.
LOCAL_MAX_DOCS, LOCAL_MAX_TOKENS, LOCAL_BATCH = 20, 384, 8
_local_models: dict[str, tuple] = {}
_local_lock = threading.Lock()

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
    def cached_call(query, documents, top_n, timeout=None):
        docs = documents[:max_docs] if max_docs else documents
        cache = _load_cache()
        keys = [hashlib.sha1(f"{name}\n{query}\n{d}".encode()).hexdigest() for d in docs]
        missing = [i for i, key in enumerate(keys) if key not in cache]
        CACHE_HITS[name] += len(docs) - len(missing)
        if missing:
            API_REQUESTS[name] += 1
            for j, score in call(query, [docs[i] for i in missing], len(missing), timeout):
                cache[keys[missing[j]]] = score
            _save_cache()
        scored = [(i, cache[key]) for i, key in enumerate(keys) if key in cache]
        return sorted(scored, key=lambda pair: -pair[1])[:top_n]
    return cached_call


def _cohere(model):
    def call(query, documents, top_n, timeout=None):
        r = httpx.post(
            "https://api.cohere.com/v2/rerank",
            headers={"Authorization": f"Bearer {config.API_KEYS['cohere']}"},
            json={"model": model, "query": query, "documents": documents, "top_n": top_n},
            timeout=timeout or config.HTTP_TIMEOUT,
        )
        r.raise_for_status()
        return [(x["index"], x["relevance_score"]) for x in r.json()["results"]]
    return call


def _voyage(model):
    def call(query, documents, top_n, timeout=None):
        r = httpx.post(
            "https://api.voyageai.com/v1/rerank",
            headers={"Authorization": f"Bearer {config.API_KEYS['voyage']}"},
            json={"model": model, "query": query, "documents": [d[:VOYAGE_DOC_CHARS] for d in documents],
                  "top_k": top_n},
            timeout=timeout or config.HTTP_TIMEOUT,
        )
        r.raise_for_status()
        return [(x["index"], x["relevance_score"]) for x in r.json()["data"]]
    return call


def local_installed(model: str) -> bool:
    return (LOCAL_DIR / model / LOCAL_FILE).exists() and (LOCAL_DIR / model / "tokenizer.json").exists()


def _load_local(model: str) -> tuple:
    """המודל נטען פעם אחת (כמה שניות), בפעם הראשונה שצריך אותו - לא בהפעלת הצ'אט."""
    with _local_lock:
        if model not in _local_models:
            import onnxruntime as ort
            from tokenizers import Tokenizer
            folder = LOCAL_DIR / model
            tok = Tokenizer.from_file(str(folder / "tokenizer.json"))
            tok.enable_truncation(LOCAL_MAX_TOKENS)  # זוג (שאלה, קטע): הקטע נחתך, השאלה נשמרת
            tok.enable_padding(pad_id=tok.token_to_id("<pad>"), pad_token="<pad>")
            session = ort.InferenceSession(str(folder / LOCAL_FILE), providers=["CPUExecutionProvider"])
            _local_models[model] = (tok, session, {i.name for i in session.get_inputs()})
        return _local_models[model]


def _local(model):
    def call(query, documents, top_n, timeout=None):
        tok, session, inputs = _load_local(model)
        scores = []
        for start in range(0, len(documents), LOCAL_BATCH):
            enc = tok.encode_batch([(query, d) for d in documents[start:start + LOCAL_BATCH]])
            feeds = {"input_ids": np.array([e.ids for e in enc], dtype=np.int64),
                     "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64)}
            if "token_type_ids" in inputs:
                feeds["token_type_ids"] = np.array([e.type_ids for e in enc], dtype=np.int64)
            logits = session.run(None, feeds)[0][:, 0]
            scores += [1 / (1 + math.exp(-float(x))) for x in logits]  # ציון 0-1, כמו אצל Cohere
        return sorted(enumerate(scores), key=lambda pair: -pair[1])[:top_n]
    return call


_FACTORIES = {"cohere": (_cohere, None), "voyage": (_voyage, VOYAGE_MAX_DOCS), "local": (_local, LOCAL_MAX_DOCS)}


def _enabled(provider: str, model: str) -> bool:
    return local_installed(model) if provider == "local" else bool(config.API_KEYS[provider])


PROVIDERS = [
    Provider(f"{p}:{m}", _cached(f"{p}:{m}", _FACTORIES[p][0](m), _FACTORIES[p][1]), enabled=_enabled(p, m))
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
        return run_chain("rerank", providers, query, documents, top_n, timeout=config.QUERY_TIMEOUT,
                         retries=config.QUERY_RETRIES, max_wait=config.QUERY_MAX_WAIT,
                         deadline=config.QUERY_TIMEOUT)
    except AllProvidersFailed as exc:
        log.warning("rerank unavailable, keeping original order: %s", exc)
        return [(i, 0.0) for i in range(top_n)], "none"


# מאיפה מורידים כל מדרג מקומי (גרסת ONNX, עם קובץ int8 מוכן)
LOCAL_REPOS = {"bge-reranker-v2-m3": "onnx-community/bge-reranker-v2-m3-ONNX",
               "gte-multilingual-reranker-base": "onnx-community/gte-multilingual-reranker-base"}


def download_local(model: str = config.LOCAL_RERANKER):
    """הורדה חד-פעמית של המדרג המקומי ל-data/models/<שם> (~590MB). הרצה: python -m phoenix.rerank download"""
    folder = LOCAL_DIR / model
    folder.mkdir(parents=True, exist_ok=True)
    for remote in [f"onnx/{LOCAL_FILE}", "tokenizer.json", "config.json"]:
        dest = folder / remote.split("/")[-1]
        if dest.exists() and dest.stat().st_size:
            print(f"{dest.name}: כבר קיים")
            continue
        part = dest.with_suffix(dest.suffix + ".part")
        with httpx.stream("GET", f"https://huggingface.co/{LOCAL_REPOS[model]}/resolve/main/{remote}",
                          follow_redirects=True, timeout=600) as r:
            r.raise_for_status()
            with open(part, "wb") as f:
                for chunk in r.iter_bytes(1 << 20):
                    f.write(chunk)
        part.replace(dest)
        print(f"{dest.name}: {dest.stat().st_size / 1e6:.0f}MB")


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["download"]:
        download_local()
    else:
        print("python -m phoenix.rerank download  - הורדת המדרג המקומי ל-data/models")
