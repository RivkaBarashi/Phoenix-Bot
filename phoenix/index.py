"""בניית האינדקסים הווקטוריים (Chroma) מתוך data/chunks.jsonl.

הרצה:  .venv\\Scripts\\python -m phoenix.index                 (שני האינדקסים)
       .venv\\Scripts\\python -m phoenix.index --only gemini

לכל ספק אמבדינגים יש אינדקס נפרד (fnx_gemini, fnx_voyage): וקטורים של מודלים
שונים לא ניתנים להשוואה, ולכן לא מערבבים ספקים באותו אינדקס.

עמידות לתקלות:
  - כל מנה נשמרת ב-Chroma מיד. הרצה חוזרת ממשיכה מאיפה שעצרה ומדלגת על קטעים
    שכבר באינדקס.
  - על 429 (חריגה ממגבלת בקשות) הספק "נח", והעבודה עוברת בינתיים לאינדקס של
    הספק השני. כשהמנוחה נגמרת - חוזרים אליו מאותה נקודה.
  - מכסה יומית/חודשית שנגמרה: האינדקס הזה נעצר והשני ממשיך. מריצים שוב מאוחר יותר.
  - קטעים שכבר לא קיימים ב-chunks.jsonl (אחרי חיתוך מחדש) נמחקים מהאינדקס.
"""
import argparse
import atexit
import json
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import chromadb
import httpx

from . import config, embeddings
from .chunk import CHUNKS_FILE
from .fallback import QUOTA, RATE_LIMIT, UNAVAILABLE, Provider, classify, retry_after

CHARS_PER_TOKEN = 2  # עברית: נמדד ~2.2 תווים לטוקן ב-Gemini ו-~2.4 ב-Voyage
BATCH_TOKENS = {"gemini": 20_000, "voyage": 8_000}  # Voyage בלי כרטיס אשראי: ~10K טוקנים לדקה
BATCH_MAX_TEXTS = {"gemini": 100, "voyage": 128}
REST_SECONDS = {RATE_LIMIT: 60, UNAVAILABLE: 20}
MAX_QUOTA_WAIT = 3700  # מכסה שמתאפסת בתוך שעה - מחכים לה; יותר מזה - עוצרים וממשיכים בהרצה הבאה
METADATA_FIELDS = ("url", "title", "heading", "fetched_at", "chunk_index", "words")


# האינדקס שב-Git (data/chroma) לא נפתח ישירות לקריאה: Chroma כותב ל-chroma.sqlite3 בכל פתיחה (נבדק 20/10) - כל
# הרצה מקומית הייתה "משנה" קובץ שב-Git, ובענן עם קבצים לקריאה בלבד הפתיחה נכשלת. לכן בקריאה (הצ'אט, הבדיקות)
# נפתח עותק בתיקייה זמנית, פעם אחת לתהליך (65MB, פחות משנייה); רק הבנייה (python -m phoenix.index) כותבת לתיקייה.
WRITE = False


@lru_cache
def db() -> chromadb.ClientAPI:
    path = config.CHROMA_DIR
    if not WRITE and path.exists():
        path = Path(tempfile.mkdtemp(prefix="phoenix-chroma-")) / "chroma"
        shutil.copytree(config.CHROMA_DIR, path)
        atexit.register(shutil.rmtree, path.parent, ignore_errors=True)
    return chromadb.PersistentClient(path=str(path))


def collection_name(provider: str) -> str:
    return f"fnx_{provider.split(':')[0]}"  # "gemini:gemini-embedding-2" -> "fnx_gemini"


def get_collection(provider: str, create: bool = False):
    """האינדקס של ספק, או None אם אין.

    אינדקס שנבנה במודל אחר או בפורמט קלט אחר (embeddings.INPUT_FORMAT) לא תואם לשאלות
    של היום: create=True בונה אותו מחדש, create=False מחזיר None.
    """
    name = collection_name(provider)
    model = provider.split(":", 1)[1].split("#")[0]  # "gemini-embedding-2#3" - אותו מודל, מפתח 3
    wanted = {"model": model, "input_format": embeddings.INPUT_FORMAT[provider.split(":")[0]]}
    if name in {c.name for c in db().list_collections()}:
        col = db().get_collection(name, embedding_function=None)
        meta = col.metadata or {}
        if meta.get("model") == model and meta.get("input_format", embeddings.LEGACY_FORMAT) == wanted["input_format"]:
            return col
        if not create:
            return None
        db().delete_collection(name)
    if not create:
        return None
    return db().create_collection(name, metadata=wanted,
                                  configuration={"hnsw": {"space": "cosine"}}, embedding_function=None)


def metadata(chunk: dict) -> dict:
    return {k: chunk[k] for k in METADATA_FIELDS}


def limit_reason(exc: Exception) -> str:
    """ההסבר של הספק למגבלה שנחצתה, בקצרה."""
    if isinstance(exc, httpx.HTTPStatusError):
        text = exc.response.text
    else:
        text = getattr(exc, "message", None) or str(exc)
    quota_ids = re.findall(r"quotaId'?\"?:\s*'?\"?([\w-]+)", str(exc))
    return ", ".join(sorted(set(quota_ids))) or " ".join(text.split())[:200]


@dataclass
class Job:
    provider: Provider
    col: object
    pending: list[dict]
    total: int
    rest_until: float = 0.0
    stopped: str = ""
    quota_strikes: int = 0
    spares: list = None  # מפתחות נוספים לאותו מודל: מכסה של מפתח נגמרה - עוברים למפתח הבא בלי לחכות

    @property
    def kind(self) -> str:
        return self.provider.name.split(":")[0]

    def next_batch(self) -> list[dict]:
        batch, tokens = [], 0
        for chunk in self.pending:
            t = len(chunk["text"]) // CHARS_PER_TOKEN + 1
            if batch and (tokens + t > BATCH_TOKENS[self.kind] or len(batch) >= BATCH_MAX_TEXTS[self.kind]):
                break
            batch.append(chunk)
            tokens += t
        return batch


def build(jobs: list[Job]):
    while True:
        active = [j for j in jobs if j.pending and not j.stopped]
        if not active:
            return
        ready = [j for j in active if j.rest_until <= time.time()]
        if not ready:
            time.sleep(max(1.0, min(j.rest_until for j in active) - time.time()))
            continue
        for job in ready:
            batch = job.next_batch()
            try:
                vectors = job.provider.call([c["text"] for c in batch], False)
            except Exception as exc:
                kind = classify(exc)
                if kind in REST_SECONDS:
                    wait = min(retry_after(exc) or REST_SECONDS[kind], 120)
                    job.rest_until = time.time() + wait
                    others = [j.kind for j in active if j is not job]
                    print(f"  {job.kind}: {kind}, ממתין {wait:.0f} שניות"
                          + (f" - ובינתיים ממשיך עם {', '.join(others)}" if others else "")
                          + f"  [{limit_reason(exc)}]", flush=True)
                elif kind == QUOTA and job.spares:
                    job.provider = job.spares.pop(0)
                    print(f"  {job.kind}: מכסה יומית במפתח הזה - ממשיך במפתח הבא ({job.provider.name})", flush=True)
                elif kind == QUOTA and (wait := retry_after(exc)) and wait <= MAX_QUOTA_WAIT:
                    # מכסה יומית שהספק אומר מתי היא מתאפסת (ג'מיני: 1,000 בקשות ביום) - מחכים ואז ממשיכים
                    job.rest_until = time.time() + wait + 30
                    print(f"  {job.kind}: מכסה יומית ({limit_reason(exc)}) - מתאפסת בעוד {wait / 60:.0f} דקות, "
                          f"ממתין וממשיך מאותה נקודה", flush=True)
                elif kind == QUOTA and job.quota_strikes == 0:
                    # לפעמים זו מגבלה לחלון זמן ארוך ולא מכסה יומית - מחכים 5 דקות ומנסים שוב
                    job.quota_strikes = 1
                    job.rest_until = time.time() + 300
                    print(f"  {job.kind}: נראה כמו מכסה - ממתין 5 דקות ומנסה שוב  [{' '.join(str(exc).split())[:400]}]",
                          flush=True)
                elif kind == QUOTA:
                    job.stopped = "המכסה נגמרה - להריץ שוב מאוחר יותר, זה ימשיך מאותה נקודה"
                else:
                    job.stopped = f"{kind}: {str(exc)[:200]}"
                continue
            job.col.add(
                ids=[c["id"] for c in batch],
                embeddings=vectors,
                documents=[c["text"] for c in batch],
                metadatas=[metadata(c) for c in batch],
            )
            job.pending = job.pending[len(batch):]
            job.quota_strikes = 0
            print(f"  {job.kind}: {job.total - len(job.pending)}/{job.total}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=list(dict.fromkeys(p.name.split(":")[0] for p in embeddings.PROVIDERS)))
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    global WRITE
    WRITE = True  # הבנייה - בתיקייה עצמה, לא בעותק

    chunks = [json.loads(line) for line in open(CHUNKS_FILE, encoding="utf-8")]
    ids = {c["id"] for c in chunks}
    jobs = []
    by_collection: dict[str, list[Provider]] = {}  # אינדקס אחד לכל ספק, וכל המפתחות שלו כגיבוי
    for p in embeddings.PROVIDERS:
        if p.enabled and not (args.only and not p.name.startswith(args.only + ":")):
            by_collection.setdefault(collection_name(p.name), []).append(p)
    for p, *spares in by_collection.values():
        col = get_collection(p.name, create=True)
        stored = col.get(include=["documents", "embeddings"])
        have = set(stored["ids"])
        # קטע שהטקסט שלו כבר הוטמע (גם אם המזהה השתנה) מקבל את הווקטור הקיים - בלי קריאה ל-API
        by_text = dict(zip(stored["documents"], stored["embeddings"]))
        missing = [c for c in chunks if c["id"] not in have]
        reuse = [c for c in missing if c["text"] in by_text]
        if reuse:
            col.add(ids=[c["id"] for c in reuse], embeddings=[list(by_text[c["text"]]) for c in reuse],
                    documents=[c["text"] for c in reuse], metadatas=[metadata(c) for c in reuse])
        current = [c for c in chunks if c["id"] in have]
        if current:  # מספור הקטעים בעמוד יכול להשתנות גם כשהטקסט לא
            col.update(ids=[c["id"] for c in current], metadatas=[metadata(c) for c in current])
        stale = list(have - ids)
        if stale:
            col.delete(ids=stale)
        pending = [c for c in missing if c["text"] not in by_text]
        print(f"{collection_name(p.name)} ({p.name}): {len(current)} כבר באינדקס, {len(reuse)} הועתקו "
              f"מווקטור קיים (טקסט זהה), {len(pending)} להטמעה, {len(stale)} ישנים נמחקו")
        jobs.append(Job(p, col, pending, len(chunks), spares=spares))

    start = time.time()
    build(jobs)
    print(f"\nסיום אחרי {time.time() - start:.0f} שניות:")
    for job in jobs:
        print(f"  {collection_name(job.provider.name)}: {job.col.count()}/{len(chunks)} קטעים"
              + (f" - נעצר: {job.stopped}" if job.stopped else ""))


if __name__ == "__main__":
    main()
