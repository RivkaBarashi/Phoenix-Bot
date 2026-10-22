"""מטמון תשובות מאומתות, משותף לשני האתרים (Upstash - store.py).

רק תשובה שעברה את כל הבדיקות בקוד (kind=answer, reason=verified - גם אחרי ניסיון תיקון), לשאלה שלא תלויה בשיחה
(נושא חדש או בלי היסטוריה) ועם שאלה אחת. המפתח: הניסוח העצמאי של השאלה (אחרי שלב ההבנה), ועוד:
  - גרסת האינדקס - טביעת אצבע של data/chunks.jsonl ושל קבצי data/chroma: בנייה מחדש מבטלת תשובות ישנות.
  - גרסת הקוד - טביעת אצבע של קבצי phoenix/*.py: שינוי בהוראות או בבדיקות מבטל תשובות ישנות.
במצב מפתחת (?dev=1) לא קוראים מהמטמון - כך ה-smoke והריצה הסופית מודדים את הבוט עצמו, לא את המטמון.
"""
import hashlib
import json
import re
from functools import lru_cache

from . import config, store

TTL_SECONDS = 7 * 24 * 3600
_SPACES = re.compile(r"\s+")


def _fingerprint(paths) -> str:
    digest = hashlib.sha1()
    for path in paths:
        digest.update(path.relative_to(config.ROOT).as_posix().encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))  # אותה טביעה ב-Windows ובענן (לינוקס)
    return digest.hexdigest()[:10]


@lru_cache
def index_version() -> str:
    files = [config.ROOT / "data" / "chunks.jsonl"] + sorted(p for p in config.CHROMA_DIR.rglob("*") if p.is_file())
    return _fingerprint([p for p in files if p.exists()])


def code_version() -> str:
    """טביעת האצבע של הקוד, ועוד המתגים שמשנים תשובה (config.BEHAVIOR_FLAGS) והנתיבים - כולם יכולים להשתנות
    במשתנה סביבה בלי לגעת בקוד (23/10: ANSWER_FACTS=false היה מקבל תשובות ששמרה הגרסה עם "עובדות קודם")."""
    flags = "|".join(f"{name}={getattr(config, name)}" for name in config.BEHAVIOR_FLAGS) + f"|{config.ROUTES}"
    return hashlib.sha1((_files_version() + flags).encode()).hexdigest()[:10]


@lru_cache
def _files_version() -> str:
    return _fingerprint(sorted((config.ROOT / "phoenix").glob("*.py")))


def key(standalone: str) -> str:
    question = _SPACES.sub(" ", standalone).strip().rstrip("?!. ").lower()
    return f"ans:{index_version()}:{code_version()}:{hashlib.sha1(question.encode()).hexdigest()[:16]}"


def get(standalone: str) -> dict | None:
    raw = store.read("GET", key(standalone))
    try:
        return json.loads(raw) if raw else None
    except ValueError:
        return None


def put(standalone: str, record: dict) -> None:
    store.write(["SET", key(standalone), json.dumps(record, ensure_ascii=False), "EX", TTL_SECONDS])
