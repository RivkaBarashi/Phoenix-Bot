"""יומן השיחות של הצ'אט: logs/conversations.jsonl (לא נכנס ל-Git).

כל שאלה = שורה אחת (type "answer"): השאלה, השאלה העצמאית (בשאלת המשך), ההחלטה וסיבתה, ציון ה-Rerank,
המקורות, המודל, הזמנים והתשובה שהוצגה. משוב (👍/👎) נכתב כשורה נפרדת (type "feedback") עם אותו id -
כך הקובץ רק מתארך, אף שורה לא נכתבת מחדש. report.py מחבר ביניהן.

פרטיות: בטקסט שהלקוחה כתבה (השאלה, והשאלה העצמאית שנבנתה ממנה) רצף של 7 ספרות ומעלה - ת"ז, מספר פוליסה,
טלפון - מוסתר. התשובה של הבוט לא מוסתרת: הטלפונים בה הם של המוקדים, מהאתר.
"""
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path

from . import config

# CONVERSATION_LOG: קובץ אחר (לבדיקות - כדי לא לערבב שאלות בדיקה עם שיחות אמיתיות)
LOG_FILE = Path(os.getenv("CONVERSATION_LOG") or config.ROOT / "logs" / "conversations.jsonl")
_lock = threading.Lock()  # Streamlit מריץ כל משתמש ב-thread משלו
LONG_NUMBER = re.compile(r"\d(?:[\s.\-]?\d){6,}")  # 7 ספרות ומעלה, גם עם מקף, רווח או נקודה ביניהן
HIDDEN = "[מספר הוסתר]"


def hide_numbers(text: str) -> str:
    """"ת"ז 123456789" -> "ת"ז [מספר הוסתר]". מספרים קצרים (גיל, ימים, סכומים קטנים) נשארים."""
    return LONG_NUMBER.sub(HIDDEN, text or "")


def _append(record: dict, path: Path = LOG_FILE):
    try:
        with _lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:  # יומן הוא תוספת - תקלה בו לא מפילה את הצ'אט
        pass


def log_answer(session: str, question: str, a, **extra) -> str:
    """a: phoenix.answer.Answer. מחזיר id שאליו יתחבר המשוב."""
    entry_id = uuid.uuid4().hex[:12]
    _append({"type": "answer", "id": entry_id, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "session": session,
             "question": hide_numbers(question),
             "standalone": hide_numbers(a.question) if a.question != question else "",
             "kind": a.kind, "reason": a.reason, "top_score": a.top_score, "reranker": a.reranker,
             "sources": a.sources, "llm": a.llm, "timings": {k: round(v, 2) for k, v in a.timings.items()},
             "answer": a.text, "warning": getattr(a, "warning", ""), **extra})
    return entry_id


def log_error(session: str, question: str, error: str) -> str:
    entry_id = uuid.uuid4().hex[:12]
    _append({"type": "answer", "id": entry_id, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "session": session,
             "question": hide_numbers(question), "standalone": "", "kind": "error", "reason": error, "top_score": None,
             "reranker": None, "sources": [], "llm": None, "timings": {}, "answer": ""})
    return entry_id


def log_feedback(entry_id: str, feedback: str | None):
    """feedback: "up" / "down" / None (המשוב בוטל)."""
    _append({"type": "feedback", "id": entry_id, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "feedback": feedback})


def read_log(path: Path = LOG_FILE) -> list[dict]:
    """שורות התשובה, כל אחת עם המשוב האחרון שלה (או None)."""
    answers, feedback = {}, {}
    if not path.exists():
        return []
    for line in open(path, encoding="utf-8"):
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if record.get("type") == "answer":
            answers[record["id"]] = record
        elif record.get("type") == "feedback":
            feedback[record["id"]] = record.get("feedback")
    return [entry | {"feedback": feedback.get(entry_id)} for entry_id, entry in answers.items()]
