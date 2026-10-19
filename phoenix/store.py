"""אחסון קבוע ומשותף לשני האתרים (Streamlit Cloud ו-Render): Upstash Redis, דרך ה-REST API שלו (רק httpx).

בענן הדיסק נמחק בכל הפעלה מחדש, ולכן מה שצריך להישאר נשמר כאן:
  - log            - היומן והמשוב (אותן שורות כמו ב-logs/conversations.jsonl, עם site: מאיזה אתר).
  - conv:<id>      - השיחה בדפדפן (?c=<id> בכתובת): רענון הדף מחזיר אותה. 24 שעות.
  - ans:<...>      - מטמון של תשובות שעברו את כל הבדיקות (answer_cache.py). 7 ימים.
  - calls:<ספק>:<חודש> - מונה קריאות ל-Cohere ול-Voyage (למכסה החודשית; Cohere לא מדווח כמה נשאר).

בלי UPSTASH_REDIS_REST_URL ו-UPSTASH_REDIS_REST_TOKEN, או כש-Upstash לא זמין - הבוט עונה כרגיל, רק בלי לשמור:
כתיבה נשלחת ברקע (לא מעכבת תשובה), קריאה מחכה עד READ_TIMEOUT, ואחרי תקלה - דקה בלי לנסות, כך שתקלה ב-Upstash
לא מוסיפה עיכוב לכל שאלה.
"""
import logging
import os
import threading
import time
from typing import Any

import httpx

from . import config  # noqa: F401 - טוען את .env (ואת הסודות של Streamlit Cloud) לפני שקוראים את המשתנים

log = logging.getLogger("phoenix")

URL = os.getenv("UPSTASH_REDIS_REST_URL", "").strip().rstrip("/")
TOKEN = os.getenv("UPSTASH_REDIS_REST_TOKEN", "").strip()
READ_TIMEOUT, WRITE_TIMEOUT, PAUSE_AFTER_ERROR = 1.5, 5.0, 60.0
_paused_until = 0.0


def enabled() -> bool:
    return bool(URL and TOKEN) and time.time() >= _paused_until


def _send(commands: list[list], timeout: float) -> list | None:
    """כמה פקודות בבקשה אחת (pipeline). מחזיר את התוצאות לפי הסדר, או None כש-Upstash לא זמין."""
    global _paused_until
    if not enabled():
        return None
    try:
        r = httpx.post(f"{URL}/pipeline", headers={"Authorization": f"Bearer {TOKEN}"},
                       json=[[str(part) for part in c] for c in commands], timeout=timeout)
        r.raise_for_status()
        return [item.get("result") for item in r.json()]
    except Exception as exc:  # noqa: BLE001 - אחסון הוא תוספת: כל תקלה -> ממשיכים בלי לשמור
        _paused_until = time.time() + PAUSE_AFTER_ERROR
        log.warning("upstash unavailable (%s) - continuing without saving for a minute", str(exc)[:150])
        return None


def read(*command) -> Any:
    """פקודת קריאה אחת (GET, LRANGE...) - מחכים לה עד READ_TIMEOUT. None כשאין או לא זמין."""
    results = _send([list(command)], READ_TIMEOUT)
    return results[0] if results else None


def write(*commands: list, wait: bool = False) -> None:
    """פקודות כתיבה (RPUSH, SET, INCR...) - ברקע, בלי לחכות. wait=True - מחכים (סקריפטים ובדיקות)."""
    if not enabled() or not commands:
        return
    if wait:
        _send(list(commands), WRITE_TIMEOUT)
    else:
        threading.Thread(target=_send, args=(list(commands), WRITE_TIMEOUT), daemon=True).start()


def month() -> str:
    """החודש של מונה הקריאות (UTC - כמו המכסה החודשית אצל הספקים)."""
    return time.strftime("%Y-%m", time.gmtime())


def count_call(service: str) -> None:
    """קריאה שיצאה לספק עם מכסה חודשית (cohere / voyage)."""
    write(["INCR", f"calls:{service}:{month()}"])


def calls_this_month(services: tuple[str, ...] = ("cohere", "voyage")) -> dict[str, int] | None:
    results = _send([["GET", f"calls:{s}:{month()}"] for s in services], READ_TIMEOUT)
    return None if results is None else {s: int(v or 0) for s, v in zip(services, results)}
