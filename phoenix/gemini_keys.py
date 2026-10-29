"""מפתחות Gemini בבוט (החלטת הבעלים 29/10): עובדים על מפתח אחד, ועוברים למפתח הבא רק כשהמכסה היומית שלו נגמרה.

  - כל מודל (gemini-3.6-flash, gemini-3.8-flash, האמבדינג של השאלה) מתחיל במפתח הראשי (config.LLM_GEMINI_KEY), ואחריו
    שאר המפתחות לפי המספר. מפתח בתשלום - המכסה לא נגמרת, והבוט נשאר עליו; מפתח אחד בלבד - אין לאן לעבור.
  - רק מכסה יומית (429 "per day") או מפתח שלא עובד בכלל (401/403, מפתח לא תקין) מעבירים מפתח. עומס (503/504), timeout
    ו-429 לדקה - לא: אלה המודל ולא המפתח, ושער המודלים (gateway.py) עובר למודל הבא בנתיב, כמו קודם.
  - מפתח כזה נח עד האיפוס היומי - חצות בשעון פסיפיק (10:00 בישראל; 09:00 בין סוף שעון הקיץ בישראל לזה של ארה"ב). לא
    לפי retryDelay של Google: הוא אמר 22 שעות, והמכסה התאפסה בחצות (19/10).
  - זיכרון משותף ב-Upstash (store): שני האתרים, ותהליך שעלה מחדש, יודעים מיד אילו מפתחות נגמרו - בלי לנסות כל אחד מהם
    (ניסיון כזה לוקח 1-3 שניות, ועם הרבה מפתחות גמורים השאלות הראשונות נגמרו בלי תשובה - 21/10).
  - כל המפתחות של המודל נגמרו - QuotaExhausted (נספרת כמכסה), והשער עובר למודל הבא בנתיב.
  - כלי הבדיקות (eval_batches.py) מכבים את המעבר (GEMINI_KEY_SWITCH=false): כל חבילה על המפתח שלה בלבד.
"""
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Callable, TypeVar
from zoneinfo import ZoneInfo

from . import config
from .fallback import AUTH, QUOTA, QuotaExhausted, classify

log = logging.getLogger("phoenix")
T = TypeVar("T")

PACIFIC = ZoneInfo("America/Los_Angeles")
SHARED = "rest:gemini_keys"   # hash ב-Upstash: "gemini-3.6-flash#4" -> עד מתי (epoch)
SYNC_EVERY = 60.0             # קריאה מהזיכרון המשותף: בפעם הראשונה בתהליך מיד, אחר כך ברקע פעם בדקה

_until: dict[str, float] = {}
_lock = threading.Lock()
_synced = 0.0
_answered = threading.local()


def quota_reset(now: float | None = None) -> float:
    """מתי המכסה היומית של Gemini מתאפסת: חצות הבא בשעון פסיפיק (epoch)."""
    pacific = datetime.fromtimestamp(time.time() if now is None else now, PACIFIC)
    return datetime.combine(pacific.date() + timedelta(days=1), datetime.min.time(), PACIFIC).timestamp()


def order() -> list[int]:
    """המפתחות לפי הסדר: הראשי, ואחריו השאר לפי המספר. GEMINI_KEY_SWITCH כבוי - רק הראשי."""
    first = config.LLM_GEMINI_KEY
    if first is None:
        return []
    if not config.GEMINI_KEY_SWITCH:
        return [first]
    return [first] + [n for n, _ in config.GEMINI_KEYS if n != first]


def resting(model: str, number: int) -> bool:
    with _lock:
        return _until.get(f"{model}#{number}", 0.0) > time.time()


def rest(model: str, number: int, kind: str) -> None:
    """המפתח לא משרת את המודל עד האיפוס - כאן, ובזיכרון המשותף."""
    name, until = f"{model}#{number}", quota_reset()
    with _lock:
        _until[name] = until
    log.warning("gemini key %s: %s - next key until the daily reset", name, kind)
    from . import store
    store.write(["HSET", SHARED, name, f"{until:.0f}"], ["EXPIRE", SHARED, 30 * 3600])


def load_shared() -> int:
    """המפתחות שנגמרו (מכל תהליך ומכל אתר). מחזיר כמה נטענו."""
    from . import store
    flat = store.read("HGETALL", SHARED) or []
    loaded, now = 0, time.time()
    for name, until in zip(flat[::2], flat[1::2]):
        try:
            until = float(until)
        except (TypeError, ValueError):
            continue
        with _lock:
            if until > now and _until.get(name, 0.0) < until:
                _until[name] = until
                loaded += 1
    return loaded


def _sync() -> None:
    global _synced
    if len(order()) < 2:  # מפתח אחד - אין מה לזכור
        return
    now = time.time()
    with _lock:
        if now - _synced < SYNC_EVERY:
            return
        first, _synced = _synced == 0.0, now
    if first:
        load_shared()
    else:
        threading.Thread(target=load_shared, daemon=True).start()


def call(model: str, ask: Callable[[int], T]) -> T:
    """ask(מספר מפתח) על המפתח הראשון שעוד משרת את המודל. מכסה יומית נגמרה (או מפתח לא תקין) - המפתח נח עד האיפוס,
    ומיד הבא בתור. כל תקלה אחרת עוברת הלאה כמו שהיא. המפתח שענה - answered_by(), באותו thread."""
    _sync()
    last: Exception | None = None
    for number in order():
        if resting(model, number):
            continue
        try:
            value = ask(number)
        except Exception as exc:  # noqa: BLE001 - רק מכסה / מפתח לא תקין מטופלים כאן
            kind = classify(exc)
            if kind not in (QUOTA, AUTH):
                raise
            rest(model, number, kind)
            last = exc
            continue
        _answered.number = number
        return value
    raise QuotaExhausted(f"{model}: daily quota used up on every key ({len(order())})") from last


def answered_by() -> int | None:
    """המפתח שענה בקריאה האחרונה ב-thread הזה."""
    return getattr(_answered, "number", None)
