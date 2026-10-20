"""מנגנון הגיבוי המשותף לכל המטרות (מודל שפה, אמבדינגים, Rerank, איסוף).

run_chain מנסה ספק אחרי ספק לפי הסדר. ספק = מודל + מפתח (ב-Gemini יש כמה מפתחות). ספק שנחסם "נח"
לזמן מה, כדי שלא נבזבז עליו זמן בכל בקשה:
  - מכסה יומית: הצירוף מפתח+מודל נח עד האיפוס שהספק מדווח, ואותו מודל ממשיך במפתח הבא.
  - מגבלה לדקה (429): הצירוף נח עד שהספק אומר שהוא פנוי (בבדיקות: ניסיון חוזר אחד אחרי המתנה קצרה).
  - עומס (503/504) או חריגת זמן: המודל נח 5 דקות בכל המפתחות, ועוברים מיד למודל הבא - בלי המתנה.
    בסקריפטי הבדיקות (config.patient) - כמו לפני כן: ניסיון חוזר אחד, ואז רק הצירוף מפתח+מודל נח 30 שניות.
  - מפתח לא תקין: המפתח נח בכל המודלים.
"""
import collections
import contextlib
import contextvars
import logging
import random
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import httpx

log = logging.getLogger("phoenix")

RATE_LIMIT = "rate_limit"    # 429: יותר מדי בקשות בדקה
QUOTA = "quota"              # מכסה יומית/חודשית נגמרה, או 402
AUTH = "auth"                # 401/403: מפתח לא תקין
NOT_FOUND = "not_found"      # 404: המודל לא קיים
UNAVAILABLE = "unavailable"  # 5xx, timeout, תקלת רשת
EMPTY = "empty"              # הספק ענה, אבל בלי תוכן שימושי
ERROR = "error"              # כל השאר (400 וכו')

# עומס (503/504) או חריגת זמן: לא מחכים ולא מנסים שוב את אותו מודל - עוברים מיד למודל הבא.
RETRYABLE = {RATE_LIMIT}
OVERLOAD_REST = 300  # עומס: המודל מדולג 5 דקות בלבד (בכל המפתחות), לא עד איפוס המכסה
# זיכרון העומס - לצ'אט. config.patient() (סקריפטי הבדיקות) מכבה אותו: עומס מקבל ניסיון חוזר באותו מודל, ואז
# רק הצירוף מפתח+מודל נח PATIENT_OVERLOAD_REST שניות - כמו לפני כן. כך ריצה על מודל אחד קבוע לא מאבדת אותו
# ל-5 דקות (ונעצרת אחרי 3 כישלונות) בגלל 503 אחד.
OVERLOAD_MEMORY = True
PATIENT_OVERLOAD_REST = 30
# עומס שחוזר מהר (503 "high demand", תקלת רשת) הוא עומס על הבקשה הבודדת, לא על המודל: נבדק 21/10 בלילה - 503 של 3.6
# ב-13 מתוך 31 שאלות, ובכל ה-13 אותו מודל ענה במפתח הבא. לכן: רק הצירוף מפתח+מודל נח FAST_OVERLOAD_REST שניות, ואותו
# מודל ממשיך מיד במפתח הבא. FAST_OVERLOAD_LIMIT תשובות כאלה ברצף באותה בקשה - עומס על המודל: הוא נח בכל המפתחות
# FAST_OVERLOAD_GROUP_REST שניות. עומס איטי (timeout, 504 אחרי 15-20 שניות) - כמו קודם: המודל נח OVERLOAD_REST בכל המפתחות.
FAST_OVERLOAD_REST = 60
FAST_OVERLOAD_LIMIT = 4
FAST_OVERLOAD_GROUP_REST = 60
COOLDOWN_SECONDS = {
    RATE_LIMIT: 60,
    QUOTA: 3600,
    UNAVAILABLE: OVERLOAD_REST,
    AUTH: float("inf"),
    NOT_FOUND: float("inf"),
}

_resting_until: dict[str, float] = {}

# כמה קריאות נשלחו לכל ספק בתהליך הנוכחי (כולל כאלה שנכשלו) - כדי לעקוב אחרי מכסות
CALLS: collections.Counter = collections.Counter()
FAILED: collections.Counter = collections.Counter()  # מתוכן - כמה נכשלו
LAST_ERRORS: dict[str, list[str]] = {}  # לכל מטרה (llm, rerank...): השגיאות בפעם האחרונה שכל הספקים נכשלו


def resting() -> dict[str, float]:
    """ספקים ש"נחים" עכשיו (אחרי מכסה/עומס) -> כמה שניות נשארו."""
    now = time.time()
    return {name: until - now for name, until in _resting_until.items() if until > now}


class EmptyResult(Exception):
    """הספק החזיר תשובה ריקה - עוברים לבא בתור."""


class StopChain(Exception):
    """תקלה שספק אחר לא יפתור (למשל עמוד שלא קיים) - עוצרים מיד."""


class AllProvidersFailed(Exception):
    def __init__(self, purpose: str, errors: list[str]):
        self.purpose = purpose
        self.errors = errors
        super().__init__(f"{purpose}: all providers failed: " + " | ".join(errors))


class OutOfTime(AllProvidersFailed):
    """נגמר הזמן הכולל לשאלה (time_budget) - לא שולחים עוד ניסיון לאף ספק."""


# זמן כולל לשאלה אחת בצ'אט: כל ניסיון אצל ספק מקבל רק את מה שנשאר. פחות מ-MIN_ATTEMPT_SECONDS - לא
# מנסים בכלל: מודל לא יספיק לענות, והבקשה עדיין תיספר במכסה.
MIN_ATTEMPT_SECONDS = 3.0
_question_deadline: contextvars.ContextVar[float | None] = contextvars.ContextVar("question_deadline", default=None)


@contextlib.contextmanager
def time_budget(seconds: float | None):
    """with time_budget(20): ... - כל הקריאות בתוך הבלוק (באותו thread) חולקות 20 שניות. None = בלי מגבלה."""
    token = _question_deadline.set(time.monotonic() + seconds if seconds else None)
    try:
        yield
    finally:
        _question_deadline.reset(token)


def time_left() -> float | None:
    deadline = _question_deadline.get()
    return None if deadline is None else deadline - time.monotonic()


@dataclass
class Provider:
    name: str                 # למשל "gemini:gemini-3.8-flash#2" (מודל + מספר מפתח)
    call: Callable[..., Any]
    enabled: bool = True      # False כשאין מפתח ב-.env
    group: str = ""           # המודל בכל המפתחות ("gemini:gemini-3.8-flash"): עומס (503/504) מנוחה לכולם
    key: str = ""             # המפתח בכל המודלים ("gemini#2"): מפתח לא תקין - מנוחה לכולם


def is_resting(p: Provider) -> bool:
    """הצירוף מפתח+מודל, המודל (עומס) או המפתח (לא תקין) עכשיו בהפסקה."""
    now = time.time()
    return any(_resting_until.get(k, 0) > now for k in (p.name, p.group, p.key) if k)


def _status_code(exc: Exception) -> int | None:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code
    code = getattr(exc, "code", None)  # שגיאות של google-genai
    return code if isinstance(code, int) else None


def _error_text(exc: Exception) -> str:
    """הודעת השגיאה כולל גוף התשובה: Groq כותב "tokens per day" רק בגוף, לא בשורת הסטטוס."""
    text = str(exc)
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            text += " " + exc.response.text[:1000]
        except Exception:  # גוף שלא נקרא (stream)
            pass
    return text


def classify(exc: Exception) -> str:
    if isinstance(exc, EmptyResult):
        return EMPTY
    if isinstance(exc, (httpx.TransportError, TimeoutError, ConnectionError)):
        return UNAVAILABLE
    code = _status_code(exc)
    if code == 429:
        msg = _error_text(exc).lower()
        long_term = any(k in msg for k in ("perday", "per day", "per-day", "daily", "month"))
        return QUOTA if long_term else RATE_LIMIT
    if code == 402:
        return QUOTA
    if code in (401, 403):
        return AUTH
    if code == 404:
        return NOT_FOUND
    if code is not None and code >= 500:
        return UNAVAILABLE
    return ERROR


def fast_overload(exc: Exception) -> bool:
    """עומס שחוזר מהר (503, 500, תקלת רשת) - לא timeout ולא 504/DEADLINE_EXCEEDED, שמגיעים רק אחרי המתנה ארוכה."""
    if isinstance(exc, TimeoutError):
        return False
    return _status_code(exc) != 504 and "DEADLINE" not in _error_text(exc).upper()


def retry_after(exc: Exception) -> float | None:
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            return float(exc.response.headers.get("retry-after", ""))
        except ValueError:
            pass
    # בתוך ההודעה: ג'מיני "retryDelay': '3068s'" / "Please retry in 51m8.04s", Groq "try again in 34m37.9s"
    text = _error_text(exc)
    if match := re.search(r"retryDelay'?\"?:\s*'?\"?([\d.]+)s", text):
        return float(match.group(1))
    if match := re.search(r"(?:retry|try again) in (?:(\d+)h)?\s*(?:(\d+)m)?\s*([\d.]+)\s*s", text, re.IGNORECASE):
        return int(match.group(1) or 0) * 3600 + int(match.group(2) or 0) * 60 + float(match.group(3))
    return None


def with_deadline(fn: Callable[..., Any], seconds: float, *args, **kwargs):
    """מריץ fn עם מגבלת זמן אמיתית (שעון קיר). timeout של httpx הוא רק מרווח בין בתים, ו-OpenRouter
    שולח רווחים כדי להשאיר את החיבור פתוח - כך בקשה "עם timeout של 15 שניות" נמשכה 166 שניות.
    בקשה שלא סיימה בזמן ממשיכה ברקע עד שתיגמר, ואנחנו כבר עוברים לספק הבא."""
    result: dict[str, Any] = {}

    def run():
        try:
            result["value"] = fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - מועבר הלאה כמו שהוא
            result["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(seconds)
    if thread.is_alive():
        raise TimeoutError(f"no answer within {seconds:.0f}s")
    if "error" in result:
        raise result["error"]
    return result["value"]


def run_chain(purpose: str, providers: list[Provider], *args, retries: int = 1, max_wait: float = 10.0,
              cooldown: bool = True, deadline: float | None = None, **kwargs):
    """מחזיר (תוצאה, שם הספק שהצליח). זורק AllProvidersFailed אם כולם נכשלו.
    deadline: מגבלת זמן לכל ניסיון אצל ספק (בשניות). ספק שלא עונה בזמן - עוברים לבא."""
    errors = []
    fast_overloads: collections.Counter = collections.Counter()  # עומס מהיר לכל מודל, בבקשה הזו
    for p in providers:
        if not p.enabled:
            continue
        if is_resting(p):
            errors.append(f"{p.name}: resting")
            continue
        for attempt in range(retries + 1):
            left = time_left()
            if left is not None and left < MIN_ATTEMPT_SECONDS:  # נגמר הזמן לשאלה - בלי ניסיון נוסף
                errors.append(f"{p.name}: out of time")
                LAST_ERRORS[purpose] = errors
                raise OutOfTime(purpose, errors)
            limit = deadline if left is None else min(deadline or left, left)
            cut = left is not None and (not deadline or left < deadline)  # הניסיון מוגבל בזמן שנשאר לשאלה
            try:
                CALLS[p.name] += 1
                if limit:
                    return with_deadline(p.call, limit, *args, **kwargs), p.name
                return p.call(*args, **kwargs), p.name
            except StopChain:
                raise
            except Exception as exc:
                FAILED[p.name] += 1
                kind = classify(exc)
                if isinstance(exc, TimeoutError) and cut:
                    # הניסיון קוצר בגלל הזמן הכולל לשאלה - זו לא אשמת הספק, ולכן הוא לא "נח"
                    errors.append(f"{p.name}: cut short by the question time limit")
                    LAST_ERRORS[purpose] = errors
                    raise OutOfTime(purpose, errors) from exc
                overload = kind == UNAVAILABLE
                if (kind in RETRYABLE or overload and not OVERLOAD_MEMORY) and attempt < retries:
                    wait = min(max_wait, retry_after(exc) or 2 ** (attempt + 1) + random.random())
                    log.info("%s: %s %s, retrying in %.0fs", purpose, p.name, kind, wait)
                    time.sleep(wait)
                    continue
                if cooldown and kind in COOLDOWN_SECONDS:
                    rest = PATIENT_OVERLOAD_REST if overload and not OVERLOAD_MEMORY else COOLDOWN_SECONDS[kind]
                    reset = retry_after(exc)
                    if kind in (RATE_LIMIT, QUOTA) and reset:  # הספק אומר מתי הוא פנוי - לנוח בדיוק עד אז
                        rest = min(max(reset, 1), 86400)
                    # מכסה / מגבלה לדקה / מודל לא קיים: רק הצירוף מפתח+מודל נח, ואותו מודל ממשיך במפתח הבא.
                    # מפתח לא תקין: המפתח נח בכל המודלים. עומס: המודל נח 5 דקות בכל המפתחות.
                    target = (p.key if kind == AUTH and p.key else p.group if overload and OVERLOAD_MEMORY and p.group
                              else p.name)
                    if overload and OVERLOAD_MEMORY and p.group and fast_overload(exc):
                        fast_overloads[p.group] += 1
                        if fast_overloads[p.group] < FAST_OVERLOAD_LIMIT:
                            target, rest = p.name, FAST_OVERLOAD_REST  # אותו מודל - מיד במפתח הבא
                        else:
                            rest = FAST_OVERLOAD_GROUP_REST  # עומס על המודל: נח בכל המפתחות
                    _resting_until[target] = time.time() + rest
                errors.append(f"{p.name}: {kind}: {str(exc)[:200]}")
                log.warning("%s: %s failed (%s: %s), moving to next provider", purpose, p.name, kind, str(exc)[:200])
                break
    LAST_ERRORS[purpose] = errors
    raise AllProvidersFailed(purpose, errors)
