"""מנגנון הגיבוי המשותף לכל המטרות (מודל שפה, אמבדינגים, Rerank, איסוף).

run_chain מנסה ספק אחרי ספק לפי הסדר. על עומס (429) או נפילה (5xx) הוא מחכה
קצת ומנסה שוב, ואם זה לא עוזר - עובר לספק הבא. ספק שנחסם "נח" לזמן מה
(דקה על מגבלה לדקה, שעה על מכסה יומית) כדי שלא נבזבז עליו זמן בכל בקשה.
"""
import collections
import logging
import random
import re
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

RETRYABLE = {RATE_LIMIT, UNAVAILABLE}
COOLDOWN_SECONDS = {
    RATE_LIMIT: 60,
    QUOTA: 3600,
    UNAVAILABLE: 30,
    AUTH: float("inf"),
    NOT_FOUND: float("inf"),
}

_resting_until: dict[str, float] = {}

# כמה קריאות נשלחו לכל ספק בתהליך הנוכחי (כולל כאלה שנכשלו) - כדי לעקוב אחרי מכסות
CALLS: collections.Counter = collections.Counter()


class EmptyResult(Exception):
    """הספק החזיר תשובה ריקה - עוברים לבא בתור."""


class StopChain(Exception):
    """תקלה שספק אחר לא יפתור (למשל עמוד שלא קיים) - עוצרים מיד."""


class AllProvidersFailed(Exception):
    def __init__(self, purpose: str, errors: list[str]):
        self.purpose = purpose
        self.errors = errors
        super().__init__(f"{purpose}: all providers failed: " + " | ".join(errors))


@dataclass
class Provider:
    name: str                 # למשל "gemini:gemini-3.8-flash"
    call: Callable[..., Any]
    enabled: bool = True      # False כשאין מפתח ב-.env


def _status_code(exc: Exception) -> int | None:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code
    code = getattr(exc, "code", None)  # שגיאות של google-genai
    return code if isinstance(code, int) else None


def classify(exc: Exception) -> str:
    if isinstance(exc, EmptyResult):
        return EMPTY
    if isinstance(exc, (httpx.TransportError, TimeoutError, ConnectionError)):
        return UNAVAILABLE
    code = _status_code(exc)
    if code == 429:
        msg = str(exc).lower()
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


def retry_after(exc: Exception) -> float | None:
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            return float(exc.response.headers.get("retry-after", ""))
        except ValueError:
            pass
    # ג'מיני כותב את זה בהודעה: "retryDelay': '3068s'" או "Please retry in 51m8.04s" / "retry in 34.5s"
    text = str(exc)
    if match := re.search(r"retryDelay'?\"?:\s*'?\"?([\d.]+)s", text):
        return float(match.group(1))
    if match := re.search(r"retry in (?:(\d+)m)?([\d.]+)\s*s", text, re.IGNORECASE):
        return int(match.group(1) or 0) * 60 + float(match.group(2))
    return None


def run_chain(purpose: str, providers: list[Provider], *args,
              retries: int = 1, max_wait: float = 10.0, cooldown: bool = True, **kwargs):
    """מחזיר (תוצאה, שם הספק שהצליח). זורק AllProvidersFailed אם כולם נכשלו."""
    errors = []
    for p in providers:
        if not p.enabled:
            continue
        if _resting_until.get(p.name, 0) > time.time():
            errors.append(f"{p.name}: resting")
            continue
        for attempt in range(retries + 1):
            try:
                CALLS[p.name] += 1
                return p.call(*args, **kwargs), p.name
            except StopChain:
                raise
            except Exception as exc:
                kind = classify(exc)
                if kind in RETRYABLE and attempt < retries:
                    wait = min(max_wait, retry_after(exc) or 2 ** (attempt + 1) + random.random())
                    log.info("%s: %s %s, retrying in %.0fs", purpose, p.name, kind, wait)
                    time.sleep(wait)
                    continue
                if cooldown and kind in COOLDOWN_SECONDS:
                    rest = COOLDOWN_SECONDS[kind]
                    if kind == QUOTA and (reset := retry_after(exc)):  # מכסה יומית: לנוח עד האיפוס
                        rest = max(rest, min(reset, 86400))
                    _resting_until[p.name] = time.time() + rest
                errors.append(f"{p.name}: {kind}: {str(exc)[:200]}")
                log.warning("%s: %s failed (%s: %s), moving to next provider", purpose, p.name, kind, str(exc)[:200])
                break
    raise AllProvidersFailed(purpose, errors)
