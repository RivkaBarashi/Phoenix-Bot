"""שער המודלים: כל קריאה למודל שפה עוברת כאן - סדר המודלים, זמנים, ניסיון חוזר, מפסק זרם ומדדים, במקום אחד.

נתיב = רשימת מודלים לפי סדר עדיפות: ראשי, ואחריו גיבויים (config.ROUTES). מפתח Gemini ראשי, והבא בתור - רק כשהמכסה
היומית נגמרה, בתוך הקריאה (gemini_keys.py); השער רואה מודל אחד.

מפסק זרם (circuit breaker) לכל מודל:
  - סגור: המודל עובד. תקלה אחת לא משביתה אותו; FAILURES_TO_OPEN תקלות ברצף - נפתח. מכסה, מפתח לא תקין, מודל לא
    קיים, או 429 עם זמן המתנה מהספק - נפתח מיד.
  - פתוח: מדלגים עליו (ומיד מתחיל הבא בנתיב). הזמן קצר וגדל בכל פתיחה ברצף: 5 -> 10 -> 20 ... עד דקה; מכסה ומפתח
    לא תקין - 10 דקות; 429 עם Retry-After - בדיוק כמו שהספק ביקש. תמיד עם תוספת אקראית קטנה (לא מוקדם יותר).
  - חצי-פתוח: הזמן עבר - בקשה אחת (ורק אחת) עוברת כבקשת ניסיון, עם זמן קצוב קצר (PROBE_TIMEOUT), והמודל הבא בנתיב
    מתחיל איתה במקביל - כך הלקוחה לא מחכה למודל שאולי עוד למטה. הצליחה - נסגר; נכשלה - נפתח שוב, לזמן ארוך יותר.
  - כל המפסקים בנתיב פתוחים: מנסים בכל זאת את המודל שחוזר הכי מוקדם. אף בקשה לא נגמרת בלי ניסיון אחד לפחות.

בצ'אט (hedge_after): הראשי מתחיל לבד. נתקל בתקלה (שגיאה, תשובה שנפסלה, מפסק פתוח), או שעברו hedge_after שניות בלי
תשובה ממנו - כל שאר הנתיב מתחיל יחד. דרגה (tier): תשובה טובה מדרגה 0 (Gemini) מנצחת מיד; תשובה טובה מדרגה גבוהה יותר
(מודל הגיבוי מחברה אחרת) מחכה למודלים מדרגה נמוכה יותר שעוד רצים, עד prefer_until שניות מההתחלה (ולא קרוב יותר
מ-margin לסוף הזמן של השאלה). בסקריפטים (hedge_after=None) - אחד אחרי השני, לפי הסדר.

ניסיון חוזר באותו מודל: רק על תקלה מהירה (429 לדקה, 503/500, תקלת רשת), אחרי המתנה גדלה ואקראית - או בדיוק
Retry-After כשהספק אמר - ורק אם זה בתוך max_wait ונשאר זמן. timeout ו-504 לא חוזרים: הם כבר אכלו את הזמן, והמודל
הבא בנתיב כבר רץ.
"""
import collections
import contextvars
import logging
import queue
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from . import fallback
from .fallback import (AUTH, EMPTY, NOT_FOUND, QUOTA, RATE_LIMIT, UNAVAILABLE, AllProvidersFailed, OutOfTime, classify,
                       fast_overload, retry_after, time_left, with_deadline)

log = logging.getLogger("phoenix")

CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"
FAILURES_TO_OPEN = 2             # תקלות ברצף שפותחות את המפסק (תקלה בודדת - 503 על בקשה אחת - לא)
OPEN_BASE, OPEN_MAX = 5.0, 60.0  # פתוח 5 שניות, ובכל פתיחה ברצף פי 2, עד דקה
OPEN_FOR = {QUOTA: 600.0, AUTH: 600.0, NOT_FOUND: 600.0}  # ואז בקשת ניסיון: מפתח ששודרג לתשלום חוזר תוך 10 דקות
RATE_LIMIT_MAX = 300.0           # Retry-After ארוך מזה - לא מאמינים לו (Gemini אמר 22 שעות, והמכסה התאפסה קודם)
JITTER = 0.2                     # עד 20% יותר זמן פתוח - כדי שבקשות הניסיון של תהליכים שונים לא יגיעו יחד
PROBE_TIMEOUT = 8.0
RETRY_BASE = 0.5                 # המתנה לפני ניסיון חוזר: 0.25-0.5 ש', ואחר כך פי 2


def open_seconds(kind: str, wait: float | None, opened: int) -> float:
    """כמה זמן המפסק נשאר פתוח. opened: כמה פעמים נפתח ברצף, בלי הצלחה ביניהן."""
    if kind in OPEN_FOR:
        base = min(wait, OPEN_FOR[kind]) if wait else OPEN_FOR[kind]
    elif kind == RATE_LIMIT and wait:
        base = min(wait, RATE_LIMIT_MAX)
    else:
        base = min(OPEN_BASE * 2 ** (opened - 1), OPEN_MAX)
    return base * random.uniform(1.0, 1.0 + JITTER)


class Breaker:
    """מפסק זרם למודל אחד. כמה threads (כמה משתמשים) משתמשים בו יחד - הכול תחת נעילה."""

    def __init__(self, name: str):
        self.name = name
        self.state = CLOSED
        self.failures = 0     # תקלות ברצף, במצב סגור
        self.opened = 0       # פתיחות ברצף בלי הצלחה ביניהן - קובע את אורך הפתיחה הבאה
        self.until = 0.0      # פתוח עד (time.monotonic)
        self.probing = False  # בקשת הניסיון בדרך
        self.reason = ""      # סוג התקלה האחרונה
        self._lock = threading.Lock()

    def admit(self) -> str | None:
        """"closed" - בקשה רגילה; "probe" - בקשת הניסיון (אחת בכל פעם); None - לדלג על המודל."""
        with self._lock:
            if self.state == CLOSED:
                return "closed"
            if self.probing or time.monotonic() < self.until:
                return None
            self.state, self.probing = HALF_OPEN, True
            return "probe"

    def force(self) -> str:
        """כל המודלים בנתיב פתוחים: הבקשה עוברת בכל זאת. במפסק לא סגור - כבקשת ניסיון."""
        with self._lock:
            if self.state == CLOSED:
                return "closed"
            self.state, self.probing = HALF_OPEN, True
            return "probe"

    def succeeded(self) -> None:
        with self._lock:
            if self.state != CLOSED:
                log.info("circuit %s: closed again", self.name)
            self.state, self.failures, self.opened, self.probing, self.reason = CLOSED, 0, 0, False, ""

    def failed(self, kind: str, wait: float | None = None, probe: bool = False) -> bool:
        """תקלה. מחזיר True אם המפסק נפתח עכשיו. תקלה מאוחרת של בקשה ישנה (המפסק כבר פתוח, או שבקשת ניסיון
        אחרת היא שמחליטה) לא מאריכה ולא משנה כלום."""
        with self._lock:
            if probe:
                self.probing = False
            elif self.state != CLOSED:
                return False
            self.reason = kind
            self.failures += 1
            hard = kind in OPEN_FOR or (kind == RATE_LIMIT and bool(wait))
            if not (probe or hard or self.failures >= FAILURES_TO_OPEN):
                return False
            self.opened += 1
            self.state, self.failures = OPEN, 0
            self.until = time.monotonic() + open_seconds(kind, wait, self.opened)
            log.warning("circuit %s: open for %.0fs (%s)", self.name, self.until - time.monotonic(), kind)
            return True

    def release(self) -> None:
        """בקשת הניסיון נחתכה בגלל הזמן של השאלה - לא אשמת המודל. בקשה אחרת תנסה, בלי לחכות."""
        with self._lock:
            self.probing = False

    def snapshot(self) -> dict:
        with self._lock:
            return {"state": self.state, "reason": self.reason, "failures": self.failures,
                    "open_for": max(self.until - time.monotonic(), 0.0) if self.state != CLOSED else 0.0}


_breakers: dict[str, Breaker] = {}
_breakers_lock = threading.Lock()


def breaker(name: str) -> Breaker:
    with _breakers_lock:
        if name not in _breakers:
            _breakers[name] = Breaker(name)
        return _breakers[name]


def circuits() -> dict[str, dict]:
    """מצב כל המפסקים שאינם סגורים - לדוחות ולמצב מפתחת."""
    with _breakers_lock:
        found = list(_breakers.values())
    return {b.name: snap for b in found if (snap := b.snapshot())["state"] != CLOSED}


# מדדים לכל מודל, בתהליך הנוכחי: ניסיונות, הצלחות, תקלות לפי סוג, וזמני התשובה האחרונים
STATS: dict[str, dict] = collections.defaultdict(
    lambda: {"calls": 0, "ok": 0, "failed": collections.Counter(), "seconds": collections.deque(maxlen=100)})
_stats_lock = threading.Lock()


def _count(name: str, kind: str | None = None, seconds: float | None = None, start: bool = False) -> None:
    with _stats_lock:
        row = STATS[name]
        if start:
            row["calls"] += 1
            fallback.CALLS[name] += 1  # usage.py סופר משם
        elif kind:
            row["failed"][kind] += 1
            fallback.FAILED[name] += 1
        else:
            row["ok"] += 1
            row["seconds"].append(seconds)


@dataclass(frozen=True)
class Policy:
    hedge_after: float | None = 3.0  # None - בזה אחר זה (סקריפטים); אחרת - הבא בנתיב מתחיל אחרי כמה שניות בלי תשובה
    prefer_until: float = 0.0        # תשובה טובה של גיבוי מחכה למודל מוקדם יותר בנתיב עד כמה שניות מההתחלה
    margin: float = 0.5              # ...ולא קרוב יותר מזה לסוף הזמן של השאלה
    timeout: float = 15.0            # לכל ניסיון (ובתוך הזמן שנשאר לשאלה)
    retries: int = 1                 # ניסיון חוזר באותו מודל - רק על תקלה מהירה
    max_wait: float = 2.0            # המתנה ארוכה מזו לפני ניסיון חוזר - לא מחכים (המודל הבא כבר רץ)


@dataclass
class Lane:
    name: str                   # למשל "gemini:gemini-3.6-flash#1"
    call: Callable[[], Any]     # ניסיון אחד; חריגה = תקלה
    tier: int = 0               # 0 - המודלים המועדפים; 1 - גיבוי שתשובתו מחכה להם (עד prefer_until)


@dataclass
class Pick:
    value: Any
    name: str
    rank: int = 0
    started: list[str] = field(default_factory=list)  # המודלים שנשאלו בבקשה הזו, לפי הסדר


def _backoff(attempt: int, max_wait: float) -> float:
    return min(RETRY_BASE * 2 ** attempt, max_wait) * random.uniform(0.5, 1.0)


def _attempts(lane: Lane, i: int, mode: str, policy: Policy, inbox: queue.Queue, stop: threading.Event) -> None:
    """רץ ב-thread משלו: ניסיון אחד או יותר באותו מודל. מדווח ל-inbox: ("trouble", ...) על תקלה שאחריה עוד ניסיון
    (כדי שהמודל הבא יתחיל כבר עכשיו), ובסוף ("ok", תוצאה) או ("error", (חריגה, סוג))."""
    b, probe = breaker(lane.name), mode == "probe"
    tries = 1 if probe else 1 + max(policy.retries, 0)
    for attempt in range(tries):
        left = time_left()
        if left is not None and left < fallback.MIN_ATTEMPT_SECONDS:
            if probe:
                b.release()
            inbox.put(("error", i, (OutOfTime(lane.name, ["out of time"]), "out_of_time")))
            return
        limit = min(policy.timeout, PROBE_TIMEOUT) if probe else policy.timeout
        cut = left is not None and left < limit  # הניסיון מוגבל בזמן שנשאר לשאלה, לא בזמן של המודל
        if cut:
            limit = left
        _count(lane.name, start=True)
        began = time.monotonic()
        try:
            value = with_deadline(lane.call, limit)
        except Exception as exc:  # noqa: BLE001 - כל תקלה מסווגת ומדווחת
            kind = classify(exc)
            _count(lane.name, kind)
            if isinstance(exc, TimeoutError) and cut:  # נגמר הזמן של השאלה - המפסק לא נוגע
                if probe:
                    b.release()
                inbox.put(("error", i, (OutOfTime(lane.name, [str(exc)]), "out_of_time")))
                return
            if kind == EMPTY:  # המודל ענה, אבל לא משהו שאפשר להשתמש בו - הוא זמין; התשובה לא
                b.succeeded()
                inbox.put(("error", i, (exc, kind)))
                return
            wait = retry_after(exc)
            opened = b.failed(kind, wait, probe=probe)
            pause = wait if wait else _backoff(attempt, policy.max_wait)
            left = time_left()
            again = (attempt + 1 < tries and not opened and not stop.is_set()
                     and kind in (RATE_LIMIT, UNAVAILABLE) and fast_overload(exc) and pause <= policy.max_wait
                     and (left is None or left - pause >= fallback.MIN_ATTEMPT_SECONDS))
            log.warning("%s: %s (%s, %.1fs)%s", lane.name, kind, str(exc)[:160], time.monotonic() - began,
                        f", retrying in {pause:.1f}s" if again else "")
            if not again:
                inbox.put(("error", i, (exc, kind)))
                return
            inbox.put(("trouble", i, kind))
            if stop.wait(pause):  # הבקשה כבר הסתיימה (מודל אחר ענה) - בלי ניסיון נוסף
                inbox.put(("error", i, (exc, kind)))
                return
            continue
        b.succeeded()
        _count(lane.name, seconds=time.monotonic() - began)
        inbox.put(("ok", i, value))
        return


def run(purpose: str, lanes: list[Lane], policy: Policy = Policy(),
        rank: Callable[[Any], int] | None = None) -> Pick:
    """שולח את הבקשה לפי הנתיב ומחזיר את התשובה שנבחרה. rank(תוצאה) -> 0: טובה; מספר גבוה יותר: פחות טובה (נפסלה
    בבדיקות) - ממשיכים לחכות למודלים שעוד רצים ומפעילים את הבא בנתיב; בסוף - הטובה ביותר שיש. rank נקרא כאן, ב-thread
    של הקורא (חיווי השלבים של Streamlit עובד רק בו). אין אף תוצאה: AllProvidersFailed, או OutOfTime כשנגמר הזמן."""
    if not lanes:
        raise AllProvidersFailed(purpose, ["no models in the route"])
    inbox: queue.Queue = queue.Queue()
    stop = threading.Event()
    hedged = policy.hedge_after is not None
    began = time.monotonic()
    started_at: dict[int, float] = {}
    running: set[int] = set()
    released: set[int] = set()  # המודל הזה נתקל בתקלה / סיים בלי תשובה טובה / דולג - הבא בנתיב רשאי להתחיל
    errors: list[str] = []
    results: list[tuple[int, int, Any]] = []  # (דירוג, מספר המודל בנתיב, תוצאה) - כל מה שחזר
    good: tuple[int, Any] | None = None       # התשובה הטובה הראשונה מהדרגה המועדפת ביותר שענתה
    nxt, tried, out_of_time = 0, False, False

    def launch(i: int, mode: str) -> None:
        started_at[i] = time.monotonic()
        running.add(i)
        if mode == "probe" and hedged:
            released.add(i)  # בקשת ניסיון: המודל הבא מתחיל איתה במקביל
        threading.Thread(target=contextvars.copy_context().run, daemon=True,
                         args=(_attempts, lanes[i], i, mode, policy, inbox, stop)).start()

    def may_start(i: int) -> bool:
        if i == 0:
            return True
        if not hedged:  # בזה אחר זה
            return i - 1 in released
        # הראשי נתקל בתקלה / דולג / עברו hedge_after שניות בלי תשובה ממנו - כל השאר מתחילים יחד
        return 0 in released or (0 in running and time.monotonic() - started_at[0] >= policy.hedge_after)

    def pick(i: int, value: Any, r: int) -> Pick:
        return Pick(value, lanes[i].name, r, [lanes[j].name for j in sorted(started_at)])

    try:
        while True:
            left = time_left()
            while good is None and nxt < len(lanes) and may_start(nxt):
                if left is not None and left < fallback.MIN_ATTEMPT_SECONDS:
                    out_of_time, nxt = True, len(lanes)  # אין זמן לניסיון נוסף
                    break
                b = breaker(lanes[nxt].name)
                if mode := b.admit():
                    launch(nxt, mode)
                    tried = True
                else:
                    snap = b.snapshot()
                    errors.append(f"{lanes[nxt].name}: circuit open ({snap['reason']}, {snap['open_for']:.0f}s)")
                    released.add(nxt)
                nxt += 1
            if not tried and nxt >= len(lanes) and not out_of_time:
                # כל המפסקים פתוחים - לא מוותרים מראש: המודל שחוזר הכי מוקדם
                i = min(range(len(lanes)), key=lambda j: breaker(lanes[j].name).until)
                log.warning("%s: every circuit is open - trying %s anyway", purpose, lanes[i].name)
                launch(i, breaker(lanes[i].name).force())
                tried = True
            now = time.monotonic()
            if good is not None:
                earlier = any(lanes[j].tier < lanes[good[0]].tier for j in running)
                if (not earlier or now - began >= policy.prefer_until
                        or (left is not None and left <= policy.margin)):
                    return pick(good[0], good[1], 0)
            if not running and nxt >= len(lanes):
                break
            if left is not None and left <= 0.05:
                out_of_time = True
                break
            waits = [left] if left is not None else []
            if good is None and nxt < len(lanes) and hedged and 0 in running and 0 not in released:
                waits.append(started_at[0] + policy.hedge_after - now)
            if good is not None:
                waits.append(began + policy.prefer_until - now)
                if left is not None:
                    waits.append(left - policy.margin)
            try:
                kind, i, payload = inbox.get(timeout=max(min(waits), 0.01) if waits else None)
            except queue.Empty:
                continue
            if kind == "trouble":
                if hedged:
                    released.add(i)
                continue
            running.discard(i)
            if kind == "error":
                exc, why = payload
                released.add(i)
                out_of_time = out_of_time or why == "out_of_time"
                errors.append(f"{lanes[i].name}: {why}: {str(exc)[:200]}")
                continue
            r = rank(payload) if rank else 0
            results.append((r, i, payload))
            if r == 0:
                if good is None or lanes[i].tier < lanes[good[0]].tier:
                    good = (i, payload)
            else:
                released.add(i)
    finally:
        stop.set()  # ניסיונות חוזרים שעוד מחכים - לא נשלחים
    if good is not None:
        return pick(good[0], good[1], 0)
    if results:  # אין תשובה טובה - הפחות גרועה (ובשוויון - מהדרגה המועדפת, ואז המוקדם בנתיב)
        r, i, value = min(results, key=lambda x: (x[0], lanes[x[1]].tier, x[1]))
        return pick(i, value, r)
    fallback.LAST_ERRORS[purpose] = errors
    raise (OutOfTime if out_of_time else AllProvidersFailed)(purpose, errors)
