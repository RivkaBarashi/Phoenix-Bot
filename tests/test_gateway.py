"""בדיקות לשער המודלים (gateway.py) ולמפסק הזרם - בלי API: ספקים מזויפים בלבד.

מכסות את התרחישים שבאפיון (א1, א3): "סערה" (503 מכל המודלים), חזרה הדרגתית, בקשת ניסיון אחת בכל פעם, סיווג 429
(לדקה מול יומית), זמן כולל לשאלה, וכלל הבסיס - אף בקשה לא נגמרת בלי ניסיון אחד לפחות.
"""
import threading
import time

import httpx
import pytest

from phoenix import config, fallback, gateway, llm
from phoenix.fallback import (AUTH, EMPTY, ERROR, NOT_FOUND, QUOTA, RATE_LIMIT, UNAVAILABLE, AllProvidersFailed,
                              EmptyResult, OutOfTime)
from phoenix.gateway import CLOSED, HALF_OPEN, OPEN, Breaker, Lane, Policy

REAL_CHAT = llm.chat  # לפני ש-conftest חוסם אותו


class FakeApiError(Exception):
    """שגיאה כמו של google-genai: קוד HTTP בשדה code."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


# ההודעות האמיתיות (מהיומנים ומהתיעוד של הספקים)
GEMINI_503 = ("503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is currently experiencing high demand. "
              "Spikes in demand are usually temporary. Please try again later.', 'status': 'UNAVAILABLE'}}")
GEMINI_504 = ("504 DEADLINE_EXCEEDED. {'error': {'code': 504, 'message': 'Deadline expired before operation could "
              "complete.', 'status': 'DEADLINE_EXCEEDED'}}")
GEMINI_DAY = ("429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check "
              "your plan and billing details.', 'details': [{'violations': [{'quotaMetric': 'generativelanguage.googleapis"
              ".com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'}]"
              "}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '81000s'}]}}")
GEMINI_MINUTE = ("429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please "
                 "check your plan and billing details.', 'details': [{'violations': [{'quotaId': "
                 "'GenerateRequestsPerMinutePerProjectPerModel-FreeTier'}]}, {'retryDelay': '17s'}]}}")
GEMINI_BAD_KEY = ("400 INVALID_ARGUMENT. {'error': {'code': 400, 'message': 'API key not valid. Please pass a valid API "
                  "key.', 'status': 'INVALID_ARGUMENT', 'details': [{'reason': 'API_KEY_INVALID'}]}}")
GEMINI_DENIED = "403 PERMISSION_DENIED. Your project has been denied access. Please contact support."


def http_error(code: int, body: str, headers: dict | None = None) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://api.example.com/v1/chat")
    response = httpx.Response(code, text=body, headers=headers or {}, request=request)
    return httpx.HTTPStatusError(f"{code} error", request=request, response=response)


# ---------------------------------------------------------------- סיווג תקלות: כל הודעה שראינו
@pytest.mark.parametrize("exc, kind, wait", [
    (FakeApiError(503, GEMINI_503), UNAVAILABLE, None),
    (FakeApiError(504, GEMINI_504), UNAVAILABLE, None),
    (FakeApiError(429, GEMINI_DAY), QUOTA, 81000.0),          # מכסה יומית - לא "מגבלה לדקה"
    (FakeApiError(429, GEMINI_MINUTE), RATE_LIMIT, 17.0),     # אותה פתיחה ("exceeded your current quota") - אבל לדקה
    (FakeApiError(429, "429 RESOURCE_EXHAUSTED. Resource has been exhausted (e.g. check quota)."), RATE_LIMIT, None),
    (FakeApiError(400, GEMINI_BAD_KEY), AUTH, None),          # מפתח שגוי ב-Gemini הוא 400, לא 401
    (FakeApiError(400, "400 INVALID_ARGUMENT. Thinking level MINIMAL is not supported for this model."), ERROR, None),
    (FakeApiError(403, GEMINI_DENIED), AUTH, None),
    (FakeApiError(404, "404 NOT_FOUND. models/gemini-9-flash is not found"), NOT_FOUND, None),
    (http_error(429, '{"error": {"message": "Rate limit reached for model `openai/gpt-oss-120b` on tokens per day '
                     '(TPD): Limit 200000, Used 199500. Please try again in 34m37.9s."}}'), QUOTA, 2077.9),
    (http_error(429, '{"error": {"message": "Rate limit reached on tokens per minute (TPM): Limit 8000. Please try '
                     'again in 7.5s."}}'), RATE_LIMIT, 7.5),
    (http_error(429, '{"message": "You are using a Trial key, which is limited to 20 API calls / minute."}'),
     RATE_LIMIT, None),
    (http_error(429, '{"message": "You are using a Trial key, which is limited to 1000 API calls / month."}'),
     QUOTA, None),
    (http_error(429, '{"error": {"message": "Rate limit exceeded: free-models-per-day."}}'), QUOTA, None),
    (http_error(429, "slow down", {"retry-after": "12"}), RATE_LIMIT, 12.0),
    (http_error(402, "payment required"), QUOTA, None),
    (http_error(500, "internal error"), UNAVAILABLE, None),
    (httpx.ConnectError("connection reset"), UNAVAILABLE, None),
    (TimeoutError("no answer within 15s"), UNAVAILABLE, None),
    (EmptyResult("unusable response"), EMPTY, None),
])
def test_every_error_we_saw_is_classified(exc, kind, wait):
    assert fallback.classify(exc) == kind
    assert fallback.retry_after(exc) == wait


@pytest.mark.parametrize("exc, fast", [
    (FakeApiError(503, GEMINI_503), True), (httpx.ConnectError("reset"), True), (FakeApiError(429, GEMINI_MINUTE), True),
    (FakeApiError(504, GEMINI_504), False), (TimeoutError("15s"), False),  # כבר אכלו את הזמן - בלי ניסיון חוזר
])
def test_only_fast_errors_are_retried(exc, fast):
    assert fallback.fast_overload(exc) is fast


# ---------------------------------------------------------------- מפסק הזרם
def test_one_failure_does_not_open_the_circuit_two_in_a_row_do():
    b = Breaker("m")
    assert b.failed(UNAVAILABLE) is False and b.admit() == "closed"   # 503 בודד - עומס על בקשה אחת
    b.succeeded()
    assert b.failed(UNAVAILABLE) is False                              # הצלחה ביניהם - הספירה מתחילה מחדש
    assert b.failed(UNAVAILABLE) is True and b.state == OPEN and b.admit() is None


def test_open_time_grows_and_is_never_shorter_than_the_base(monkeypatch):
    monkeypatch.setattr(gateway.random, "uniform", lambda a, b: b)  # הפיזור המקסימלי
    assert [round(gateway.open_seconds(UNAVAILABLE, None, n), 1) for n in (1, 2, 3, 4, 5, 6)] == \
           [6.0, 12.0, 24.0, 48.0, 72.0, 72.0]                      # 5 -> 10 -> 20 -> 40 -> 60 (+20%), עד דקה
    monkeypatch.setattr(gateway.random, "uniform", lambda a, b: a)  # בלי פיזור - אף פעם לא מוקדם מהבסיס
    assert gateway.open_seconds(UNAVAILABLE, None, 1) == gateway.OPEN_BASE


@pytest.mark.parametrize("kind, wait, low, high", [
    (QUOTA, 81000.0, 600, 720),       # Gemini אמר 22.5 שעות - בודקים שוב אחרי 10 דקות (המכסה התאפסה קודם; מפתח שודרג)
    (QUOTA, None, 600, 720),
    (AUTH, None, 600, 720),
    (NOT_FOUND, None, 600, 720),
    (RATE_LIMIT, 17.0, 17, 20.4),     # בדיוק כמו שהספק ביקש (ולא פחות)
    (RATE_LIMIT, 86400.0, 300, 360),  # Retry-After לא סביר - עד 5 דקות
])
def test_hard_errors_open_at_once_for_the_right_time(kind, wait, low, high):
    b = Breaker("m")
    assert b.failed(kind, wait) is True
    assert low <= b.snapshot()["open_for"] <= high


def test_a_per_minute_limit_without_a_wait_is_like_a_503():
    """Cohere Trial (20 לדקה) בלי Retry-After: 429 אחד לא משבית את מודל הגיבוי (21/10 הוא נח דקה שלמה, וכל שאלה באותה
    דקה נשארה בלי תשובה)."""
    b = Breaker("cohere:x")
    assert b.failed(RATE_LIMIT) is False and b.admit() == "closed"
    assert b.failed(RATE_LIMIT) is True and b.snapshot()["open_for"] <= gateway.OPEN_BASE * 1.2


def test_after_the_open_time_exactly_one_probe_passes_even_under_load(monkeypatch):
    monkeypatch.setattr(gateway, "OPEN_BASE", 0.05)
    b = Breaker("m")
    b.failed(UNAVAILABLE), b.failed(UNAVAILABLE)
    time.sleep(0.08)
    got, start = [], threading.Barrier(20)

    def ask():
        start.wait()
        got.append(b.admit())
    threads = [threading.Thread(target=ask) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert got.count("probe") == 1 and got.count(None) == 19 and b.state == HALF_OPEN


def test_probe_success_closes_probe_failure_reopens_for_longer(monkeypatch):
    monkeypatch.setattr(gateway, "OPEN_BASE", 0.05)
    monkeypatch.setattr(gateway.random, "uniform", lambda a, b: a)
    b = Breaker("m")
    b.failed(UNAVAILABLE), b.failed(UNAVAILABLE)                      # פתוח 0.05
    time.sleep(0.06)
    assert b.admit() == "probe" and b.failed(UNAVAILABLE, probe=True) is True
    assert 0.08 < b.snapshot()["open_for"] <= 0.1                    # פי 2
    time.sleep(0.11)
    assert b.admit() == "probe"
    b.succeeded()
    assert b.state == CLOSED and b.admit() == "closed" and b.opened == 0


def test_a_late_failure_of_an_old_request_does_not_extend_the_open_time():
    b = Breaker("m")
    b.failed(UNAVAILABLE), b.failed(UNAVAILABLE)
    until = b.until
    assert b.failed(UNAVAILABLE) is False and b.until == until


def test_a_probe_cut_by_the_question_time_lets_the_next_request_probe(monkeypatch):
    monkeypatch.setattr(gateway, "OPEN_BASE", 0.01)
    b = Breaker("m")
    b.failed(UNAVAILABLE), b.failed(UNAVAILABLE)
    time.sleep(0.02)
    assert b.admit() == "probe" and b.admit() is None
    b.release()
    assert b.admit() == "probe"


# ---------------------------------------------------------------- הנתיב: ספקים מזויפים
class Model:
    """מודל מזויף: plan - רשימת התנהגויות לפי הסדר (האחרונה חוזרת): ערך, חריגה, או (השהיה, ערך/חריגה)."""

    def __init__(self, name: str, *plan):
        self.name, self.plan, self.calls, self.active, self.peak = name, list(plan), 0, 0, 0
        self._lock = threading.Lock()

    def __call__(self):
        with self._lock:
            step = self.plan[min(self.calls, len(self.plan) - 1)]
            self.calls += 1
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            delay, result = step if isinstance(step, tuple) else (0.0, step)
            time.sleep(delay)
            if isinstance(result, Exception):
                raise result
            return result
        finally:
            with self._lock:
                self.active -= 1

    def lane(self, tier: int = 0) -> Lane:
        return Lane(self.name, self, tier)


BUSY = FakeApiError(503, GEMINI_503)
FAST = Policy(hedge_after=0.3, prefer_until=0.0, retries=1, max_wait=0.2)


@pytest.fixture
def quick(monkeypatch):
    """זמנים קצרים לבדיקה: מפסק פתוח 0.2 ש', ניסיון חוזר אחרי עד 0.05 ש'."""
    monkeypatch.setattr(gateway, "OPEN_BASE", 0.2)
    monkeypatch.setattr(gateway, "RETRY_BASE", 0.05)
    monkeypatch.setattr(fallback, "MIN_ATTEMPT_SECONDS", 0.2)


def test_storm_every_model_busy_every_request_still_tries(quick):
    """"סערה": 503 מכל המודלים. אף בקשה לא נגמרת בלי ניסיון - גם כשכל המפסקים כבר פתוחים (22/10 16:41: "לא הצלחתי
    לאמת" אחרי 0 שניות). ובזמן שהמפסקים פתוחים - ניסיון אחד לבקשה, לא הפצצה."""
    a, b = Model("a", BUSY), Model("b", BUSY)
    for _ in range(3):
        with pytest.raises(AllProvidersFailed):
            gateway.run("t", [a.lane(), b.lane(1)], FAST)
    assert gateway.breaker("a").state == OPEN and gateway.breaker("b").state == OPEN
    before = a.calls + b.calls
    with pytest.raises(AllProvidersFailed) as failed:
        gateway.run("t", [a.lane(), b.lane(1)], FAST)
    assert a.calls + b.calls == before + 1                      # כולם פתוחים - בדיוק ניסיון אחד (לא אפס)
    assert "circuit open" in str(failed.value)


def test_storm_then_gradual_recovery(quick):
    """המודל חוזר: אחרי זמן הפתיחה בקשת ניסיון אחת מגלה את זה, המפסק נסגר, והבקשות הבאות רגילות."""
    a = Model("a", BUSY, BUSY, "ok")
    with pytest.raises(AllProvidersFailed):
        gateway.run("t", [a.lane()], Policy(hedge_after=None, retries=1, max_wait=0.1))  # 503 + ניסיון חוזר - נפתח
    assert gateway.breaker("a").state == OPEN
    assert gateway.run("t", [a.lane()], FAST).value == "ok"     # כולם פתוחים - ניסיון בכל זאת, והוא עבר
    assert gateway.breaker("a").state == CLOSED
    assert gateway.run("t", [a.lane()], FAST).value == "ok"


def test_the_backup_answers_the_moment_the_primary_fails(quick):
    a, b = Model("a", BUSY), Model("b", "backup")
    start = time.perf_counter()
    picked = gateway.run("t", [a.lane(), b.lane(1)], Policy(hedge_after=3.0, prefer_until=12.0, retries=0))
    assert picked.value == "backup" and time.perf_counter() - start < 0.5  # לא מחכים 3 שניות
    assert picked.started == ["a", "b"]


def test_an_open_primary_is_skipped_without_waiting(quick):
    gateway.breaker("a").failed(QUOTA)
    a, b = Model("a", "primary"), Model("b", "backup")
    start = time.perf_counter()
    assert gateway.run("t", [a.lane(), b.lane(1)], Policy(hedge_after=3.0)).value == "backup"
    assert a.calls == 0 and time.perf_counter() - start < 0.5


def test_the_probe_runs_alongside_the_backup_so_the_customer_does_not_wait(quick, monkeypatch):
    """בקשת הניסיון מגיעה מלקוחה אמיתית: אם המודל עדיין למטה (תקוע), הגיבוי כבר רץ איתה - בלי לחכות hedge_after."""
    monkeypatch.setattr(gateway, "PROBE_TIMEOUT", 0.5)
    b0 = gateway.breaker("a")
    b0.failed(UNAVAILABLE), b0.failed(UNAVAILABLE)
    time.sleep(0.25)                                             # עבר זמן הפתיחה - הבקשה הבאה היא בקשת ניסיון
    a, b = Model("a", (5.0, "late")), Model("b", "backup")
    start = time.perf_counter()
    assert gateway.run("t", [a.lane(), b.lane(1)], Policy(hedge_after=3.0, prefer_until=0.0)).value == "backup"
    assert time.perf_counter() - start < 0.5 and a.calls == 1
    time.sleep(0.6)                                              # בקשת הניסיון נחתכה ב-PROBE_TIMEOUT ונרשמה כתקלה
    assert b0.state == OPEN


def test_many_users_during_an_outage_send_one_probe_at_a_time(quick, monkeypatch):
    monkeypatch.setattr(gateway, "PROBE_TIMEOUT", 1.0)
    b0 = gateway.breaker("a")
    b0.failed(UNAVAILABLE), b0.failed(UNAVAILABLE)
    time.sleep(0.25)
    a, b = Model("a", (0.4, BUSY)), Model("b", "backup")
    results = []
    threads = [threading.Thread(target=lambda: results.append(
        gateway.run("t", [a.lane(), b.lane(1)], Policy(hedge_after=3.0)).value)) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == ["backup"] * 10 and a.calls == 1 and a.peak == 1  # עשרה משתמשים - בקשת ניסיון אחת


def test_retry_on_the_same_model_after_a_fast_503(quick):
    a = Model("a", BUSY, "ok")
    picked = gateway.run("t", [a.lane()], Policy(hedge_after=None, retries=1, max_wait=0.2))
    assert picked.value == "ok" and a.calls == 2 and gateway.breaker("a").state == CLOSED


@pytest.mark.parametrize("error", [FakeApiError(504, GEMINI_504), TimeoutError("15s"),
                                   http_error(429, "slow down", {"retry-after": "30"}),  # מבקש לחכות יותר מ-max_wait
                                   FakeApiError(429, GEMINI_DAY), FakeApiError(400, GEMINI_BAD_KEY)])
def test_no_retry_when_it_cannot_help(quick, error):
    a, b = Model("a", error, "ok"), Model("b", "backup")
    assert gateway.run("t", [a.lane(), b.lane(1)], Policy(hedge_after=None, retries=1, max_wait=2.0)).value == "backup"
    assert a.calls == 1


def test_no_retry_after_the_request_is_over(quick, monkeypatch):
    """הגיבוי ענה בזמן שהראשי חיכה לניסיון חוזר - הניסיון החוזר לא נשלח (לא שורפים מכסה על בקשה שהסתיימה)."""
    monkeypatch.setattr(gateway, "RETRY_BASE", 0.4)
    a, b = Model("a", BUSY, "ok"), Model("b", "backup")
    assert gateway.run("t", [a.lane(), b.lane(1)], Policy(hedge_after=3.0, retries=1, max_wait=1.0)).value == "backup"
    time.sleep(0.5)
    assert a.calls == 1


def test_a_timeout_does_not_take_the_backup_out(quick):
    """21/10: Cohere איטי פעם אחת (timeout) נח 5 דקות, והשאלות הבאות נשארו בלי גיבוי. עכשיו - תקלה אחת לא פותחת."""
    b = Model("cohere:x", (0.5, "late"))
    with pytest.raises(AllProvidersFailed):
        gateway.run("t", [b.lane()], Policy(hedge_after=None, timeout=0.1))
    assert gateway.breaker("cohere:x").state == CLOSED


def test_the_question_time_limit_is_not_the_models_fault(quick):
    a = Model("a", (2.0, "late"))
    start = time.perf_counter()
    with fallback.time_budget(0.5):
        with pytest.raises(OutOfTime):
            gateway.run("t", [a.lane()], Policy(hedge_after=3.0))
    assert time.perf_counter() - start < 0.8 and gateway.breaker("a").state == CLOSED
    assert gateway.breaker("a").failures == 0


def test_no_attempt_when_almost_no_time_is_left():
    a = Model("a", "ok")
    with fallback.time_budget(1.0):  # פחות מ-MIN_ATTEMPT_SECONDS (3)
        with pytest.raises(OutOfTime):
            gateway.run("t", [a.lane()], Policy())
    assert a.calls == 0


def test_an_unusable_answer_moves_on_but_the_model_stays_available(quick):
    a, b = Model("a", EmptyResult("bad JSON")), Model("b", "ok")
    assert gateway.run("t", [a.lane(), b.lane()], Policy(hedge_after=None)).value == "ok"
    assert gateway.breaker("a").state == CLOSED and gateway.breaker("a").failures == 0


def test_a_rejected_answer_starts_the_backup_and_the_best_answer_wins(quick):
    a, b = Model("a", "bad"), Model("b", "good")
    rank = {"bad": 2, "good": 0}.get
    assert gateway.run("t", [a.lane(), b.lane(1)], Policy(hedge_after=3.0), rank=rank).value == "good"
    a2, b2 = Model("a2", "bad"), Model("b2", "worse")
    picked = gateway.run("t", [a2.lane(), b2.lane(1)], Policy(hedge_after=3.0), rank={"bad": 1, "worse": 2}.get)
    assert (picked.value, picked.rank) == ("bad", 1)              # אין טובה - הפחות גרועה


def test_the_backup_waits_for_gemini_until_prefer_until(quick):
    a, b = Model("a", (0.3, "gemini")), Model("b", "backup")
    assert gateway.run("t", [a.lane(), b.lane(1)], Policy(hedge_after=0.0, prefer_until=1.0)).value == "gemini"
    a, b = Model("a2", (3.0, "gemini")), Model("b2", "backup")
    start = time.perf_counter()
    assert gateway.run("t", [a.lane(), b.lane(1)], Policy(hedge_after=0.0, prefer_until=0.3)).value == "backup"
    assert time.perf_counter() - start < 0.6


def test_the_backup_answer_is_shown_before_the_question_time_runs_out(quick):
    a, b = Model("a", (5.0, "gemini")), Model("b", "backup")
    start = time.perf_counter()
    with fallback.time_budget(1.0):
        picked = gateway.run("t", [a.lane(), b.lane(1)], Policy(hedge_after=0.0, prefer_until=12.0, margin=0.4))
    assert picked.value == "backup" and time.perf_counter() - start < 0.8


def test_gemini_answers_of_the_same_tier_win_at_once(quick):
    """3.8 תקוע, 3.6 ענה - לא מחכים ל-3.8 (אותה דרגה)."""
    strong, second = Model("3.8", (3.0, "3.8")), Model("3.6", (0.05, "3.6"))
    start = time.perf_counter()
    assert gateway.run("t", [strong.lane(), second.lane()], Policy(hedge_after=0.2, prefer_until=6.0)).value == "3.6"
    assert time.perf_counter() - start < 0.6


def test_hedging_starts_the_rest_of_the_route_together(quick):
    first, second, backup = Model("1", (1.0, "x")), Model("2", (1.0, "y")), Model("3", (1.0, "z"))
    gateway.run("t", [first.lane(), second.lane(), backup.lane(1)], Policy(hedge_after=0.1, prefer_until=0.0))
    assert second.calls == backup.calls == 1


def test_scripts_run_one_model_at_a_time(quick):
    a, b = Model("a", (0.2, BUSY)), Model("b", (0.1, "ok"))
    assert gateway.run("t", [a.lane(), b.lane(1)], Policy(hedge_after=None, retries=0)).value == "ok"
    assert a.peak == 1 and b.peak == 1


def test_an_empty_route_fails_clearly():
    with pytest.raises(AllProvidersFailed, match="no models"):
        gateway.run("t", [], Policy())


def test_stats_count_every_attempt(quick):
    a = Model("stats:a", BUSY, "ok")
    gateway.run("t", [a.lane()], Policy(hedge_after=None, retries=1, max_wait=0.2))
    row = gateway.STATS["stats:a"]
    assert row["calls"] >= 2 and row["ok"] >= 1 and row["failed"][UNAVAILABLE] >= 1
    assert fallback.CALLS["stats:a"] >= 2


# ---------------------------------------------------------------- llm.chat דרך השער
def providers(monkeypatch, plan: dict) -> list:
    """llm.PROVIDERS מזויפים. plan: שם -> רשימת התנהגויות (כמו Model). מחזיר את רשימת הקריאות (שם, חשיבה נמוכה, thread)."""
    calls = []

    def make(name, steps):
        model = Model(name, *steps)

        def call(messages, system, temperature, json_mode=False, timeout=None, low_thinking=False):
            calls.append((name, low_thinking))
            return model()
        return fallback.Provider(name, call, group=name.split("#")[0] if name.startswith("gemini:") else "")
    monkeypatch.setattr(llm, "PROVIDERS", [make(n, s) for n, s in plan.items()])
    monkeypatch.setattr(llm, "CACHE_MODE", "off")
    return calls


def test_chat_parse_failure_moves_to_the_next_model(monkeypatch):
    def parse(text):
        if text == "broken":
            raise ValueError("not JSON")
        return text
    providers(monkeypatch, {"gemini:a#1": ["broken"], "gemini:b#1": ["fine"]})
    monkeypatch.setitem(config.ROUTES, "helper", "gemini:a, gemini:b")
    reply = REAL_CHAT([{"role": "user", "content": "x"}], parse=parse, route="helper", hedge=False)
    assert (reply.provider, reply.data) == ("gemini:b#1", "fine")


def test_chat_rank_runs_in_the_callers_thread(monkeypatch):
    """rank קורא ל-finish ול-on_stage - בצ'אט, חיווי השלבים של Streamlit עובד רק ב-thread של הסקריפט."""
    providers(monkeypatch, {"gemini:a#1": ["x"]})
    monkeypatch.setitem(config.ROUTES, "helper", "gemini:a")
    seen = []
    REAL_CHAT([{"role": "user", "content": "x"}], route="helper", hedge=True,
              rank=lambda reply: seen.append(threading.current_thread()) or 0)
    assert seen == [threading.current_thread()]


def test_chat_caches_only_the_answer_that_was_picked(monkeypatch, tmp_path):
    providers(monkeypatch, {"gemini:a#1": ["bad"], "cohere:x": ["good"]})
    monkeypatch.setattr(llm, "CACHE_MODE", "on")
    monkeypatch.setattr(llm, "CACHE_FILE", tmp_path / "llm.json")
    monkeypatch.setattr(llm, "_cache", {})
    monkeypatch.setitem(config.ROUTES, "answer", "gemini:a | cohere:x")
    messages = [{"role": "user", "content": "x"}]
    reply = REAL_CHAT(messages, route="answer", hedge=True, rank=lambda r: {"bad": 2, "good": 0}[r.text])
    assert (reply.text, reply.provider) == ("good", "cohere:x") and " מול " in reply.route
    again = REAL_CHAT(messages, route="answer", hedge=True)
    assert again.cached and again.text == "good"


def test_route_lanes_get_their_own_thinking_level(monkeypatch):
    """הבנת השאלה: 3.8 בחשיבה רגילה, 3.6 ומודל הגיבוי - נמוכה (החלטת הבעלים 20/10, 22/10)."""
    monkeypatch.setattr(config, "BACKUP_LLM", "cohere:command-a-plus-05-2026")
    calls = providers(monkeypatch, {"gemini:gemini-3.8-flash#1": [BUSY], "gemini:gemini-3.6-flash#1": [BUSY],
                                    "cohere:command-a-plus-05-2026": [BUSY]})
    with pytest.raises(AllProvidersFailed):
        REAL_CHAT([{"role": "user", "content": "x"}], route="understand", hedge=False)
    assert dict(calls) == {"gemini:gemini-3.8-flash#1": False, "gemini:gemini-3.6-flash#1": True,
                           "cohere:command-a-plus-05-2026": True}


def test_a_route_without_keys_is_skipped_not_crashed(monkeypatch):
    providers(monkeypatch, {"gemini:a#1": ["x"]})
    llm.PROVIDERS[0].enabled = False  # אין מפתח
    monkeypatch.setitem(config.ROUTES, "helper", "gemini:a")
    with pytest.raises(AllProvidersFailed, match="no models"):
        REAL_CHAT([{"role": "user", "content": "x"}], route="helper")


# ---------------------------------------------------------------- run_chain (ווקטורים, Rerank): גם שם לא מוותרים מראש
def test_chain_tries_the_soonest_provider_when_all_are_resting(monkeypatch):
    monkeypatch.setattr(fallback, "_resting_until", {})
    called = []
    p1 = fallback.Provider("t:a", lambda *a: called.append("a") or "a")
    p2 = fallback.Provider("t:b", lambda *a: called.append("b") or "b")
    now = time.time()
    fallback._resting_until.update({"t:a": now + 600, "t:b": now + 30})
    assert fallback.run_chain("embed_query", [p1, p2], retries=0) == ("b", "t:b")  # b חוזר מוקדם יותר
    assert called == ["b"]


def test_chain_never_forces_an_invalid_key(monkeypatch):
    monkeypatch.setattr(fallback, "_resting_until", {"t#1": float("inf")})
    called = []
    p = fallback.Provider("t:a#1", lambda *a: called.append(1), key="t#1")
    with pytest.raises(AllProvidersFailed):
        fallback.run_chain("embed_query", [p], retries=0)
    assert called == []
