"""בדיקות יחידה לכללים שבקוד (בלי API): ברכות, שאלה אישית, מספרים, מילים שלא באתר וציטוטים.

הרצה:  .venv\\Scripts\\python -m pytest
"""
import json
import time

import pytest

from phoenix import answer as A
from phoenix import fallback, llm
from phoenix.search import Hit, SearchResult

REAL_CHAT = llm.chat  # לפני ש-conftest חוסם אותו - לבדיקות של llm.chat עצמו עם ספקים מדומים

LRM = "\u200e"

# ---------------------------------------------------------------- split_greeting


@pytest.mark.parametrize("text, rest, greeted, thanked", [
    ("היי, איך מגישים תביעה?", "איך מגישים תביעה?", True, False),
    ("בוקר טוב", "", True, False),
    ("מה קורה?", "", True, False),
    ("תודה רבה!", "", True, True),
    ("שלום, תודה, מה המספר של המוקד?", "מה המספר של המוקד?", True, True),
    ("מה קורה עם התביעה שלי?", "מה קורה עם התביעה שלי?", False, False),  # שאלה, לא ברכה
    ("שלומית מבקשת לבטל ביטוח", "שלומית מבקשת לבטל ביטוח", False, False),  # שם, לא "שלום"
    ("הלוואה מהפניקס", "הלוואה מהפניקס", False, False),                    # לא "הלו"
])
def test_split_greeting(text, rest, greeted, thanked):
    assert A.split_greeting(text) == (rest, greeted, thanked)


def test_greeting_answer_only_for_pure_greetings():
    assert A.greeting_answer("שלום").text == A.GREETING_REPLY
    assert A.greeting_answer("תודה רבה").text == A.THANKS_REPLY
    assert A.greeting_answer("היי, איך מגישים תביעה?") is None


# ---------------------------------------------------------------- is_personal_question

@pytest.mark.parametrize("question", [
    "האם הפוליסה שלי מכסה את הנזק לדוד השמש?",
    "אני מבוטחת בביטוח דירה אצלכם, נזק מצנרת מכוסה אצלי?",
    "האם הביטוח שלי כולל רכב חלופי?",
    "כמה כסף אקבל על הרכב שלי שהוכרז טוטאל לוס?",
    "כמה יש לי בקופת חיסכון לכל ילד?",
    "מה היתרה בקרן הפנסיה שלי?",
    "האם אני זכאית להחזר על פיזיותרפיה?",
    "מגיע לי פיצוי על הרכב שלי?",
])
def test_personal_questions(question):
    assert A.is_personal_question(question)


@pytest.mark.parametrize("question", [
    "איך אני מגישה תביעה על הביטוח שלי?",       # "איך עושים" בגוף ראשון
    "איפה אני מעדכנת פרטים בפוליסה שלי?",
    "איך אני מבטלת את הביטוח שלי?",
    "אני מבוטחת אצלכם, איך מגישים תביעה?",
    "האם ביטוח תכולה כולל צד שלישי?",            # שאלה כללית
    "כמה זמן יש לי לתבוע את הפניקס?",            # "כמה זמן", לא "כמה יש לי"
    "אני עוברת דירה. ביטוח התכולה מכסה את הרהיטים בהובלה?",
    "מה היתרון של ביטוח מקיף?",                  # "יתרון", לא "יתרה"
])
def test_not_personal_questions(question):
    assert not A.is_personal_question(question)


# ---------------------------------------------------------------- unsupported_numbers

SITE = ["מוקד שירות לקוחות: 5432* או 054-9893455. גבול האחריות 3,500 ₪. בתוקף מ-26.11.2020, שעות 8:30-16:00."]


@pytest.mark.parametrize("answer_text", [
    f"פנו ל-054-9893455 או {LRM}*5432.",   # קוד כוכבית בצורה האחידה
    "עד 3500 ₪ או 3,500 ₪",               # אותו סכום עם פסיק ובלי
    "מ-26.11.2020, בין 8:30 ל-16:00",
    "בלי מספרים בכלל",
])
def test_numbers_from_the_site_are_allowed(answer_text):
    assert A.unsupported_numbers(answer_text, SITE) == []


@pytest.mark.parametrize("answer_text, bad", [
    ("התקשרו ל-03-1234567", ["03-1234567"]),  # טלפון שהומצא
    ("עד 4,000 ₪", ["4,000"]),
    ("מ-27.11.2020", ["27.11.2020"]),
])
def test_invented_numbers_are_caught(answer_text, bad):
    assert A.unsupported_numbers(answer_text, SITE) == bad


# ---------------------------------------------------------------- terms_not_on_site (מול האינדקס המקומי)

@pytest.mark.parametrize("question, missing", [
    ("אפשר לבטח אצלכם טרקטורון?", ["טרקטורון"]),       # באתר יש רק "טרקטורים"
    ("כמה עולה ביטוח לקורולה?", ["לקורולה"]),            # "קורונה" באתר - מילה אחרת
    ("יש לכם ביטוח לרחפן?", ["לרחפן"]),
    ("יש לכם ביטוח לקרוואן?", []),                      # "קרוואנים" באתר - ן סופית
    ("איך מגישים תביעה על נזקי מים?", []),
])
def test_terms_not_on_site(question, missing):
    assert A.terms_not_on_site(question) == missing


# ---------------------------------------------------------------- quote_found

TEXT = "תשלום התביעה מתבצע תוך 30 יום מרגע הגעת כל המסמכים הנדרשים לחברה, בכפוף לאישורה."


def test_exact_quote():
    assert A.quote_found("תוך 30 יום מרגע הגעת כל המסמכים", [TEXT])


def test_quote_with_small_wording_change():
    # 10 מתוך 11 מילים (90%) וכל המספרים
    assert A.quote_found("תשלום התביעה מתבצע תוך 30 יום מרגע הגעת כל המסמכים הדרושים לחברה", [TEXT])


def test_quote_with_wrong_number_fails():
    assert not A.quote_found("תשלום התביעה מתבצע תוך 20 יום מרגע הגעת כל המסמכים הנדרשים לחברה", [TEXT])


def test_quote_too_short_or_absent_fails():
    assert not A.quote_found("תשלום", [TEXT])
    assert not A.quote_found("הפיצוי מחושב לפי מחירון לוי יצחק", [TEXT])


# ---------------------------------------------------------------- כללים נוספים מהסבב האחרון

@pytest.mark.parametrize("question, depends", [
    ("ותוך כמה זמן?", True), ("ומה עם ביטוח רכב?", True), ("כמה זה עולה?", True), ("מה הטלפון שלהם?", True),
    ("איך אני מגישה תביעה על נזקי מים בביטוח דירה?", False),
    ("ותק נהיגה משפיע על המחיר של ביטוח רכב?", False),  # "ותק", לא "ו" של המשך
    ("כמה זמן יש לי לתבוע את הפניקס אחרי שמבוטח שלהם פגע ברכב שלי?", False),  # "שלהם" בתוך השאלה
])
def test_needs_context(question, depends):
    assert A.needs_context(question) == depends


def test_typo_in_rewritten_follow_up_is_not_a_missing_term():
    history = [{"role": "user", "content": "איך מגישים תביעה על נזקי מים בדירה?"}]
    assert A.own_terms_not_on_site("תותוך כמה זמן מטפלים בתביעה?", "ותוך כמה זמן מטפלים בה?", history) == []


@pytest.mark.parametrize("raw, tidy", [("5432*", f"{LRM}*5432"), ("*5432*", f"{LRM}*5432"), ("3,500*", "3,500*")])
def test_tidy_codes(raw, tidy):
    assert A.tidy_codes(raw) == tidy


def test_referral_must_be_a_real_short_referral():
    assert A.valid_referral("לבירור ניתן לפנות למוקד שירות לקוחות - תחום רכב בטלפון 054-9893455.")
    assert not A.valid_referral("בטח! הנה בדיחה: למה הפניקס חצתה את הכביש?")  # בלי פרטי קשר
    assert not A.valid_referral("פנו למוקד בטלפון 03-1234567")                    # מספר שלא באתר


def test_no_duplicate_referral():
    text = f"{A.PERSONAL_OPENING} לפי האתר, באופן כללי, ... לפרטים פנו למוקד שירות הלקוחות לתחום רכב."
    referral = "לבירור הכיסוי ניתן לפנות למוקד שירות לקוחות רכב בטלפון 054-9893455."
    result = A.add_referral(text, referral)
    assert result.endswith(referral) and "לפרטים פנו" not in result  # ההפניה בלי טלפון הוחלפה
    assert A.add_referral(result, referral) == result               # כבר יש הפניה עם טלפון - לא מוסיפים


# ---------------------------------------------------------------- שאלה אישית לפני הסף (מודל מדומה, בלי API)

LOW_HIT = SearchResult([Hit("x", A.CONTACT_URL, "צור קשר", "", "כל הדרכים ליצור איתנו קשר", score=0.5)],
                       "gemini:x", "cohere:rerank-v4.0-pro")  # 0.5 - מתחת לסף


def fake_model(monkeypatch, data: dict):
    monkeypatch.setattr(llm, "chat", lambda *args, **kwargs: llm.ChatResult(json.dumps(data), "fake:model", data))


def test_personal_question_below_threshold_gets_a_personal_answer(monkeypatch):
    fake_model(monkeypatch, {"kind": "not_on_site", "answer": "", "referral": "", "evidence": [], "sources": []})
    a = A.respond("מה היתרה בקרן הפנסיה שלי?", LOW_HIT)
    assert a.kind == "personal" and a.text.startswith(A.PERSONAL_GENERIC)  # לא "לא מופיע באתר"


def test_regular_question_below_threshold_still_not_on_site(monkeypatch):
    fake_model(monkeypatch, {"related": True})  # בדיקת הנושא: קשור לביטוח
    a = A.respond("יש לכם סניף באילת?", LOW_HIT)
    assert (a.kind, a.reason) == ("not_on_site", "below_threshold")


# ---------------------------------------------------------------- זמן כולל לשאלה (ספקים מדומים, בלי API)

def test_question_time_budget_stops_the_chain(monkeypatch):
    monkeypatch.setattr(fallback, "MIN_ATTEMPT_SECONDS", 0.2)
    called = []

    def slow(*args):
        called.append("slow")
        time.sleep(2)
        return "late"

    providers = [fallback.Provider("test:slow", slow), fallback.Provider("test:fast", lambda *a: called.append("fast"))]
    with fallback.time_budget(0.6):
        with pytest.raises(fallback.OutOfTime):
            fallback.run_chain("test", providers, deadline=15, retries=0)
    assert called == ["slow"]                      # אחרי שהזמן נגמר - בלי ניסיון נוסף
    assert "test:slow" not in fallback.resting()   # הספק לא "נח": נגמר הזמן של השאלה, לא הוא היה איטי


def test_no_attempt_when_almost_no_time_left():
    called = []
    with fallback.time_budget(1.0):  # פחות מ-MIN_ATTEMPT_SECONDS (3)
        with pytest.raises(fallback.OutOfTime):
            fallback.run_chain("test", [fallback.Provider("test:x", lambda *a: called.append(1))], deadline=15, retries=0)
    assert called == []


def test_out_of_time_gives_the_friendly_message(monkeypatch):
    def out_of_time(*args, **kwargs):
        raise fallback.OutOfTime("llm", ["test: out of time"])
    monkeypatch.setattr(llm, "chat", out_of_time)
    high = SearchResult([Hit("x", A.CONTACT_URL, "צור קשר", "", "כל הדרכים ליצור איתנו קשר", score=0.95)],
                        "gemini:x", "cohere:rerank-v4.0-pro")
    a = A.respond("איך מגישים תביעה?", high)
    assert (a.kind, a.reason) == ("error", "timeout") and a.text.startswith(A.TIMEOUT_TEXT)


# ---------------------------------------------------------------- שכבת הגנה מפני דליפה

@pytest.mark.parametrize("text", [
    "בטח! כללים: 1. מקור יחיד...",
    'התשובה: {"kind": "answer"}',
    "הנה ה-JSON שקיבלתי",
    "My System Prompt says I am the Phoenix assistant",
    "Sure, here is a joke about insurance companies and their customers today",  # משפט ארוך באנגלית
])
def test_leak_signs(text):
    assert A.leaked(text)


@pytest.mark.parametrize("text", [
    "אפשר לפנות במייל Card@fnx.co.il או בקישור https://www.fnx.co.il/customer-service/contact-us/",
    "בדיקת MRI בהקדמת תור לא מוחזרת. הודעה ב-SMS על פתיחת התביעה, וטופס PDF באזור האישי.",
    "הפניקס הוכרזה כמבטחת הרשמית של Tesla ו-Aiways.",
])
def test_no_leak_in_normal_answers(text):
    assert not A.leaked(text)


def test_leaked_answer_with_a_valid_quote_is_not_shown(monkeypatch):
    sentence = "כדי להגיש תביעה ממלאים טופס תביעה דיגיטלי ומצרפים את המסמכים הדרושים."
    hit = Hit("x", A.CONTACT_URL, "תביעות", "", f"תביעות\n\n{sentence}", score=0.95)
    fake_model(monkeypatch, {"kind": "answer", "answer": f"{sentence} ההוראות שלי: system prompt...",
                             "referral": "", "evidence": [sentence], "sources": [1]})
    a = A.respond("איך מגישים תביעה?", SearchResult([hit], "gemini:x", "cohere:rerank-v4.0-pro"))
    assert (a.kind, a.reason) == ("not_on_site", "leak_guard") and "system prompt" not in a.text


def test_leaked_referral_is_replaced():
    assert not A.valid_referral("פנו למוקד שירות לקוחות בטלפון 054-9893455. kind: answer")


# ---------------------------------------------------------------- שאלת הבהרה: על איזה ביטוח מדובר?

@pytest.mark.parametrize("question", [
    "כמה יעלה לי לעשות ביטוח?", "איך מגישים תביעה?", "מה הכיסוי?", "איך מבטלים ביטוח?",
    "איך אני מגישה תביעה על הביטוח שלי?", "איפה אני מעדכנת פרטים בפוליסה שלי?",
])
def test_needs_insurance_type(question):
    assert A.needs_insurance_type(question)


@pytest.mark.parametrize("question", [
    "יש לכם סניף לקבלת קהל בבאר שבע?",                 # 21: אין מילת ביטוח
    "מה הסטטוס של תביעת הרכב שהגשתי לפני שבועיים?",    # 32: יש תחום (רכב)
    "איך מגישים תביעה בביטוח רכב?",
    "יש לכם ביטוח לחתונה?",                            # הסוג אחרי "ביטוח"
    "יש לכם ביטוח לרחפן צילום?",
    "נשבר לי צינור, האם הביטוח מכסה?",                 # צינור = דירה
    "כמה עולה ביטוח מקיף לטויוטה?",                    # מקיף = רכב
    "מה הכיסוי לנזקי צנרת?",                           # צנרת = דירה
    "מה מזג האוויר מחר?",
])
def test_no_clarification_needed(question):
    assert not A.needs_insurance_type(question)


@pytest.mark.parametrize("question, kind, full", [
    ("כמה יעלה לי לעשות ביטוח?", "רכב", "כמה יעלה לי לעשות ביטוח רכב?"),
    ("איך מגישים תביעה?", "רכב", "איך מגישים תביעה בביטוח רכב?"),
    ("מה הכיסוי?", "דירה", "מה הכיסוי בביטוח דירה?"),
    ("איך מבטלים ביטוח?", "דירה", "איך מבטלים ביטוח דירה?"),
    ("איך אני מגישה תביעה על הביטוח שלי?", "בריאות", "איך אני מגישה תביעה על ביטוח הבריאות שלי?"),
    ("איפה אני מעדכנת פרטים בפוליסה שלי?", "חיים", "איפה אני מעדכנת פרטים בפוליסת ביטוח החיים שלי?"),
    ("איך מגישים תביעה?", "ביטוח לרחפן", "איך מגישים תביעה בביטוח לרחפן?"),  # "אחר" - מה שהלקוחה כתבה
])
def test_with_insurance_type(question, kind, full):
    assert A.with_insurance_type(question, kind) == full
    assert not A.needs_insurance_type(full)  # אחרי הבחירה - שאלה רגילה, בלי הבהרה נוספת


def test_clarification_without_search_or_model(monkeypatch):
    def no_search(*args, **kwargs):
        raise AssertionError("שאלת הבהרה לא אמורה לחפש")
    monkeypatch.setattr(A, "search", no_search)
    a = A.answer("איך מגישים תביעה?")
    assert (a.kind, a.text) == ("clarify", A.CLARIFY_TEXT)


def test_no_clarification_when_the_type_was_said_earlier(monkeypatch):
    calls = []
    monkeypatch.setattr(A, "search", lambda *a, **k: calls.append("search") or SearchResult([], None, "none"))
    monkeypatch.setattr(A, "standalone_question", lambda q, h: "איך מגישים תביעה בביטוח רכב?")
    history = [{"role": "user", "content": "יש לי ביטוח רכב אצלכם"}, {"role": "assistant", "content": "..."}]
    a = A.answer("ואיך מגישים תביעה?", history)
    assert a.kind != "clarify" and calls == ["search"]


# ---------------------------------------------------------------- תשובה אישית בלי מודל

HIGH_HIT = SearchResult([Hit("x", A.CONTACT_URL, "צור קשר", "", "כל הדרכים ליצור איתנו קשר", score=0.95)],
                        "gemini:x", "cohere:rerank-v4.0-pro")


@pytest.mark.parametrize("error", [fallback.AllProvidersFailed, fallback.OutOfTime])
def test_personal_question_without_a_model(monkeypatch, error):
    def unavailable(*args, **kwargs):
        raise error("llm", ["all models: quota"])
    monkeypatch.setattr(llm, "chat", unavailable)
    a = A.respond("האם הביטוח שלי כולל רכב חלופי?", HIGH_HIT)
    assert (a.kind, a.reason) == ("personal", "personal_no_model")
    assert a.text.startswith(A.PERSONAL_GENERIC) and A.GENERIC_REFERRAL in a.text


def test_regular_question_without_a_model_is_still_an_error(monkeypatch):
    def unavailable(*args, **kwargs):
        raise fallback.AllProvidersFailed("llm", ["all models: quota"])
    monkeypatch.setattr(llm, "chat", unavailable)
    assert A.respond("מה המספר של מוקד תביעות רכב?", HIGH_HIT).reason == "llm_unavailable"


# ---------------------------------------------------------------- מאגר מפתחות Gemini (ספקים מדומים, בלי API)

class FakeApiError(Exception):
    """שגיאה כמו של google-genai: קוד HTTP בשדה code."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


DAILY_QUOTA = "Quota exceeded for metric GenerateRequestsPerDay, limit 20 per day. Please retry in 6h12m3s."


def chain_providers(behaviour: dict, called: list) -> list:
    """2 מודלים x 2 מפתחות, לפי הסדר: מודל ראשון בכל המפתחות, ואז השני."""
    def make(name):
        def call(*args):
            called.append(name)
            if isinstance(behaviour.get(name), Exception):
                raise behaviour[name]
            return name
        return call
    return [fallback.Provider(f"t:{m}#{k}", make(f"{m}#{k}"), group=f"t:{m}", key=f"t#{k}")
            for m in ("m1", "m2") for k in (1, 2)]


def test_daily_quota_moves_to_the_next_key_of_the_same_model(monkeypatch):
    monkeypatch.setattr(fallback, "_resting_until", {})
    called = []
    providers = chain_providers({"m1#1": FakeApiError(429, DAILY_QUOTA)}, called)
    assert fallback.run_chain("test", providers, retries=0)[1] == "t:m1#2"
    assert called == ["m1#1", "m1#2"]                 # אותו מודל, המפתח הבא - לפני מודל אחר
    rest = fallback.resting()
    assert rest["t:m1#1"] > 6 * 3600                  # הצירוף מפתח+מודל נח עד האיפוס שהספק דיווח
    assert "t:m1" not in rest and "t#1" not in rest   # המודל (בשאר המפתחות) והמפתח (בשאר המודלים) - לא
    called.clear()
    fallback.run_chain("test", providers, retries=0)
    assert called == ["m1#2"]                         # בקריאה הבאה: מדלגים על m1#1 בלי לנסות


def test_invalid_key_rests_on_all_models(monkeypatch):
    monkeypatch.setattr(fallback, "_resting_until", {})
    called = []
    providers = chain_providers({"m1#1": FakeApiError(403, "API key not valid")}, called)
    fallback.run_chain("test", providers, retries=0)
    called.clear()
    assert fallback.run_chain("test", providers[2:], retries=0)[1] == "t:m2#2"
    assert called == ["m2#2"]                         # מפתח 1 לא נוסה גם במודל השני


# ---------------------------------------------------------------- עומס: 5 דקות, ומעבר מיד למודל הבא

def test_overload_rests_the_model_five_minutes_on_all_keys(monkeypatch):
    monkeypatch.setattr(fallback, "_resting_until", {})
    called = []
    providers = chain_providers({"m1#1": FakeApiError(503, "The model is overloaded. Please try again later.")}, called)
    assert fallback.run_chain("test", providers, retries=1, max_wait=10)[1] == "t:m2#1"
    assert called == ["m1#1", "m2#1"]       # בלי ניסיון חוזר ובלי המתנה, ובלי m1 במפתח 2: עוברים למודל הבא
    rest = fallback.resting()
    assert 290 < rest["t:m1"] <= 300         # המודל נח 5 דקות - לא עד איפוס המכסה
    called.clear()
    fallback.run_chain("test", providers, retries=0)
    assert called == ["m2#1"]               # ב-5 הדקות הבאות - המודל מדולג בכל המפתחות


def test_timeout_counts_as_overload(monkeypatch):
    monkeypatch.setattr(fallback, "_resting_until", {})
    called = []

    def slow(*args):
        called.append("slow")
        time.sleep(1)

    providers = [fallback.Provider("t:m1#1", slow, group="t:m1", key="t#1"),
                 fallback.Provider("t:m2#1", lambda *a: called.append("fast") or "ok", group="t:m2", key="t#1")]
    assert fallback.run_chain("test", providers, retries=1, deadline=0.2)[1] == "t:m2#1"
    assert called == ["slow", "fast"] and 290 < fallback.resting()["t:m1"] <= 300


def test_llm_chat_keeps_model_and_key_groups(monkeypatch):
    """דרך llm.chat (לא רק run_chain): עומס במודל 1 - המודל מדולג בכל המפתחות."""
    monkeypatch.setattr(fallback, "_resting_until", {})
    monkeypatch.setattr(llm, "CACHE_MODE", "off")
    called = []

    def make(name, error=None):
        def call(messages, system, temperature, json_mode=False, timeout=None):
            called.append(name)
            if error:
                raise error
            return '{"ok": true}'
        return call

    fake = [fallback.Provider("gemini:m1#1", make("m1#1", FakeApiError(503, "overloaded")), group="gemini:m1",
                              key="gemini#1"),
            fallback.Provider("gemini:m1#2", make("m1#2"), group="gemini:m1", key="gemini#2"),
            fallback.Provider("gemini:m2#1", make("m2#1"), group="gemini:m2", key="gemini#1")]
    monkeypatch.setattr(llm, "PROVIDERS", fake)
    assert REAL_CHAT([{"role": "user", "content": "x"}]).provider == "gemini:m2#1"
    assert called == ["m1#1", "m2#1"]  # m1 במפתח 2 לא נוסה: עומס הוא של המודל


# ---------------------------------------------------------------- בקשות במקביל בצ'אט (מודל מזויף, בלי API)

SENTENCE = "כדי להגיש תביעה ממלאים טופס תביעה דיגיטלי ומצרפים את המסמכים הדרושים."
RACE_HIT = SearchResult([Hit("x", A.CONTACT_URL, "תביעות", "", f"תביעות\n\n{SENTENCE}", score=0.95)],
                        "gemini:x", "cohere:rerank-v4.0-pro")
VALID = {"kind": "answer", "answer": SENTENCE, "referral": "", "evidence": [SENTENCE], "sources": [1]}
UNVERIFIED = {"kind": "answer", "answer": "ממלאים טופס.", "referral": "", "evidence": ["משפט שלא מופיע באתר בכלל"],
              "sources": [1]}


def fake_race(monkeypatch, plan: dict):
    """plan: הספק הראשון של כל צד (או "chain" לשרשרת הרגילה) -> (השהיה בשניות, תשובה או חריגה)."""
    monkeypatch.setattr(llm, "CACHE_MODE", "off")
    monkeypatch.setattr(llm, "race_pair", lambda service="gemini": [["gemini:a#1"], ["gemini:b#2"]])

    def chat(messages, names=None, **kwargs):
        delay, data = plan[names[0] if names else "chain"]
        time.sleep(delay)
        if isinstance(data, Exception):
            raise data
        return llm.ChatResult(json.dumps(data, ensure_ascii=False), names[0] if names else "gemini:chain#1", data)
    monkeypatch.setattr(llm, "chat", chat)


def test_race_first_valid_answer_wins(monkeypatch):
    fake_race(monkeypatch, {"gemini:a#1": (0.0, VALID), "gemini:b#2": (2.0, VALID)})
    start = time.perf_counter()
    a = A.respond("איך מגישים תביעה?", RACE_HIT, parallel=True)
    assert (a.kind, a.llm) == ("answer", "gemini:a#1") and "במקביל" in a.race
    assert time.perf_counter() - start < 1.5      # לא מחכים לתשובה השנייה


def test_race_waits_when_the_first_answer_fails_verification(monkeypatch):
    fake_race(monkeypatch, {"gemini:a#1": (0.0, UNVERIFIED), "gemini:b#2": (0.3, VALID)})
    a = A.respond("איך מגישים תביעה?", RACE_HIT, parallel=True)
    assert (a.kind, a.reason, a.llm) == ("answer", "verified", "gemini:b#2")


def test_race_both_fail_continues_with_the_chain(monkeypatch):
    failed = fallback.AllProvidersFailed("llm", ["quota"])
    fake_race(monkeypatch, {"gemini:a#1": (0.0, failed), "gemini:b#2": (0.0, failed), "chain": (0.0, VALID)})
    a = A.respond("איך מגישים תביעה?", RACE_HIT, parallel=True)
    assert (a.kind, a.llm) == ("answer", "gemini:chain#1")


def test_eval_is_never_parallel(monkeypatch):
    fake_race(monkeypatch, {"chain": (0.0, VALID)})
    monkeypatch.setattr(llm, "race_pair", lambda *args, **kwargs: pytest.fail("בבדיקות אין בקשות במקביל"))
    assert A.respond("איך מגישים תביעה?", RACE_HIT).llm == "gemini:chain#1"
