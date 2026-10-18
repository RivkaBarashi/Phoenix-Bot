"""בדיקות יחידה לכללים שבקוד (בלי API): ברכות, שאלה אישית, מספרים, מילים שלא באתר וציטוטים.

הרצה:  .venv\\Scripts\\python -m pytest
"""
import pytest

from phoenix import answer as A

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
