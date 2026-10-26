"""build_question_sets.py: 30 מכל סט לפי היחס בין הסוגים, בהגרלה קבועה, בלי לשנות אף תשובה צפויה; הודעות עם כמה
שאלות - מתשובות שכבר אושרו, מילה במילה; ובדיקת המספרים בקובץ הבדיקה."""
import csv

import build_question_sets as Q
from phoenix import answer as A


def rows_of(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


MAIN = rows_of(Q.DATA / "questions.csv")


def test_thirty_by_the_type_ratio_the_same_every_time_without_changing_a_row():
    chosen = Q.stratified(MAIN, Q.QUOTAS)
    assert len(chosen) == 30 and Q.stratified(MAIN, Q.QUOTAS) == chosen  # זרע קבוע - אותה בחירה
    for kind, count in Q.QUOTAS.items():
        assert sum(r["type"] == kind for r in chosen) == count
    for r in chosen:  # מועתקת כמו שהיא - התשובה הצפויה לא משתנה
        assert {k: r[k] for k in MAIN[0]} == MAIN[r["orig_n"] - 1]
    assert len({" ".join(r["question"].split()) for r in chosen}) == 30  # בלי שאלות זהות (60 ו-72 בסט הראשי)


def test_multi_questions_are_built_from_approved_answers_word_for_word():
    multi = Q.multi_rows(MAIN)
    assert len(multi) == 10
    for row in multi:
        numbers = [int(n) for n in row["orig_n"].split("+")]
        assert all(MAIN[n - 1]["expected_answer"] in row["expected_answer"] for n in numbers)
        assert all(url in row["source_url"].split() for n in numbers for url in MAIN[n - 1]["source_url"].split())
    as_is = next(r for r in multi if r["orig_n"] == "68")
    assert {k: as_is[k] for k in MAIN[0]} == MAIN[67]  # השאלה עם שני החלקים שכבר בסט - כמו שהיא
    assert max(len(r["orig_n"].split("+")) for r in multi) == 3


def test_the_numbers_check_skips_step_numbers_and_tells_another_page_from_nowhere():
    page = "https://www.fnx.co.il/x/"
    pages = {Q._page_key(page): ["כותרת\n\nתשובה תוך 30 יום מהגשת הטופס."], Q._page_key(A.CONTACT_URL): ["*5432"]}
    site = "\n".join(t for texts in pages.values() for t in texts) + "\nאזרח ותיק (67+)"

    def check(expected, kind="answerable"):
        row = {"question": "q", "expected_answer": expected, "source_url": page, "type": kind}
        return Q.support(row, pages, site)[1]
    assert check("חובה: 1) ממלאים טופס. 2) מקבלים תשובה תוך 30 יום.") == "כל המספרים בעמוד"
    assert check("תוך 30 יום, ואזרח ותיק (67+) בעל פה.") == "לא בעמוד, אבל באתר (בעמוד אחר): 67"
    assert check("תוך 45 יום.") == "לא באתר בכלל: 45"
    assert check("לפנות למוקד *5432.", kind="personal") == "כל המספרים בעמוד"  # מספרי ההפניה - מעמוד צור קשר
