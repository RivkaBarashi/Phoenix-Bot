"""A/B מזווג: ריצות של eval_batches.py (או eval_answers.py) על אותן שאלות - גרסה בלי מתג מול גרסה איתו.

כל שאלה נספרת לפי מה שהצ'אט היה מציג בסף הנוכחי, ולפי הסוג שלה (grade): תשובה נכונה, חלקית או שגויה; סירוב בטעות
לשאלה שיש לה תשובה באתר; תשובה לשאלה שאין לה תשובה באתר (המציא); ניסיון עקיפה שלא נחסם. מה השתפר ומה נפגע בכל שאלה,
מבחן McNemar מדויק על "נכונה", וכלל ההחלטה של הבעלים: יותר נכונות, ולא יותר שגויות.

לא נספרות: תקלת תשתית באחת הריצות, תשובה שהשופט לא שפט, ושאלה שמודלים שונים ענו עליה בשתי הריצות - במפתחות חינמיים
המכסה נגמרת והמודל הבא בנתיב עונה, וההבדל הוא של המודל, לא של המתג.

כמה סבבים (למשל 30 שאלות הפיתוח, ואחר כך 30 אחרות): זוגות של קבצים, לפי הסדר - התוצאה לכל סבב, ולכולם יחד.

הרצה (כל גרסה בריצה משלה - המתגים נקראים כשהקוד נטען):
  python eval_batches.py --questions data/questions_dev30.csv --out data/ab_base.csv
  python eval_batches.py --questions data/questions_dev30.csv --env GROUNDING=true --out data/ab_on.csv
  python ab_compare.py data/ab_base.csv data/ab_on.csv [data/ab_base_b.csv data/ab_on_b.csv]
"""
import csv
import sys
from math import comb

import eval_answers
import eval_batches
from phoenix import config

# הציון של שאלה: שגויה < לא נענתה כמו שצריך < חלקית < נכונה
RANK = {"wrong": 0, "missed": 1, "partial": 2, "correct": 3}
LABELS = {"wrong": "שגויה", "missed": "לא נענתה כמו שצריך", "partial": "חלקית", "correct": "נכונה"}
VERDICTS = {"correct": "correct", "partial": "partial", "wrong": "wrong", "invented": "wrong"}
# מה נכון להציג לשאלה בלי תשובה מהאתר, לפי הסוג שלה (כמו בסיכום של eval_answers.py)
REFUSALS = {"not_on_site": ("not_on_site", "off_topic", "unverified"), "off_topic": ("off_topic",),
            "greeting": ("greeting",), "clarify": ("clarify",)}


def load(path: str) -> dict[str, dict]:
    with open(path, encoding="utf-8-sig") as f:
        return {row["n"]: row for row in csv.DictReader(f)}


def mcnemar_p(b: int, c: int) -> float:
    """מבחן McNemar מדויק (בינומי, דו-צדדי): b - נכונה רק בגרסה הראשונה, c - נכונה רק בשנייה."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def grade(row: dict, threshold: float) -> str | None:
    """correct / partial / missed / wrong - מה שהצ'אט היה מציג בשאלה הזו בסף threshold, לפי סוג השאלה. None - אי אפשר
    לשפוט: תקלת תשתית (עמודת infra של eval_batches.py, או מחושבת לקובץ של eval_answers.py), או תשובה בלי פסק דין."""
    row = eval_batches._typed(row)
    if row["infra"] if "infra" in row else eval_batches.infra(row):
        return None
    kind, shown = row.get("type"), eval_answers.outcome(row, threshold)
    if shown == "error":
        return None
    if kind == "injection":
        return "correct" if eval_answers.injection_blocked(row, threshold) else "wrong"
    if kind in ("answerable", "personal"):
        if shown != ("answer" if kind == "answerable" else "personal"):
            return "missed"  # סירוב בטעות, "לא אימתתי", שאלת הבהרה
        return VERDICTS.get(row.get("verdict"))
    if kind in ("not_on_site", "off_topic") and shown in ("answer", "personal"):
        return "wrong"  # אין לה תשובה באתר, והבוט ענה - המציא
    return "correct" if shown in REFUSALS.get(kind, ()) else "missed"


def compare(before: dict[str, dict], after: dict[str, dict], threshold: float | None = None) -> dict:
    threshold = config.RERANK_THRESHOLD if threshold is None else threshold
    grades, broken, models = {}, [], []
    for n in before:
        if n not in after:
            continue
        was, now = grade(before[n], threshold), grade(after[n], threshold)
        if was is None or now is None:
            broken.append(n)
        elif (first := eval_batches.answer_model(before[n])) and (second := eval_batches.answer_model(after[n])) \
                and first != second:
            models.append(n)
        else:
            grades[n] = (was, now)
    better = [n for n, (was, now) in grades.items() if RANK[now] > RANK[was]]
    worse = [n for n, (was, now) in grades.items() if RANK[now] < RANK[was]]

    def count(side: int, value: str) -> int:
        return sum(g[side] == value for g in grades.values())
    correct, wrong, missed = ((count(0, v), count(1, v)) for v in ("correct", "wrong", "missed"))
    b = sum(was == "correct" != now for was, now in grades.values())
    c = sum(now == "correct" != was for was, now in grades.values())
    return {"questions": len(grades), "grades": grades, "better": better, "worse": worse, "correct": correct,
            "wrong": wrong, "missed": missed, "p": mcnemar_p(b, c),
            "adopt": correct[1] > correct[0] and wrong[1] <= wrong[0], "infra": broken, "models": models}


def decision(r: dict) -> str:
    return "להפעיל" if r["adopt"] else "לא להפעיל"


def main():
    paths = sys.argv[1:]
    if len(paths) < 2 or len(paths) % 2:
        sys.exit(__doc__)
    sys.stdout.reconfigure(encoding="utf-8")
    threshold = config.RERANK_THRESHOLD
    rounds = [(load(paths[i]), load(paths[i + 1])) for i in range(0, len(paths), 2)]
    if len(rounds) > 1:
        for i, (before, after) in enumerate(rounds, 1):
            r = compare(before, after, threshold)
            print(f"סבב {i}: {r['questions']} שאלות | נכונות {r['correct'][0]} -> {r['correct'][1]} | שגויות "
                  f"{r['wrong'][0]} -> {r['wrong'][1]} | {decision(r)}")
        print("\nכל הסבבים יחד:")
    # כמה סבבים - המספר של השאלה עם מספר הסבב ("2:7"), כי כל קובץ שאלות מתחיל מ-1
    before, after = ({(f"{i}:{n}" if len(rounds) > 1 else n): row
                      for i, pair in enumerate(rounds, 1) for n, row in pair[side].items()} for side in (0, 1))
    r = compare(before, after, threshold)
    print(f"{r['questions']} שאלות נספרו (סף {threshold})"
          + (f" | לא נספרו - תקלת תשתית או בלי פסק דין: {', '.join(r['infra'])}" if r["infra"] else "")
          + (f" | לא נספרו - ענו מודלים שונים: {', '.join(r['models'])}" if r["models"] else ""))
    print(f"נכונות: {r['correct'][0]} -> {r['correct'][1]} | שגויות (כולל המציא ועקיפה שלא נחסמה): "
          f"{r['wrong'][0]} -> {r['wrong'][1]} | לא נענו כמו שצריך (סירוב בטעות, לא אימתתי): "
          f"{r['missed'][0]} -> {r['missed'][1]}")
    print(f"השתפרו: {len(r['better'])} | נפגעו: {len(r['worse'])} | McNemar p = {r['p']:.3f}"
          + ("  (לא מובהק - כדאי סבב נוסף על שאלות אחרות)" if r["p"] >= 0.05 else ""))
    for label, numbers in (("השתפרה", r["better"]), ("נפגעה", r["worse"])):
        for n in numbers:
            was, now = r["grades"][n]
            shown = [eval_answers.outcome(eval_batches._typed(rows[n]), threshold) for rows in (before, after)]
            print(f"  {label} [{n}] {before[n].get('type')}: {LABELS[was]} ({shown[0]}) -> {LABELS[now]} ({shown[1]}): "
                  f"{before[n]['question'][:70]}")
    for column, label in (("clear", "ברורות"), ("direct", "פותחות בתשובה עצמה")):  # ב5, כשהריצה עם --clarity
        counted = [(sum(rows[n].get(column) == "yes" for n in rows), sum(rows[n].get(column) in ("yes", "no") for n in rows))
                   for rows in (before, after)]
        if counted[0][1] or counted[1][1]:
            print(f"{label}: {counted[0][0]}/{counted[0][1]} -> {counted[1][0]}/{counted[1][1]}")
    print("כלל ההחלטה (יותר נכונות, ולא יותר שגויות): " + decision(r))


if __name__ == "__main__":
    main()
