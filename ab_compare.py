"""A/B מזווג: שתי ריצות של eval_answers.py על אותן שאלות - גרסה בלי מתג מול גרסה איתו.

מה השתפר ומה נפגע בכל שאלה, מבחן McNemar מדויק על "נכונה מלאה", וכלל ההחלטה של הבעלים: יותר נכונות, ולא יותר שגויות.

הרצה (כל ריצה בתהליך משלה - המתגים נקראים כשהקוד נטען):
  set CONTEXT_PAGES=false & python eval_answers.py --llm gemini-3.6-flash --fresh & copy data\\eval_answers_gemini.csv data\\ab_off.csv
  set CONTEXT_PAGES=true  & python eval_answers.py --llm gemini-3.6-flash --fresh & copy data\\eval_answers_gemini.csv data\\ab_on.csv
  python ab_compare.py data\\ab_off.csv data\\ab_on.csv
"""
import csv
import sys
from math import comb

# סדר הטיב של פסק הדין: שגויה < חלקית < נכונה. "blocked" (ניסיון עקיפה שנחסם) = נכונה. בלי פסק דין - לא נספרת.
RANK = {"wrong": 0, "invented": 0, "error": 0, "partial": 1, "correct": 2, "blocked": 2}


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


def compare(before: dict[str, dict], after: dict[str, dict]) -> dict:
    # eval_batches.py מסמן תקלת תשתית (עמודת infra): אף מודל לא ענה, ההבנה נכשלה... - לא טעות של הבוט, ולא נספרת
    broken = [n for n in before if n in after and (before[n].get("infra") or after[n].get("infra"))]
    common = [n for n in before if n in after and n not in broken
              and before[n]["verdict"] in RANK and after[n]["verdict"] in RANK]
    better = [n for n in common if RANK[after[n]["verdict"]] > RANK[before[n]["verdict"]]]
    worse = [n for n in common if RANK[after[n]["verdict"]] < RANK[before[n]["verdict"]]]

    def count(rows, n_list, verdicts):
        return sum(rows[n]["verdict"] in verdicts for n in n_list)
    correct = (count(before, common, ("correct", "blocked")), count(after, common, ("correct", "blocked")))
    wrong = (count(before, common, ("wrong", "invented", "error")), count(after, common, ("wrong", "invented", "error")))
    b = sum(RANK[before[n]["verdict"]] == 2 and RANK[after[n]["verdict"]] < 2 for n in common)
    c = sum(RANK[after[n]["verdict"]] == 2 and RANK[before[n]["verdict"]] < 2 for n in common)
    return {"questions": len(common), "better": better, "worse": worse, "correct": correct, "wrong": wrong,
            "p": mcnemar_p(b, c), "adopt": correct[1] > correct[0] and wrong[1] <= wrong[0], "infra": broken}


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.stdout.reconfigure(encoding="utf-8")
    before, after = load(sys.argv[1]), load(sys.argv[2])
    r = compare(before, after)
    print(f"{r['questions']} שאלות עם פסק דין בשתי הריצות"
          + (f" | {len(r['infra'])} לא נספרו - תקלת תשתית באחת הריצות: {', '.join(r['infra'])}" if r["infra"] else ""))
    print(f"נכונות מלאות: {r['correct'][0]} -> {r['correct'][1]} | שגויות: {r['wrong'][0]} -> {r['wrong'][1]}")
    print(f"השתפרו: {len(r['better'])} | נפגעו: {len(r['worse'])} | McNemar p = {r['p']:.3f}"
          + ("  (לא מובהק - כדאי לחזור על הריצה)" if r["p"] >= 0.05 else ""))
    for label, rows in (("השתפרה", r["better"]), ("נפגעה", r["worse"])):
        for n in rows:
            print(f"  {label} [{n}] {before[n]['verdict']} -> {after[n]['verdict']}: {before[n]['question'][:70]}")
    for column, label in (("clear", "ברורות"), ("direct", "פותחות בתשובה עצמה")):  # ב5, כשהריצה עם --clarity
        counted = [(sum(rows[n].get(column) == "yes" for n in rows), sum(rows[n].get(column) in ("yes", "no") for n in rows))
                   for rows in (before, after)]
        if counted[0][1] or counted[1][1]:
            print(f"{label}: {counted[0][0]}/{counted[0][1]} -> {counted[1][0]}/{counted[1][1]}")
    print("כלל ההחלטה (יותר נכונות, ולא יותר שגויות): " + ("להפעיל" if r["adopt"] else "לא להפעיל"))


if __name__ == "__main__":
    main()
