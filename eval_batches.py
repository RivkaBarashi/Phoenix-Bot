"""eval_answers.py בחבילות, כל חבילה על מפתח Gemini אחר - רק לבדיקות, ורק כשהמפתחות חינמיים.

למה: הבוט עובד עם מפתח Gemini אחד (config.LLM_GEMINI_KEY), ובמפתח חינמי יש 20 בקשות ביום לכל מודל. שאלה אחת צריכה
עד 2 בקשות מכל מודל (3.8: הבנת השאלה ובדיקת העיגון; 3.6: התשובה וניסיון תיקון), ולכן 30 שאלות לא נכנסות במפתח אחד.
כאן הבדיקה מתחלקת לחבילות של עד 8 שאלות (16 בקשות לכל מודל לכל היותר - מרווח לניסיון חוזר), וכל חבילה רצה בתהליך
משלה עם LLM_GEMINI_KEY אחר. הבוט עצמו לא משתנה. בבדיקה בתשלום אין בזה צורך: eval_answers.py רגיל, על המפתח הראשון.

תשתית מול איכות: שאלה שנכשלה בגלל תשתית (אף מודל לא ענה, הבנת השאלה נכשלה, הבודק או השופט לא זמינים, החיפוש נכשל)
רצה שוב בחבילה אחרת, על המפתח הבא, עד --attempts פעמים. מפתח שכל החבילה שלו נכשלה בתשתית (המכסה היומית נגמרה) - לא
חוזרים אליו. מה שנשאר תקלת תשתית מדווח בנפרד ולא נספר כטעות של הבוט: ריצה בתשלום תתקן אותו. טעות איכות (המודל ענה,
ובטעות) - תשלום לא יתקן, וצריך לתקן את הקוד. אחרי האיפוס היומי (10:00 בשעון ישראל) ובשעות שקטות - פחות תקלות.

הרצה (כל גרסה בריצה משלה - המתגים נקראים כשהקוד נטען):
  python eval_batches.py --questions data/questions.csv --only 1,2,5 --out data/ab_off.csv
  python eval_batches.py --questions data/questions.csv --only 1,2,5 --env TERMS_IN_CITED=true --out data/ab_on.csv
  python ab_compare.py data/ab_off.csv data/ab_on.csv
  --keys 3-28: רק המפתחות האלה (ברירת מחדל: כולם). אחרי -- עובר ל-eval_answers.py, למשל: -- --llm gemini-3.6-flash
החבילות נשמרות ב-logs/batches/<התווית>/ (פלט ופירוט לכל חבילה). מספרי המפתחות בלבד - הערכים לא מודפסים.
"""
import argparse
import collections
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

import eval_answers
from eval_retrieval import TYPES, select_rows
from phoenix import config

ROOT = Path(__file__).resolve().parent
BATCH_TIMEOUT = 3600  # שניות לחבילה - תהליך שנתקע לא עוצר את כל הריצה
KEY_FAILURES = ("no_model", "understand", "error", "verifier")  # תקלות שתלויות במפתח Gemini (לא השופט)


def parse_keys(text: str | None, available: list[int]) -> list[int]:
    """"3,5,7-9" -> [3, 5, 7, 8, 9]; בלי - כל המפתחות. מפתח שלא מוגדר - שגיאה (לא מדלגים בשקט)."""
    if not text:
        return list(available)
    keys = []
    for part in text.split(","):
        low, _, high = part.strip().partition("-")
        if not low.isdigit() or (high and not high.isdigit()):
            raise SystemExit(f"--keys: '{part}' - מספרים או טווחים, למשל 3,5,7-9")
        keys += range(int(low), int(high or low) + 1)
    if missing := [k for k in keys if k not in available]:
        raise SystemExit(f"--keys: אין מפתחות {missing} (מוגדרים: {available})")
    return list(dict.fromkeys(keys))


def infra(row: dict) -> str:
    """למה השורה היא תקלת תשתית, או "" - לא תקלה: מה שתשלום או ריצה חוזרת מתקנים, ולא טעות של הבוט."""
    reason = row.get("reason") or ""
    if row.get("decision") == "error" or reason.startswith(("search_failed", "llm_unavailable", "batch_failed")):
        return "error"
    if reason.startswith("no_model") or reason == "personal_no_model":
        return "no_model"
    if row.get("intent") == "rules":  # הבנת השאלה לא ענתה - כללי הקוד במקומה (תוצאה שונה מהצ'אט)
        return "understand"
    if (row.get("grounding") or "").startswith("unavailable"):  # הבודק לא ענה - התשובה בלי בדיקת העיגון
        return "verifier"
    if row.get("verdict") == "n/a":
        return "judge"
    return ""


def run_all(numbers: list[int], keys: list[int], size: int, attempts: int, run_batch) -> dict[int, dict]:
    """כל השאלות, בחבילות של size, ומפתח אחר לכל חבילה (לפי הסדר, בסבב). שאלה עם תקלת תשתית - לסוף התור: חבילה אחרת,
    מפתח אחר, עד attempts פעמים. תוצאה בלי תקלה לא מוחלפת בתקלה. run_batch(numbers, key) -> {n: row}; שאלה שחסרה
    בתוצאה (התהליך קרס) - תקלת תשתית. מפתח שכל החבילה שלו נכשלה בגללו - לא חוזרים אליו (המכסה שלו נגמרה)."""
    results: dict[int, dict] = {}
    tries: collections.Counter = collections.Counter()
    queue, usable, turn = list(numbers), list(keys), 0
    while queue and usable:
        batch, queue = queue[:size], queue[size:]
        key = usable[turn % len(usable)]
        turn += 1
        rows = run_batch(batch, key)
        failed_by_key = 0
        for n in batch:
            tries[n] += 1
            row = (rows.get(n) or {"n": str(n), "decision": "error", "reason": "batch_failed"}) | {
                "key": key, "attempts": tries[n]}
            kind = infra(row)
            failed_by_key += kind in KEY_FAILURES
            if n not in results or not kind or infra(results[n]):
                results[n] = row
            if kind and tries[n] < attempts:
                queue.append(n)
        if failed_by_key == len(batch):
            usable.remove(key)
            turn -= 1  # המפתח הבא בתור הוא עכשיו במקום של זה
    for n in queue:  # נגמרו המפתחות - מה שנשאר בתור נשאר עם התוצאה האחרונה שלו (או בלי תוצאה בכלל)
        results.setdefault(n, {"n": str(n), "decision": "error", "reason": "batch_failed: no keys left",
                               "attempts": tries[n]})
    return results


def summary(results: dict[int, dict]) -> list[str]:
    """תשתית בנפרד מאיכות: לכל סוג שאלה - פסקי הדין או ההחלטות של מה שנבדק באמת; ורשימת תקלות התשתית שנשארו."""
    rows = [results[n] for n in sorted(results)]
    broken = [r for r in rows if infra(r)]
    checked = [r for r in rows if not infra(r)]
    lines = [f"{len(rows)} שאלות | נבדקו באיכות: {len(checked)} | תקלות תשתית שנשארו: {len(broken)} "
             f"| רצו יותר מפעם אחת: {sum(int(r.get('attempts') or 1) > 1 for r in rows)}"]
    lines.append("\nאיכות (מה שהבוט ענה, בלי תקלות תשתית):")
    for kind in TYPES:
        group = [r for r in checked if r.get("type") == kind]
        if not group:
            continue
        decisions = collections.Counter(eval_answers.outcome(_typed(r), config.RERANK_THRESHOLD) for r in group)
        verdicts = collections.Counter(r["verdict"] for r in group if r.get("verdict"))
        judged = " | פסקי דין: " + ", ".join(f"{k}: {v}" for k, v in verdicts.most_common()) if verdicts else ""
        lines.append(f"   {kind:<12} " + ", ".join(f"{k}: {v}" for k, v in decisions.most_common()) + judged)
    if broken:
        lines.append("\nתשתית (לא נספרות כטעות - ריצה חוזרת או בתשלום):")
        lines += [f"   [{r['n']}] {infra(r)} ({(r.get('reason') or '')[:50]}) {(r.get('question') or '')[:60]}"
                  for r in broken]
    keys = collections.Counter(r.get("key") for r in rows if r.get("key") is not None)
    lines.append("\nמפתחות שענו: " + ", ".join(f"{k}: {v}" for k, v in sorted(keys.items())))
    return lines


def _typed(row: dict) -> dict:
    """שורה מקובץ -> מה ש-outcome צריך: top כמספר, והדגלים (בקובץ - "True" / "False" / ריק) כ-bool."""
    flags = {name: str(row.get(name)) == "True" for name in ("personal_rule", "topic_off", "pre_search", "chat_path")}
    return row | {"top": float(row.get("top") or 0)} | flags


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", default=str(eval_answers.QUESTIONS))
    parser.add_argument("--only", help="רק השאלות האלה, לפי מספר: 3,7,12")
    parser.add_argument("--category", help="רק הסוגים האלה: " + ", ".join(TYPES))
    parser.add_argument("--keys", help="מספרי המפתחות לסבב, למשל 3-28 (ברירת מחדל: כולם)")
    parser.add_argument("--batch", type=int, default=8, help="שאלות בחבילה (8: עד 16 בקשות לכל מודל, מתוך 20)")
    parser.add_argument("--attempts", type=int, default=3, help="כמה פעמים שאלה רצה לכל היותר (תקלת תשתית - שוב)")
    parser.add_argument("--env", action="append", default=[], metavar="NAME=VALUE",
                        help="מתג לגרסה הזו, למשל TERMS_IN_CITED=true (אפשר כמה)")
    parser.add_argument("--out", required=True, help="הפירוט המאוחד, למשל data/ab_on.csv (ל-ab_compare.py)")
    parser.add_argument("--label", help="שם לתיקיית החבילות ב-logs/batches (ברירת מחדל: שם קובץ הפלט)")
    parser.add_argument("--final", action="store_true", help="הריצה הסופית - רק איתה אפשר את הסט הנסתר")
    args, extra = parser.parse_known_args()
    extra = [a for a in extra if a != "--"]
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    if "holdout" in Path(args.questions).name.lower() and not args.final:
        sys.exit("הסט הנסתר רץ רק פעם אחת, בריצה הסופית (--final)")
    changes = {}
    for item in args.env:
        name, sep, value = item.partition("=")
        if not sep or not name.isidentifier():
            sys.exit(f"--env: '{item}' - בצורה NAME=VALUE")
        changes[name] = value
    changes.setdefault("COHERE_MIN_INTERVAL", "6.7")  # מפתח Cohere מסוג Trial: עד 9 קריאות Rerank לדקה (מתוך 10)
    keys = parse_keys(args.keys, [number for number, _ in config.GEMINI_KEYS])
    numbers = [n for n, _ in select_rows(args.questions, args.only, args.category)]
    if not numbers or not keys:
        sys.exit("אין שאלות או אין מפתחות להרצה")
    folder = ROOT / "logs" / "batches" / (args.label or Path(args.out).stem)
    folder.mkdir(parents=True, exist_ok=True)
    print(f"{len(numbers)} שאלות, חבילות של {args.batch}, מפתחות {keys[0]}..{keys[-1]} ({len(keys)}), "
          f"מתגים: {changes or '-'} | פירוט החבילות: {folder}")
    counter = iter(range(1, 10_000))

    def run_batch(batch: list[int], key: int) -> dict[int, dict]:
        tag = f"{next(counter):02d}_key{key}"
        out, log = folder / f"batch_{tag}.csv", folder / f"batch_{tag}.log"
        command = [sys.executable, "-X", "utf8", str(ROOT / "eval_answers.py"), "--questions", args.questions,
                   "--only", ",".join(map(str, batch)), "--out", str(out), "--no-history", *extra]
        env = os.environ | changes | {"LLM_GEMINI_KEY": str(key), "PYTHONIOENCODING": "utf-8"}
        start = time.perf_counter()
        try:
            with open(log, "w", encoding="utf-8") as f:
                code = subprocess.run(command, env=env, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT,
                                      timeout=BATCH_TIMEOUT).returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
        rows = {}
        if out.exists():
            with open(out, encoding="utf-8-sig") as f:
                rows = {int(r["n"]): r for r in csv.DictReader(f)}
        broken = sum(1 for n in batch if n not in rows or infra(rows[n]))
        print(f"[{tag}] שאלות {','.join(map(str, batch))}: {len(batch) - broken} תקינות, {broken} תקלות תשתית "
              f"({time.perf_counter() - start:.0f} ש', יציאה {code})")
        return rows

    results = run_all(numbers, keys, args.batch, args.attempts, run_batch)
    fields = eval_answers.CSV_FIELDS + ["key", "attempts", "infra"]
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results[n] | {"infra": infra(results[n])} for n in sorted(results))
    print("\n" + "\n".join(summary(results)))
    print(f"\nהפירוט המאוחד: {args.out} (ל-ab_compare.py: תקלות תשתית בלי פסק דין - לא נספרות)")


if __name__ == "__main__":
    main()
