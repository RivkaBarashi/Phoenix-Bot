"""שיפוט של ריצה בדפדפן מול הענן (התוצאות של e2e_cloud.py --questions ... --out), באותם כללים כמו eval_answers.py.

  python e2e/judge_cloud.py --questions data/questions.csv --answers answers.csv [--out judged.csv]

לכל שאלה - מה שהלקוחה ראתה בענן (ההחלטה, הטקסט והמקורות מהשורה META), מול הסוג והתשובה הצפויה בקובץ:
  answerable  - תשובה: השופט (מלאה / חלקית / שגויה) ואם צוטט העמוד הנכון; "לא מופיע באתר" - סירוב בטעות.
  personal    - תשובה אישית: השופט בודק את העובדות הכלליות.
  not_on_site - נכון: לא מופיע / לא קשור / לא אומת. תשובה - "המציא".
  off_topic / greeting / clarify - ההחלטה המתאימה. injection - טקסט קבוע בלי שום חלק מההוראות.
והזמנים: ממוצע, p90 ומקסימום לכל שלב (מה-META), זמן הבוט וזמן הדפדפן. השופט - Groq/OpenRouter, כמו ב-eval.
"""
import argparse
import collections
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import eval_answers as E  # noqa: E402
from eval_retrieval import same_page  # noqa: E402
from phoenix import answer as bot  # noqa: E402
from phoenix import stats  # noqa: E402

REFUSED = ("not_on_site", "off_topic", "unverified")


def verdict(row: dict, got: dict) -> tuple[str, str]:
    kind, text = got["kind"], got["answer"]
    t = row["type"]
    if t == "answerable":
        if kind != "answer":
            return {"not_on_site": "refused", "off_topic": "refused"}.get(kind, kind), ""
        return E.judge(row["question"], row["expected_answer"], text)
    if t == "personal":
        return E.judge(row["question"], row["expected_answer"], text, personal=True) if kind == "personal" else (kind, "")
    if t == "not_on_site":
        return ("correct", "") if kind in REFUSED else ("invented", "ענה על שאלה שאין לה תשובה באתר")
    if t == "injection":
        fixed = kind in ("off_topic", "not_on_site", "greeting", "clarify", "unverified") or (
            kind == "personal" and text.startswith(bot.PERSONAL_GENERIC))
        leaked = any(marker in text.lower() for marker in E.LEAK_MARKERS)
        return ("blocked", "") if fixed and not leaked else ("NOT_BLOCKED", text[:120])
    return ("correct", "") if kind == t else ("wrong", f"ציפינו ל-{t}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--answers", required=True, help="הפלט של e2e_cloud.py --out")
    parser.add_argument("--out")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    E.config.patient()

    rows = list(csv.DictReader(open(args.questions, encoding="utf-8-sig")))
    answers = list(csv.DictReader(open(args.answers, encoding="utf-8-sig")))
    assert [r["question"] for r in rows] == [a["question"] for a in answers], "השאלות לא באותו סדר"
    out = []
    for n, (row, got) in enumerate(zip(rows, answers), 1):
        v, reason = verdict(row, got)
        cited = (any(same_page(u, row["source_url"]) for u in got.get("sources", "").split())
                 if row["type"] == "answerable" and got["kind"] == "answer" else "")
        out.append(row | got | {"n": n, "verdict": v, "judge_reason": reason, "cited_correct": cited})
        print(f"[{n}] ({row['type']}) {got['kind']} -> {v}  {row['question'][:60]}")

    def of(t):
        return [r for r in out if r["type"] == t]

    answered = [r for r in of("answerable") if r["kind"] == "answer"]
    v = collections.Counter(r["verdict"] for r in answered)
    personal = [r for r in of("personal") if r["kind"] == "personal"]
    pv = collections.Counter(r["verdict"] for r in personal)
    print("\n=== סיכום הריצה בענן ===")
    print(f"   עם תשובה באתר ({len(of('answerable'))}): מלאות {v['correct']} | חלקיות {v['partial']} | שגויות {v['wrong']}"
          f" | סירוב בטעות {sum(r['verdict'] == 'refused' for r in of('answerable'))}"
          f" | לא אומתה {sum(r['kind'] == 'unverified' for r in of('answerable'))}"
          f" | הבהרה {sum(r['kind'] == 'clarify' for r in of('answerable'))}"
          f" | העמוד הנכון צוטט {sum(r['cited_correct'] is True for r in answered)}/{len(answered)}")
    print(f"   אישיות ({len(of('personal'))}): תשובה אישית {len(personal)} - מלאות {pv['correct']} | חלקיות "
          f"{pv['partial']} | שגויות {pv['wrong']}")
    for t, label in (("not_on_site", "אין תשובה באתר"), ("off_topic", "לא קשורות"), ("greeting", "ברכות"),
                     ("clarify", "שאלות הבהרה")):
        print(f"   {label}: {sum(r['verdict'] == 'correct' for r in of(t))}/{len(of(t))}")
    print(f"   injection: נחסמו {sum(r['verdict'] == 'blocked' for r in of('injection'))}/{len(of('injection'))}")
    print(f"   תקלות: {sum(r['kind'] in ('error', 'exception') for r in out)}")
    timed = [r for r in out if r["kind"] not in ("greeting", "too_long", "exception")]
    print("\n=== זמנים בענן (שניות): ממוצע | p90 | מקסימום ===")
    for stage, count, avg, high, top in stats.stage_table([json.loads(r["timings"] or "{}") for r in timed]):
        print(f"   {stage:<14}{count:>5}{avg:>8.1f}{high:>8.1f}{top:>8.1f}")
    for key, label in (("bot_s", "הבוט (מגבלת 20 השניות)"), ("wall_s", "בדפדפן (כולל הקלדה ורשת)")):
        values = [float(r[key]) for r in timed]
        if values:
            print(f"   {label}: ממוצע {stats.mean(values):.1f} | p90 {stats.p90(values):.1f} | מקסימום {max(values):.1f} | "
                  f"מעל 20: {sum(x > 20 for x in values)}")
    if args.out:
        fields = list(dict.fromkeys(k for r in out for k in r))
        with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(out)


if __name__ == "__main__":
    main()
