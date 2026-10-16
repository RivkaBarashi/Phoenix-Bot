"""בדיקה מקצה לקצה של הבוט על כל השאלות ב-data/questions.csv, וכיול הסף.

לכל שאלה: חיפוש משולב + Rerank של Cohere (מהמטמון כשאפשר) -> המודל מחליט ועונה.
בבדיקה המודל נשאל על כל שאלה - גם מתחת לסף - כדי לחשב מה היה קורה בכל סף.
שופט מספק אחר (Groq, לא Gemini שעונה) משווה כל תשובה לתשובה הצפויה.

הרצה:  .venv\\Scripts\\python eval_answers.py
פירוט מלא, כולל טקסט התשובות: data/eval_answers_<index>.csv
"""
import argparse
import collections
import csv
import json
import sys
import time

from eval_retrieval import RERANKER, best_rank, run_search, same_page, short
from phoenix import answer as answer_layer
from phoenix import config, embeddings, llm, rerank
from phoenix.answer import respond
from phoenix.fallback import CALLS, AllProvidersFailed
from phoenix.search import RRF_K, SearchResult

PAUSE = 6.5  # בין שאלות: Rerank של Cohere (10 בדקה) ומודל השפה החינמי
THRESHOLDS = [0.70, 0.75, 0.80, 0.82, 0.85, 0.88, 0.90, 0.92]
JUDGE_PROVIDERS = ["groq", "openrouter"]

JUDGE_PROMPT = """אתה בודק תשובות של בוט שירות לקוחות של חברת ביטוח. משווים את תשובת הבוט לתשובה הצפויה.
- correct: הבוט עונה נכון על השאלה עם העובדה המרכזית (למשל כן/לא, המספר, הצעד העיקרי). פרטים משניים
  יכולים לחסור, וניסוח אחר או סדר אחר הם בסדר.
- partial: חסרה עובדה מרכזית שצריך כדי לענות על השאלה, או שהתשובה מעורפלת - אבל אין בה טעות.
- wrong: יש בתשובת הבוט עובדה שסותרת את התשובה הצפויה, או משהו שהומצא.
החזר JSON בלבד: {"verdict": "correct" | "partial" | "wrong", "reason": "משפט קצר"}"""


def parse_verdict(text: str) -> dict:
    data = json.loads(text[text.find("{"):text.rfind("}") + 1])
    if data.get("verdict") not in {"correct", "partial", "wrong"}:
        raise ValueError(f"bad verdict: {data.get('verdict')!r}")
    return data


def judge(question: str, expected: str, got: str) -> tuple[str, str]:
    try:
        r = llm.chat([{"role": "user", "content": f"שאלה: {question}\n\nתשובה צפויה: {expected}\n\nתשובת הבוט: {got}"}],
                     system=JUDGE_PROMPT, temperature=0, json_mode=True, parse=parse_verdict, only=JUDGE_PROVIDERS)
        return r.data["verdict"], r.data.get("reason", "")
    except AllProvidersFailed:
        return "n/a", "השופט לא זמין"


def outcome(record: dict, threshold: float) -> str:
    """מה הבוט היה עונה בסף הזה: מתחת לסף - not_on_site, אחרת ההחלטה של המודל."""
    return "not_on_site" if record["top"] < threshold else record["decision"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default=config.EMBEDDER, choices=[p.name.split(":")[0] for p in embeddings.PROVIDERS])
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--only", help="רק השאלות האלה, לפי מספר: 12,13,30")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    # בבדיקה בלי Cohere כמודל שפה: הוא חולק את מכסת ה-Trial (1,000 בחודש) עם ה-Rerank
    answer_layer.COMPACT_LLMS = [p for p in answer_layer.COMPACT_LLMS if p != "cohere"]

    rows = list(csv.DictReader(open(config.ROOT / "data" / "questions.csv", encoding="utf-8-sig")))
    numbers = {int(n) for n in args.only.split(",")} if args.only else None
    print(f"אינדקס: {args.index} | Rerank: {RERANKER} | סף נוכחי: {config.RERANK_THRESHOLD} | {len(rows)} שאלות")

    records = []
    for n, row in enumerate(rows, 1):
        if numbers and n not in numbers:
            continue
        if records:
            time.sleep(PAUSE)
        ranked, used, reranked_by = run_search(row["question"], args.index)
        a = respond(row["question"], SearchResult(ranked[:5], used, reranked_by), threshold=0.0)
        rec = {"n": n, "type": row["type"], "question": row["question"], "top": ranked[0].score if ranked else 0.0,
               "decision": a.kind, "reason": a.reason, "llm": a.llm or "", "sources": " ".join(a.sources),
               "answer": a.text, "verdict": "", "judge_reason": "", "cited_correct": ""}
        if row["type"] == "answerable":
            correct = [h for h in ranked if same_page(h.url, row["source_url"])]
            by_rrf = sorted(ranked, key=lambda h: -sum(1 / (RRF_K + r) for r in (h.vector_rank, h.bm25_rank) if r))
            rec.update(
                final=next((i for i, h in enumerate(ranked, 1) if same_page(h.url, row["source_url"])), None),
                vector=best_rank(h.vector_rank for h in correct), bm25=best_rank(h.bm25_rank for h in correct),
                rrf=next((i for i, h in enumerate(by_rrf, 1) if same_page(h.url, row["source_url"])), None),
                cited_correct=any(same_page(u, row["source_url"]) for u in a.sources),
            )
            if a.kind == "answer" and not args.no_judge:
                rec["verdict"], rec["judge_reason"] = judge(row["question"], row["expected_answer"], a.text)
        records.append(rec)
        source_mark = ("✓מקור" if rec["cited_correct"] else "✗מקור") if row["type"] == "answerable" and a.kind == "answer" else ""
        print(f"\n[{n}] ({row['type']}) {row['question']}")
        print(f"    rerank {rec['top']:.3f} -> {a.kind} ({a.reason}) {rec['verdict']} {source_mark}  [{a.llm}]")
        print("    " + " ".join(a.text.split())[:220])

    answerable = [r for r in records if r["type"] == "answerable"]
    total = len(answerable)

    def hit_at(key, k):
        return sum(1 for r in answerable if r.get(key) and r[key] <= k)

    print(f"\n=== חיפוש ({args.index}): העמוד הנכון בין k הראשונים, {total} שאלות עם תשובה ===")
    print(f"   {'שלב':<16}{'k=1':>6}{'k=3':>6}{'k=5':>6}")
    for label, key in [("וקטורי בלבד", "vector"), ("BM25 בלבד", "bm25"), ("איחוד (RRF)", "rrf"), ("אחרי Rerank", "final")]:
        print(f"   {label:<16}{hit_at(key, 1):>6}{hit_at(key, 3):>6}{hit_at(key, 5):>6}")

    print("\n=== החלטות המודל (בלי סף) ===")
    for kind in ("answerable", "not_on_site", "personal"):
        group = [r for r in records if r["type"] == kind]
        print(f"   {kind:<12} " + ", ".join(f"{k}: {v}" for k, v in collections.Counter(r["decision"] for r in group).items()))
    print("   סיבות לדחייה: " + ", ".join(f"{k}: {v}" for k, v in collections.Counter(
        r["reason"] for r in records if r["decision"] == "not_on_site").items()))
    verdicts = collections.Counter(r["verdict"] for r in answerable if r["decision"] == "answer")
    print(f"   שיפוט התשובות לשאלות עם תשובה: {dict(verdicts)} | ציטטו את העמוד הנכון: "
          f"{sum(1 for r in answerable if r['decision'] == 'answer' and r['cited_correct'])}")

    print("\n=== כיול הסף: מה היה קורה בכל סף (שתי השכבות יחד) ===")
    print(f"   {'סף':>5} | {'עם תשובה: נענו / נכון / נדחו':>28} | {'לא באתר: קיבלו תשובה':>21} | {'אישיות: הופנו / נדחו / נענו':>27}")
    for t in THRESHOLDS + [config.RERANK_THRESHOLD]:
        ans = [outcome(r, t) for r in answerable]
        good = sum(1 for r in answerable if outcome(r, t) == "answer" and r["verdict"] == "correct")
        leaks = sum(1 for r in records if r["type"] == "not_on_site" and outcome(r, t) == "answer")
        pers = [outcome(r, t) for r in records if r["type"] == "personal"]
        mark = "  <- נוכחי" if t == config.RERANK_THRESHOLD else ""
        print(f"   {t:>5.2f} | {ans.count('answer'):>12} / {good:>5} / {ans.count('not_on_site'):>5} | {leaks:>21} | "
              f"{pers.count('personal'):>11} / {pers.count('not_on_site'):>5} / {pers.count('answer'):>5}{mark}")

    out = config.ROOT / "data" / f"eval_answers_{args.index}{'_partial' if numbers else ''}.csv"
    fields = ["n", "type", "question", "top", "decision", "reason", "verdict", "judge_reason", "cited_correct", "llm",
              "sources", "answer"]
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(f"\nפירוט מלא: {out.relative_to(config.ROOT)}")

    print("\n=== קריאות שיצאו ===")
    for name, count in sorted(rerank.API_REQUESTS.items()):
        print(f"   {name}: {count} קריאות API אמיתיות ({rerank.CACHE_HITS[name]} ציונים מהמטמון)")
    for name, count in sorted(CALLS.items()):
        if not name.startswith(("cohere:rerank", "voyage:rerank")):
            print(f"   {name}: {count} ניסיונות")


if __name__ == "__main__":
    main()
