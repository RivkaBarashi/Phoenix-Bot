"""בדיקת החיפוש על כל השאלות ב-data/questions.csv.

לכל שאלה: חיפוש משולב (וקטורים + BM25) ואז Rerank של Cohere (תמיד Cohere, כדי שהציונים
יהיו ניתנים להשוואה). מדפיס:
  - שאלות עם תשובה: באיזה מקום העמוד הנכון - בחיפוש הווקטורי, ב-BM25 ואחרי Rerank.
  - שאלות שאין להן תשובה באתר ושאלות אישיות: ציון ה-Rerank הכי גבוה שנמצא.
  - טבלת ספים: לכל סף - כמה שאלות עם תשובה הבוט היה דוחה, וכמה בלי תשובה היה עונה.
  - כמה קריאות יצאו לכל שירות (כדי לעקוב אחרי המכסות).

הרצה:  .venv\\Scripts\\python eval_retrieval.py                  (שאלות עם תשובה, בלי תשובה ואישיות; אינדקס gemini)
       .venv\\Scripts\\python eval_retrieval.py --index voyage
       .venv\\Scripts\\python eval_retrieval.py --category answerable --only 3,7,12
       .venv\\Scripts\\python eval_retrieval.py --questions data/questions_holdout.csv
3 שאלות ברצף שנכשלות באותה שגיאה - הריצה נעצרת ומדווחת.
"""
import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

from phoenix import config, embeddings, usage
from phoenix.fallback import LAST_ERRORS, resting
from phoenix.search import RRF_K, candidates, rerank_hits

SITE = "https://www.fnx.co.il"
RERANKER = "cohere"
COHERE_PAUSE = 6.5  # מפתח Trial של Cohere: עד 10 קריאות Rerank בדקה
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
QUESTIONS = config.ROOT / "data" / "questions.csv"
TYPES = ["answerable", "not_on_site", "personal", "greeting", "off_topic", "injection"]


def select_rows(path: str | Path, only: str | None, category: str | None) -> list[tuple[int, dict]]:
    """השאלות להרצה, עם המספר שלהן בקובץ (1 = השורה הראשונה אחרי הכותרת).
    only: "3,7,12" - רק השאלות האלה. category: "personal" או "answerable,personal" - רק הסוגים האלה."""
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    numbers = {int(n) for n in only.split(",")} if only else None
    types = set(category.split(",")) if category else None
    if types and (unknown := types - set(TYPES)):
        raise SystemExit(f"סוג לא מוכר: {', '.join(unknown)}. הסוגים: {', '.join(TYPES)}")
    return [(n, row) for n, row in enumerate(rows, 1)
            if (not numbers or n in numbers) and (not types or row["type"] in types)]


class FailureStreak:
    """3 שאלות ברצף שנכשלו באותה שגיאה -> עוצרים את הריצה ומדווחים, במקום לשרוף מכסה על כל השאר."""

    def __init__(self, limit: int = 3):
        self.limit, self.error, self.questions = limit, None, []

    def add(self, n: int, error: str | None) -> bool:
        """error=None: השאלה עברה. מחזיר True כשצריך לעצור."""
        if error != self.error:
            self.error, self.questions = error, []
        if error is not None:
            self.questions.append(n)
        return len(self.questions) >= self.limit

    def report(self) -> str:
        lines = [f"\n!!! הריצה נעצרה: {len(self.questions)} שאלות ברצף ({', '.join(map(str, self.questions))}) "
                 f"נכשלו באותה שגיאה: {self.error}"]
        lines += [f"    {purpose}: " + " | ".join(e[:160] for e in errors)
                  for purpose, errors in LAST_ERRORS.items() if errors]
        if rest := resting():
            lines.append("    נחים עכשיו: " + ", ".join(f"{n} ({s / 60:.0f} דק')" for n, s in sorted(rest.items())))
        lines.append("    מה עושים: לחכות שהמכסה תתאפס או לתקן את התקלה, ולהריץ שוב - "
                     "תשובות שכבר התקבלו יגיעו מהמטמון ולא יישלחו שוב.")
        return "\n".join(lines)


def short(url: str) -> str:
    return url.replace(SITE, "")


def same_page(url: str, expected: str) -> bool:
    """expected: כתובת אחת, או כמה כתובות מופרדות ברווח - לשאלה כללית שכמה עמודים עונים עליה
    ("איך אני מבטלת את הביטוח שלי?": ביטול ביטוח דירה וביטול ביטוח רכב)."""
    return any(url.rstrip("/").lower() == e.rstrip("/").lower() for e in expected.split())


def best_rank(ranks) -> int | None:
    ranks = [r for r in ranks if r]
    return min(ranks) if ranks else None


def run_search(question: str, index: str, timings: dict | None = None) -> tuple[list, str, str]:
    """חיפוש שבו גם האינדקס וגם ה-Rerank הם בדיוק מה שביקשנו - אחרת מחכים ומנסים שוב.
    מחזיר (כל המועמדים מדורגים, האינדקס, ה-Rerank).

    קודם מוודאים שהחיפוש הווקטורי הצליח, ורק אז שולחים ל-Rerank - כדי לא לבזבז
    קריאות Cohere על מועמדים שנמצאו בלי וקטורים.
    """
    for _ in range(6):
        hits, used = candidates(question, embedder=index, fallback=False, timings=timings)
        if (used or "").startswith(index):
            break
        print(f"    (מגבלת קצב באינדקס {index} - מחכה 25 שניות)")
        time.sleep(25)
    else:
        raise RuntimeError(f"vector search on {index} kept failing - try again later")
    for _ in range(4):
        ranked, reranked_by = rerank_hits(question, hits, RERANKER, timings=timings)
        if reranked_by.startswith(RERANKER):
            return ranked, used, reranked_by
        print(f"    (מגבלת קצב ב-{RERANKER} - מחכה 25 שניות)")
        time.sleep(25)
    raise RuntimeError(f"{RERANKER} rerank kept failing - try again later")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default=config.EMBEDDER, choices=[p.name.split(":")[0] for p in embeddings.PROVIDERS])
    parser.add_argument("--questions", default=str(QUESTIONS), help="קובץ השאלות (למשל data/questions_holdout.csv)")
    parser.add_argument("--only", help="רק השאלות האלה, לפי מספר: 3,7,12")
    parser.add_argument("--category", "--types", dest="category", default="answerable,not_on_site,personal",
                        help="רק הסוגים האלה: " + ", ".join(TYPES))
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    config.patient()  # בבדיקות מחכים יותר לכל ספק (בצ'אט: 15 שניות ועוברים מיד לבא)

    types = args.category.split(",")
    selected = select_rows(args.questions, args.only, args.category)
    print(f"אינדקס: {args.index} | Rerank: {RERANKER} | {len(selected)} שאלות מ-{Path(args.questions).name}")

    results = []
    streak = FailureStreak()
    for i, (n, row) in enumerate(selected):
        if i:
            time.sleep(COHERE_PAUSE)
        try:
            hits = run_search(row["question"], args.index)[0]
        except RuntimeError as exc:
            print(f"\n[{n}] ({row['type']}) {row['question']}\n    נכשל: {exc}")
            if streak.add(n, f"search: {exc}"):
                print(streak.report())
                break
            continue
        streak.add(n, None)
        top = hits[0] if hits else None
        record = {"row": row, "top_score": top.score if top else 0.0, "top_url": top.url if top else ""}
        print(f"\n[{n}] ({row['type']}) {row['question']}")
        if row["type"] == "answerable":
            correct = [h for h in hits if same_page(h.url, row["source_url"])]
            by_rrf = sorted(hits, key=lambda h: -sum(1 / (RRF_K + r) for r in (h.vector_rank, h.bm25_rank) if r))
            record.update(
                final=next((i for i, h in enumerate(hits, 1) if same_page(h.url, row["source_url"])), None),
                vector=best_rank(h.vector_rank for h in correct),
                bm25=best_rank(h.bm25_rank for h in correct),
                rrf=next((i for i, h in enumerate(by_rrf, 1) if same_page(h.url, row["source_url"])), None),
            )
            mark = "✓" if record["final"] and record["final"] <= 3 else "✗"
            print(f"    {mark} עמוד נכון: {short(row['source_url'])}")
            print(f"      וקטורי #{record['vector'] or '-'} | BM25 #{record['bm25'] or '-'} | "
                  f"איחוד #{record['rrf'] or '-'} | אחרי Rerank #{record['final'] or '-'}")
        for rank, h in enumerate(hits[:3], 1):
            mark = "✓" if row["type"] == "answerable" and same_page(h.url, row["source_url"]) else " "
            print(f"      {rank}. {mark} {h.score:.3f}  {short(h.url)}  |  {(h.heading or h.title)[:40]}")
        results.append(record)

    answerable = [r for r in results if r["row"]["type"] == "answerable"]
    if answerable:
        total = len(answerable)

        def hit_at(key, k):
            return sum(1 for r in answerable if r[key] and r[key] <= k)

        print(f"\n=== שאלות עם תשובה ({total}): העמוד הנכון בין k הראשונים ===")
        print(f"   {'שלב':<16}{'k=1':>6}{'k=3':>6}{'k=5':>6}")
        for label, key in [("וקטורי בלבד", "vector"), ("BM25 בלבד", "bm25"), ("איחוד (RRF)", "rrf"),
                           ("אחרי Rerank", "final")]:
            print(f"   {label:<16}{hit_at(key, 1):>6}{hit_at(key, 3):>6}{hit_at(key, 5):>6}")

    print("\n=== ציון ה-Rerank הכי גבוה לכל שאלה ===")
    for kind in types:
        scores = sorted(r["top_score"] for r in results if r["row"]["type"] == kind)
        if scores:
            print(f"   {kind:<12} מינימום {scores[0]:.3f} | חציון {statistics.median(scores):.3f} | מקסימום {scores[-1]:.3f}")
    for r in results:
        if r["row"]["type"] != "answerable":
            print(f"   {r['top_score']:.3f}  ({r['row']['type']}) {r['row']['question'][:45]}  ->  {short(r['top_url'])}")

    print("\n=== סף: מתחת לציון הזה הבוט עונה 'לא מופיע באתר' ===")
    print(f"   {'סף':>5} | {'עם תשובה שיידחו':>16} | {'לא באתר שיקבלו תשובה':>21} | {'אישיות מעל הסף':>15}")
    for t in THRESHOLDS:
        refused = sum(r["top_score"] < t for r in answerable)
        leaked = sum(r["top_score"] >= t for r in results if r["row"]["type"] == "not_on_site")
        personal = sum(r["top_score"] >= t for r in results if r["row"]["type"] == "personal")
        print(f"   {t:>5.1f} | {refused:>16} | {leaked:>21} | {personal:>15}")

    print("\n" + usage.report())


if __name__ == "__main__":
    main()
