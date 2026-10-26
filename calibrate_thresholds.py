"""כיול סף ה-Rerank (docs/before-paid-tests.md, סעיף 1) - רק חיפוש ודירוג, בלי מודל שפה.

לכל שאלה עם תשובה באתר (answerable) - הציון של הקטע הכי טוב. הסף נבחר כך ששאלות עם תשובה כמעט לא ייחסמו: הסף הגבוה
ביותר שבו הגבול העליון (95%, Clopper-Pearson) של שיעור השאלות עם תשובה שמתחת לסף קטן או שווה ל-alpha. הבדיקה רצה מהסף
הנמוך לגבוה ועוצרת בראשון שנכשל ("fixed sequence" של Learn then Test): השיעור עולה ככל שהסף עולה, ולכן אין צורך בתיקון
להשוואות מרובות. שאלות שאין להן תשובה באתר (not_on_site) - רק למידע: כמה מהן עוברות את הסף ומגיעות למודל (שם מחליטים
המודל והבדיקות). שאלות אישיות - לא (הן לא נחסמות בסף).

הרצה:
  python calibrate_thresholds.py                       הסט הראשי + החדש, Cohere, alpha 0.10 - רק מדפיס
  python calibrate_thresholds.py --write               ושומר ל-data/thresholds.json (עם גרסת האינדקס ומודל הדירוג)
  python calibrate_thresholds.py --reranker voyage --alpha 0.05
את הסט הנסתר הסקריפט לא מקבל.

הערה: מחפשים על השאלה כמו שנכתבה. בצ'אט שלב ההבנה מנסח אותה מחדש (תיקון כתיב, תרגום), כך שהציונים שם קרובים אבל לא זהים.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import eval_retrieval
from phoenix import answer_cache, config, rerank, stats
from phoenix.rerank import API_REQUESTS

GRID = [round(0.30 + 0.01 * i, 2) for i in range(70)]  # 0.30 ... 0.99
DEFAULT_SETS = [config.ROOT / "data" / "questions.csv", config.ROOT / "data" / "questions_v2.csv"]


def choose(scores: list[float], alpha: float, grid: list[float] = GRID) -> tuple[float, int, float]:
    """(סף, כמה שאלות עם תשובה מתחתיו, הגבול העליון) - הסף הגבוה ביותר ברשת שעוד עומד בדרישה, לפי הסדר מלמטה.
    אם גם הסף הנמוך ברשת לא עומד בה - הוא מוחזר (הכי מקל שיש), והגבול העליון שלו מראה שהדרישה לא התקיימה."""
    chosen = None
    for t in grid:
        k = sum(s < t for s in scores)
        upper = stats.clopper_pearson_upper(k, len(scores))
        if upper > alpha:
            break
        chosen = (t, k, upper)
    if chosen is None:
        k = sum(s < grid[0] for s in scores)
        chosen = (grid[0], k, stats.clopper_pearson_upper(k, len(scores)))
    return chosen


def load_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        if "holdout" in path.name.lower():
            sys.exit(f"{path.name}: הסט הנסתר לא משמש לכיול - הוא רץ רק פעם אחת, בסוף")
        with open(path, encoding="utf-8-sig") as f:
            rows += [row | {"set": path.name} for row in csv.DictReader(f) if row["type"] in ("answerable", "not_on_site")]
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", nargs="+", default=[str(p) for p in DEFAULT_SETS])
    parser.add_argument("--reranker", default="cohere", choices=list(config.RERANK_THRESHOLDS))
    parser.add_argument("--alpha", type=float, default=0.10,
                        help="שיעור מקסימלי (בגבול העליון) של שאלות עם תשובה שנחסמות בטעות")
    parser.add_argument("--write", action="store_true", help="לשמור ל-data/thresholds.json")
    parser.add_argument("--per-minute", type=float, default=9,
                        help="קריאות Cohere לדקה לכל היותר (מפתח Trial: 10 לדקה; 0 - בלי הגבלה, למפתח בתשלום)")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    config.patient()
    rerank.COHERE_MIN_INTERVAL = 60 / args.per_minute if args.per_minute > 0 else 0
    eval_retrieval.RERANKER = args.reranker
    rows = load_rows([Path(p) for p in args.questions])
    scored = []
    for i, row in enumerate(rows, 1):
        try:
            ranked, _, _ = eval_retrieval.run_search(row["question"], config.EMBEDDER)
        except RuntimeError as exc:  # החיפוש או הדירוג לא הצליחו גם אחרי המתנות - השאלה לא נספרת, ומדווחת
            print(f"[{i}/{len(rows)}] נכשל ({exc}) - {row['question'][:60]}")
            continue
        top = ranked[0].score if ranked else 0.0
        scored.append(row | {"top": top})
        print(f"[{i}/{len(rows)}] {top:.3f} {row['type']:12} {row['question'][:70]}")
        time.sleep(0.5)

    answerable = [r["top"] for r in scored if r["type"] == "answerable"]
    missing = [r["top"] for r in scored if r["type"] == "not_on_site"]
    if not answerable:
        sys.exit("אין שאלות עם תשובה שנבדקו - אין מה לכייל")
    threshold, blocked, upper = choose(answerable, args.alpha)
    current = config.RERANK_THRESHOLDS[args.reranker]
    print(f"\n{len(answerable)} שאלות עם תשובה, {len(missing)} בלי | {sum(1 for r in rows) - len(scored)} נכשלו")
    print(f"{'סף':>6} {'עם תשובה שנחסמו':>18} {'גבול עליון':>11} {'בלי תשובה שעברו למודל':>22}")
    for t in sorted({0.70, 0.75, 0.80, 0.85, 0.90, threshold, current}):
        k = sum(s < t for s in answerable)
        mark = " <- נבחר" if t == threshold else " <- היום" if t == current else ""
        print(f"{t:>6.2f} {k:>9}/{len(answerable):<8} {stats.clopper_pearson_upper(k, len(answerable)) * 100:>9.1f}% "
              f"{sum(s >= t for s in missing):>12}/{len(missing)}{mark}")
    print(f"\nנבחר: {threshold} (alpha {args.alpha}: {blocked} נחסמות מתוך {len(answerable)}, גבול עליון "
          f"{upper * 100:.1f}%) | היום: {current}")
    if args.write:
        data = json.loads(config.THRESHOLDS_FILE.read_text(encoding="utf-8")) if config.THRESHOLDS_FILE.exists() else {}
        data[args.reranker] = {
            "threshold": threshold, "model": dict(config.RERANK_CHAIN)[args.reranker],
            "index_version": answer_cache.index_version(), "alpha": args.alpha, "answerable": len(answerable),
            "blocked": blocked, "upper_bound": round(upper, 4), "not_on_site": len(missing),
            "not_on_site_passed": sum(s >= threshold for s in missing),
            "sets": sorted({r["set"] for r in scored}), "calibrated": time.strftime("%Y-%m-%d %H:%M"),
            "rerank_requests": sum(API_REQUESTS.values()),
        }
        config.THRESHOLDS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        scores_file = config.ROOT / "data" / f"threshold_scores_{args.reranker}.csv"
        with open(scores_file, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["set", "type", "top", "question"], extrasaction="ignore")
            writer.writeheader()
            writer.writerows(sorted(scored, key=lambda r: r["top"]))
        print(f"נשמר: {config.THRESHOLDS_FILE.relative_to(config.ROOT)} ו-{scores_file.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
