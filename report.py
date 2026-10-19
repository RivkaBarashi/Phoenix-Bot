"""סיכום יומן השיחות של הצ'אט (logs/conversations.jsonl).

מה שואלים, מה הבוט החליט, כמה זמן זה לקח (ממוצע ו-p90), איזה מודל ענה, מה המשוב (👍/👎), אילו
תשובות קיבלו 👎 (לבדיקה), ומה אנשים מחפשים ולא מוצאים באתר.

הרצה:  .venv\\Scripts\\python report.py
       .venv\\Scripts\\python report.py --since 2026-10-20
       .venv\\Scripts\\python report.py --log path\\to\\conversations.jsonl
"""
import argparse
import collections
import sys
from pathlib import Path

from phoenix import conversation_log, stats, store

STAGES = ["total", "understand", "condense", "search", "vector", "bm25", "rerank", "search_wait", "llm", "repair",
          "topic"]
STAGE_NAMES = {"total": "סה\"כ", "understand": "הבנת השאלה", "condense": "השלמת שאלת המשך",
               "search": "חיפוש (מה שהוסיף לזמן)", "vector": "חיפוש וקטורי", "bm25": "מילות מפתח", "rerank": "Rerank",
               "search_wait": "המתנה לחיפוש המוקדם", "llm": "מודל התשובה", "repair": "ניסיון תיקון",
               "topic": "בדיקת נושא"}
FEEDBACK = {"up": "👍", "down": "👎", None: "בלי משוב"}


def short(text: str, n: int = 110) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[:n] + "..."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default=str(conversation_log.LOG_FILE))
    parser.add_argument("--since", help="רק מהתאריך הזה, למשל 2026-10-20 או \"2026-10-20 18:00\"")
    parser.add_argument("--cloud", action="store_true", help="היומן המשותף בענן (Upstash) - שני האתרים")
    parser.add_argument("--site", help="רק אתר אחד: streamlit-cloud / render / local")
    parser.add_argument("--legacy", choices=["only", "without"],
                        help="מדידת לפני/אחרי: only - רק שאלות עם ?legacy=1, without - בלעדיהן")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    entries = [e for e in conversation_log.read_log(Path(args.log), cloud=args.cloud)
               if (not args.since or e["time"] >= args.since) and (not args.site or e.get("site") == args.site)
               and (args.legacy is None or bool(e.get("legacy")) == (args.legacy == "only"))]
    if args.cloud:
        args.log = "Upstash (log)"
    if not entries:
        print(f"אין שיחות ביומן ({args.log}).")
        return
    n = len(entries)
    print(f"יומן: {args.log}\n{n} שאלות ב-{len({e.get('session') for e in entries})} שיחות, "
          f"{entries[0]['time']} עד {entries[-1]['time']}")

    print("\n=== מה הבוט החליט ===")
    for kind, count in collections.Counter(e["kind"] for e in entries).most_common():
        print(f"   {kind:<12} {count:>4}  ({count / n:.0%})")
    reasons = collections.Counter(e["reason"].split(":")[0] for e in entries if e["kind"] in ("not_on_site", "error"))
    if reasons:
        print("   סיבות ל'לא מופיע באתר' ולתקלות: " + ", ".join(f"{k}: {v}" for k, v in reasons.most_common()))

    print("\n=== משוב ===")
    counts = collections.Counter(e["feedback"] for e in entries)
    print("   " + " | ".join(f"{FEEDBACK[k]}: {counts[k]}" for k in ("up", "down", None)))
    by_kind = collections.defaultdict(collections.Counter)
    for e in entries:
        if e["feedback"]:
            by_kind[e["kind"]][e["feedback"]] += 1
    for kind, c in by_kind.items():
        print(f"   {kind:<12} 👍 {c['up']}  👎 {c['down']}")

    print("\n=== זמנים (שניות, בלי ברכות) ===")
    print(f"   {'שלב':<24}{'שאלות':>7}{'ממוצע':>8}{'p90':>7}{'מקס':>7}")
    timed = [e["timings"] for e in entries if e["kind"] not in ("greeting", "too_long")]
    for stage, count, avg, high, top in stats.stage_table(timed, STAGES):
        print(f"   {STAGE_NAMES.get(stage, stage):<24}{count:>7}{avg:>8.1f}{high:>7.1f}{top:>7.1f}")
    early = collections.Counter(e.get("early_search") or "-" for e in entries if e["kind"] not in ("greeting",))
    print("   חיפוש מוקדם: " + " | ".join(f"{k}: {v}" for k, v in early.most_common())
          + f" | מהמטמון: {sum(1 for e in entries if e.get('cached'))}")
    if args.cloud and (calls := store.calls_this_month()):
        print(f"   קריאות החודש (כל האתרים): Cohere {calls['cohere']} מתוך 1,000 | Voyage {calls['voyage']}")

    print("\n=== מי ענה ===")
    for model, count in collections.Counter(e["llm"] or "- (בלי מודל)" for e in entries).most_common():
        print(f"   {model:<45} {count:>4}")

    follow_ups = [e for e in entries if e.get("standalone")]
    handoffs = sum(1 for e in entries if e.get("handoff"))
    print(f"\nשאלות המשך שנוסחו מחדש: {len(follow_ups)} | הוצע להמשיך עם נציג בוואטסאפ: {handoffs}")

    if down := [e for e in entries if e["feedback"] == "down"]:
        print("\n=== קיבלו 👎 - לבדיקה ===")
        for e in down:
            print(f"   [{e['time']}] {e['question']}  ->  {e['kind']} ({e['reason']})\n      {short(e.get('answer'))}")

    if missing := [e for e in entries if e["kind"] == "not_on_site"]:
        print(f"\n=== 'לא מופיע באתר' - מה חיפשו ({len(missing)}, האחרונות) ===")
        for e in missing[-15:]:
            print(f"   {e['question']}" + (f"  (כלומר: {e['standalone']})" if e.get("standalone") else ""))


if __name__ == "__main__":
    main()
