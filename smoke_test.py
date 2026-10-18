"""בדיקת עשן: 5 שאלות מייצגות דרך אותה פונקציה שהצ'אט קורא לה (answer), עם אותן מגבלות זמן.

רגילה, אישית, לא באתר, ברכה ולא קשורה. בודקת שהמערכת כולה עובדת - מפתחות, מכסות, אינדקס, Rerank
ומודל - בלי להריץ את כל הבדיקות. המודל נשאל באמת ולא מהמטמון, כדי לגלות מפתח או מכסה שנגמרו
(--cached: להשתמש בתשובות שמורות, בלי קריאות למודל). כמו הבדיקות - שרשרת המודלים הרגילה, Gemini קודם
(CHAT_LLM_PROVIDERS משפיע רק על הממשק).

הרצה:  .venv\\Scripts\\python smoke_test.py
"""
import argparse
import sys
import time

from phoenix import eval_history, llm, usage
from phoenix.answer import answer

CASES = [  # (סוג, שאלה, התשובה שמצפים לה)
    ("רגילה", "מה המספר של מוקד תביעות רכב ורכוש, ומתי הוא פתוח?", "answer"),
    ("אישית", "האם הביטוח שלי כולל רכב חלופי?", "personal"),
    ("לא באתר", "אפשר לבטח אצלכם קלנועית?", "not_on_site"),
    ("ברכה", "שלום", "greeting"),
    ("לא קשורה", "מה מזג האוויר מחר?", "off_topic"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached", action="store_true", help="תשובות שמורות מהמטמון, בלי קריאות למודל")
    parser.add_argument("--providers", help="רק מודלי השפה האלה, למשל groq,openrouter - איך הבוט עונה כש-Gemini חסום")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if not args.cached:
        llm.CACHE_MODE = "write"  # שואלים את המודל באמת, ושומרים את התשובה
    if args.providers:
        only = [p.strip() for p in args.providers.split(",")]
        llm.PROVIDERS[:] = [p for p in llm.PROVIDERS if p.name.split(":")[0] in only]
        print(f"רק המודלים: {', '.join(p.name for p in llm.PROVIDERS)}\n")
    failed = 0
    for label, question, expected in CASES:
        start = time.perf_counter()
        try:
            a = answer(question)
            got, detail, text = a.kind, f"{a.reason} | {a.llm or '-'}", a.text
        except Exception as exc:  # בדיקת עשן מדווחת על תקלה, היא לא קורסת בגללה
            got, detail, text = "exception", f"{type(exc).__name__}: {exc}", ""
        ok = got == expected
        failed += not ok
        print(f"{'✓' if ok else '✗'} {label:<9} {time.perf_counter() - start:5.1f} ש'  {got:<11} ({detail})  {question}")
        if not ok:
            print(f"      ציפינו ל-{expected}. התשובה: {' '.join(text.split())[:200]}")
    print(f"\n{len(CASES) - failed}/{len(CASES)} עברו")
    print("\n" + usage.report())
    selection = ("cached" if args.cached else "live") + (f" providers={args.providers}" if args.providers else "")
    eval_history.append(script="smoke_test", selection=selection, n=len(CASES),
                        smoke_passed=f"{len(CASES) - failed}/{len(CASES)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
