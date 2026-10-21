"""בדיקה בדפדפן אמיתי מול הקישור הציבורי (Playwright) - כמו שלקוחה משתמשת בבוט.

  python e2e/e2e_cloud.py --url https://<app>.streamlit.app --wake
      פותח את הדף בלי לשאול כלום (בלי קריאה למודלים). Streamlit Cloud ישן אחרי 12 שעות בלי ביקור - כאן הוא מתעורר.
  python e2e/e2e_cloud.py --url ... --smoke
      5 השאלות של smoke_test.py. עובר רק אם 5/5 נכונות ו-p90 של זמן הבוט עד 20 שניות.
  python e2e/e2e_cloud.py --url ... --questions data/questions.csv --out answers.csv
      הריצה בדפדפן על קובץ שאלות: התשובה, ההחלטה והזמנים של כל שאלה ל-CSV (לשיפוט - judge_cloud.py).
  --legacy   ?legacy=1 - בלי שיפורי הזמנים (מדידת "לפני", סעיף 6ה).  --browser msedge - Edge המותקן (במחשב).

כל שאלה בשיחה חדשה (?dev=1, בלי ?c) - בלי הקשר משיחה קודמת, ובלי מטמון התשובות (במצב מפתחת לא קוראים ממנו).
ההחלטה והזמנים נקראים מהשורה META בפרטים הטכניים. שני זמנים לכל שאלה: של הבוט (timings.total - מה שמגבלת 20
השניות מודדת) ושל הדפדפן (מהשליחה עד שהתשובה המלאה על המסך, כולל ההקלדה ההדרגתית והרשת).
"""
import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path
import time

CHAT_INPUT = 'textarea[data-testid="stChatInputTextArea"]'
META = re.compile(r"^META ")
WAKE_BUTTON = re.compile(r"get this app back up|wake", re.IGNORECASE)
# כמו smoke_test.CASES (בדיקת יחידה מוודאת שהן זהות). כאן בלי לייבא את הבוט: ב-GitHub Actions מותקן רק Playwright.
SMOKE = [
    ("רגילה", "מה המספר של מוקד תביעות רכב ורכוש, ומתי הוא פתוח?", "answer"),
    ("אישית", "האם הביטוח שלי כולל רכב חלופי?", "personal"),
    ("לא באתר", "אפשר לבטח אצלכם קלנועית?", "not_on_site"),
    ("ברכה", "שלום", "greeting"),
    ("לא קשורה", "מה מזג האוויר מחר?", "off_topic"),
]
LIMIT_SECONDS = 20


def p90(values: list[float]) -> float:
    return statistics.quantiles(values, n=10, method="inclusive")[-1] if len(values) > 1 else (values or [0])[0]


def app_frame(page, timeout: float = 240):
    """המסגרת שבה הצ'אט (Streamlit Cloud מציג את האפליקציה בתוך iframe). אפליקציה ישנה - לוחצים "להעיר"."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        for frame in page.frames:
            try:
                if frame.locator(CHAT_INPUT).count():
                    return frame
                wake = frame.get_by_role("button", name=WAKE_BUTTON)
                if wake.count():
                    print("   האתר ישן - מעירים אותו...", flush=True)
                    wake.first.click()
            except Exception:  # noqa: BLE001 - מסגרת שנטענת מחדש באמצע
                pass
        time.sleep(2)
    raise TimeoutError(f"the chat did not load within {timeout:.0f}s")


def open_chat(page, url: str, legacy: bool):
    page.goto(f"{url.rstrip('/')}/?dev=1" + ("&legacy=1" if legacy else ""), wait_until="domcontentloaded")
    return app_frame(page)


def ask(frame, question: str, timeout: float = 120) -> dict:
    metas = frame.get_by_text(META)
    before = metas.count()
    box = frame.locator(CHAT_INPUT)
    box.fill(question)
    start = time.perf_counter()
    box.press("Enter")
    while metas.count() <= before:  # השורה META מופיעה אחרי שהתשובה כולה על המסך
        if time.perf_counter() - start > timeout:
            raise TimeoutError(f"no answer within {timeout:.0f}s")
        time.sleep(0.2)
    wall = time.perf_counter() - start
    meta = json.loads(metas.nth(before).text_content().split("META ", 1)[1])
    text = frame.locator("div.answer").last.text_content() or ""
    return {"question": question, "kind": meta.get("kind"), "reason": meta.get("reason"), "llm": meta.get("llm"),
            "reranker": meta.get("reranker"), "top_score": meta.get("top_score"),
            "early_search": meta.get("early_search"), "cached": meta.get("cached"),
            "sources": " ".join(meta.get("sources") or []),
            "bot_s": round((meta.get("timings") or {}).get("total", 0.0), 2), "wall_s": round(wall, 2),
            "timings": json.dumps(meta.get("timings") or {}, ensure_ascii=False), "answer": " ".join(text.split())}


def run_conversations(url: str, conversations: list[list[str]], browser_name: str, shots: str = "") -> list[dict]:
    """שאלות המשך: כל שיחה - כמה שאלות ברצף באותו צ'אט (כמו בהדגמה של האתגר), צילום מסך בסוף כל שיחה."""
    from playwright.sync_api import sync_playwright
    out = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=browser_name, headless=True) if browser_name else p.chromium.launch()
        page = browser.new_page(viewport={"width": 900, "height": 1400})
        for c, questions in enumerate(conversations, 1):
            frame = open_chat(page, url, False)
            for t, question in enumerate(questions, 1):
                try:
                    row = ask(frame, question)
                except Exception as exc:  # noqa: BLE001
                    row = {"question": question, "kind": "exception", "reason": f"{type(exc).__name__}: {exc}"[:200],
                           "bot_s": 0.0, "wall_s": 0.0, "answer": ""}
                row["conversation"], row["turn"] = c, t
                out.append(row)
                print(f"{c}.{t} {row['kind']:<12} bot {row['bot_s']:5.1f}s ({row['reason']})  {question[:50]}\n"
                      f"      {row['answer'][:160]}", flush=True)
            if shots:
                page.screenshot(path=f"{shots}/conversation_{c}.png", full_page=True)
        browser.close()
    return out


def run(url: str, questions: list[str], legacy: bool, browser_name: str, min_gap: float = 0.0) -> list[dict]:
    """min_gap: שאלה מתחילה לפחות min_gap שניות אחרי הקודמת (קצב של אדם; בלי - מהר ככל האפשר)."""
    from playwright.sync_api import sync_playwright
    out = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=browser_name, headless=True) if browser_name else p.chromium.launch()
        page = browser.new_page()
        started = 0.0
        for n, question in enumerate(questions, 1):
            if min_gap and (wait := started + min_gap - time.time()) > 0:
                time.sleep(wait)
            started = time.time()
            try:
                frame = open_chat(page, url, legacy)  # כל שאלה בשיחה חדשה
                row = ask(frame, question)
            except Exception as exc:  # noqa: BLE001 - שאלה שנכשלה נרשמת, הריצה ממשיכה
                row = {"question": question, "kind": "exception", "reason": f"{type(exc).__name__}: {exc}"[:200],
                       "bot_s": 0.0, "wall_s": 0.0, "answer": ""}
            out.append(row)
            print(f"{n:3} {row['kind']:<12} bot {row['bot_s']:5.1f}s  browser {row['wall_s']:5.1f}s  "
                  f"({row['reason']})  {question[:60]}", flush=True)
        browser.close()
    return out


def wake(url: str, browser_name: str):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=browser_name, headless=True) if browser_name else p.chromium.launch()
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded")
        start = time.perf_counter()
        app_frame(page)
        print(f"הצ'אט עלה תוך {time.perf_counter() - start:.0f} שניות: {url}")
        browser.close()


def summary(rows: list[dict]) -> str:
    bot = [r["bot_s"] for r in rows if r["kind"] not in ("exception", "greeting")]
    wall = [r["wall_s"] for r in rows if r["kind"] not in ("exception", "greeting")]
    if not bot:
        return "אין זמנים"
    return (f"זמן הבוט: ממוצע {statistics.mean(bot):.1f} | p90 {p90(bot):.1f} | מקסימום {max(bot):.1f} ש'  ·  "
            f"בדפדפן: ממוצע {statistics.mean(wall):.1f} | p90 {p90(wall):.1f} | מקסימום {max(wall):.1f} ש'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--wake", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--questions", help="CSV עם עמודה question")
    mode.add_argument("--conversations", help="קובץ טקסט: שאלה בכל שורה, שורה '---' בין שיחות (שאלות המשך באותו צ'אט)")
    parser.add_argument("--shots", default="", help="עם --conversations: תיקייה לצילום מסך של כל שיחה")
    parser.add_argument("--out", help="CSV לתוצאות")
    parser.add_argument("--smoke-csv", help="עם --smoke: CSV של שאלות והתשובה הצפויה (עמודות question, expected) במקום 5 השאלות "
                                            "הקבועות - לבדיקת התהליך בלי קריאות למודלים")
    parser.add_argument("--legacy", action="store_true", help="בלי שיפורי הזמנים - מדידת \"לפני\"")
    parser.add_argument("--browser", default="", help="msedge / chrome - דפדפן מותקן (ברירת מחדל: Chromium של Playwright)")
    parser.add_argument("--min-gap", type=float, default=0.0,
                        help="שניות לפחות בין תחילת שאלה לשאלה (קצב של אדם; 21/10: מכסת Cohere Trial - 20 קריאות לדקה)")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    if args.wake:
        wake(args.url, args.browser)
        return
    cases = SMOKE
    if args.smoke and args.smoke_csv:
        cases = [(r["question"][:20], r["question"], r["expected"])
                 for r in csv.DictReader(open(args.smoke_csv, encoding="utf-8-sig"))]
    if args.conversations:
        blocks = Path(args.conversations).read_text(encoding="utf-8").split("---")
        convs = [[q.strip() for q in b.splitlines() if q.strip()] for b in blocks]
        rows = run_conversations(args.url, [c for c in convs if c], args.browser, args.shots)
    elif args.smoke:
        rows = run(args.url, [q for _, q, _ in cases], args.legacy, args.browser)
        for (label, _, expected), row in zip(cases, rows):
            row["expected"], row["ok"] = expected, row["kind"] == expected
            if not row["ok"]:
                print(f"   ✗ {label}: ציפינו ל-{expected}, התקבל {row['kind']} - {row['answer'][:150]}")
    else:
        rows = run(args.url, [r["question"] for r in csv.DictReader(open(args.questions, encoding="utf-8-sig"))],
                   args.legacy, args.browser, args.min_gap)
    print("\n" + summary(rows))
    if args.out:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()
            csv.DictWriter(f, fieldnames=fields).writerows(rows)
    if args.smoke:
        passed = sum(r["ok"] for r in rows)
        bot = [r["bot_s"] for r in rows if r["kind"] not in ("exception", "greeting")]
        ok = passed == len(cases) and (p90(bot) <= LIMIT_SECONDS if bot else bool(args.smoke_csv))
        print(f"{passed}/{len(cases)} נכונות, p90 של הבוט {p90(bot) if bot else 0:.1f} ש' -> {'עבר' if ok else 'נכשל'}")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
