"""הורדת העמודים החשובים ושמירתם כטקסט נקי.

הרצה:  .venv\\Scripts\\python -m phoenix.crawl            (ממשיך מאיפה שעצר)
       .venv\\Scripts\\python -m phoenix.crawl --refresh  (מוריד הכול מחדש)
       .venv\\Scripts\\python -m phoenix.crawl --urls data/urls_all.txt  (כל האתר; יורדים רק עמודים שעוד אין)

קורא את data/urls_important.txt (או --urls) ושומר כל עמוד ל-data/pages/<שם>.json
עם הכתובת, הכותרת, הטקסט, תאריך האיסוף ודרך האיסוף. עמוד ריק או בלי תוכן (נבנה ב-JavaScript) יורד שוב דרך
Firecrawl ואז Jina (fetch.py). כתובת שמפנה (301) לאחרת נרשמת ב-redirects.json.

כל כישלון נרשם ב-data/crawl_failed.txt: כתובת, סוג (404 / 418 / http-NNN / timeout / empty / other) ופירוט. הקובץ
נשמר בין הרצות (כתובת שהצליחה יוצאת ממנו). יותר מ-5% כישלונות שאינם 404 אמיתי (אחרי 40 ניסיונות לפחות) - הסריקה
נעצרת ומדווחת, כדי לא לשרוף מכסה של Firecrawl על בעיה שצריך להבין (קוד יציאה 3).
"""
import argparse
import json
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from urllib.parse import urlparse

from tqdm import tqdm

from . import config
from .fallback import AllProvidersFailed
from .fetch import PageGone, fetch_page

DATA_DIR = config.ROOT / "data"
PAGES_DIR = DATA_DIR / "pages"
REDIRECTS_FILE = DATA_DIR / "redirects.json"
FAILED_FILE = DATA_DIR / "crawl_failed.txt"
DELAY_SECONDS = 1.0  # קצב מנומס: הפסקה בין עמוד לעמוד
MAX_BAD_PERCENT, MIN_ATTEMPTS_FOR_STOP = 5.0, 40
SITE_HOST = "www.fnx.co.il"


def page_file(url: str):
    name = re.sub(r"[^\w\-]+", "_", urlparse(url).path.strip("/")) or "index"
    return PAGES_DIR / f"{name}.json"


def failure_kind(errors: list[str]) -> str:
    """סוג הכישלון מתוך שגיאות כל הספקים (ישיר, Firecrawl, Jina): 418 / http-NNN / timeout / empty / other."""
    text = " | ".join(errors)
    if match := re.search(r"\b(418)\b", text):
        return match.group(1)
    if codes := re.findall(r"(?:Client|Server) error '(\d{3})", text):
        return f"http-{codes[0]}"
    if re.search(r"timed? ?out|Timeout|no answer within", text):
        return "timeout"
    if "too little text" in text or "empty" in text:
        return "empty"
    return "other"


def load_failed() -> dict[str, str]:
    if not FAILED_FILE.exists():
        return {}
    out = {}
    for line in FAILED_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            url, _, rest = line.partition("\t")
            out[url] = rest if "\t" in rest else f"{'404' if rest == 'gone' else 'other'}\t{rest}"
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls", default=str(DATA_DIR / "urls_important.txt"))
    parser.add_argument("--refresh", action="store_true", help="להוריד מחדש גם עמודים שכבר נשמרו")
    parser.add_argument("--workers", type=int, default=1, help="כמה עמודים במקביל (בקשה לאתר בשנייה לכל היותר)")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    PAGES_DIR.mkdir(parents=True, exist_ok=True)

    urls = open(args.urls, encoding="utf-8").read().split()
    # כתובות ישנות שמפנות (301) לעמוד אחר: נשמרות כאן, והעמוד נשמר רק פעם אחת - בכתובת הסופית
    redirects = {} if args.refresh or not REDIRECTS_FILE.exists() else json.loads(REDIRECTS_FILE.read_text(encoding="utf-8"))
    todo = [u for u in urls if args.refresh or not (u in redirects or page_file(u).exists())]
    print(f"{len(urls)} כתובות, {len(urls) - len(todo)} כבר נשמרו, מורידים {len(todo)}")

    failed = load_failed()
    via, saved_now, duplicates, offsite, attempts, bad, stopped = Counter(), set(), 0, 0, 0, 0, False

    # בכמה תהליכונים במקביל (--workers): ההמתנה בין בקשות לאתר היא משותפת לכולם (בקשה אחת בשנייה לכל היותר), והחלק
    # האיטי - Firecrawl לעמוד שנבנה ב-JavaScript - רץ במקביל. כך סריקה של אלף עמודים לא לוקחת שעתיים.
    gate, next_slot = threading.Lock(), [0.0]

    def fetch(url: str):
        with gate:
            now = time.monotonic()
            time.sleep(max(next_slot[0] - now, 0.0))
            next_slot[0] = max(now, next_slot[0]) + DELAY_SECONDS
        try:
            return url, fetch_page(url), None
        except (PageGone, AllProvidersFailed) as exc:
            return url, None, exc

    pool = ThreadPoolExecutor(max_workers=args.workers)
    try:
        for url, page, exc in tqdm(pool.map(fetch, todo), total=len(todo), unit="page"):
            attempts += 1
            if page:
                failed.pop(url, None)
                if page.url != url:
                    redirects[url] = page.url
                if urlparse(page.url).netloc != SITE_HOST:
                    offsite += 1  # הפניה לדומיין אחר (למשל phoenixfinancial.co.il): נרשמת ב-redirects.json, העמוד לא נשמר
                elif page.url in saved_now:
                    duplicates += 1
                else:
                    page_file(page.url).write_text(json.dumps(asdict(page), ensure_ascii=False, indent=1), encoding="utf-8")
                    saved_now.add(page.url)
                    via[page.via] += 1
            elif isinstance(exc, PageGone):
                failed[url] = "404\tgone"
            else:
                kind = failure_kind(exc.errors)
                failed[url] = f"{kind}\t{' | '.join(exc.errors)[:300]}"
                bad += kind != "404"
            if attempts >= MIN_ATTEMPTS_FOR_STOP and 100 * bad / attempts > MAX_BAD_PERCENT:
                stopped = True
                break
    finally:  # גם בעצירה או בקריסה: מה שנאסף נשמר, והכישלונות והפניות נרשמים
        pool.shutdown(wait=False, cancel_futures=True)
        REDIRECTS_FILE.write_text(json.dumps(redirects, ensure_ascii=False, indent=1), encoding="utf-8")
        FAILED_FILE.write_text("".join(f"{u}\t{why}\n" for u, why in sorted(failed.items())), encoding="utf-8")

    kinds = Counter(why.split("\t")[0] for why in failed.values())
    print(f"\nנשמרו: {sum(via.values())} (לפי מקור: {dict(via)})")
    print(f"הפניות לכתובת אחרת: {len(redirects)}, מתוכן {duplicates} כפילויות שדולגו ו-{offsite} לדומיין אחר (לא נשמרו)")
    print(f"נכשלו (בקובץ, כולל הרצות קודמות): {len(failed)} - לפי סוג: {dict(kinds)}" +
          (" - פירוט ב-data/crawl_failed.txt" if failed else ""))
    print(f"סה״כ בתיקייה: {len(list(PAGES_DIR.glob('*.json')))} עמודים")
    if stopped:
        print(f"\nנעצר: {bad} מתוך {attempts} ניסיונות ({100 * bad / attempts:.1f}%) נכשלו בסיבה שאינה 404 אמיתי "
              f"(מעל {MAX_BAD_PERCENT:.0f}%). ראו data/crawl_failed.txt")
        sys.exit(3)


if __name__ == "__main__":
    main()
