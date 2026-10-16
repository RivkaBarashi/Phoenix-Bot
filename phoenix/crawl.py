"""הורדת העמודים החשובים ושמירתם כטקסט נקי.

הרצה:  .venv\\Scripts\\python -m phoenix.crawl            (ממשיך מאיפה שעצר)
       .venv\\Scripts\\python -m phoenix.crawl --refresh  (מוריד הכול מחדש)

קורא את data/urls_important.txt ושומר כל עמוד ל-data/pages/<שם>.json
עם הכתובת, הכותרת, הטקסט, תאריך האיסוף ודרך האיסוף.
"""
import argparse
import json
import re
import sys
import time
from collections import Counter
from dataclasses import asdict
from urllib.parse import urlparse

from tqdm import tqdm

from . import config
from .fallback import AllProvidersFailed
from .fetch import PageGone, fetch_page

DATA_DIR = config.ROOT / "data"
PAGES_DIR = DATA_DIR / "pages"
REDIRECTS_FILE = DATA_DIR / "redirects.json"
DELAY_SECONDS = 1.0  # קצב מנומס: הפסקה בין עמוד לעמוד


def page_file(url: str):
    name = re.sub(r"[^\w\-]+", "_", urlparse(url).path.strip("/")) or "index"
    return PAGES_DIR / f"{name}.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls", default=str(DATA_DIR / "urls_important.txt"))
    parser.add_argument("--refresh", action="store_true", help="להוריד מחדש גם עמודים שכבר נשמרו")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    PAGES_DIR.mkdir(parents=True, exist_ok=True)

    urls = open(args.urls, encoding="utf-8").read().split()
    # כתובות ישנות שמפנות (301) לעמוד אחר: נשמרות כאן, והעמוד נשמר רק פעם אחת - בכתובת הסופית
    redirects = {} if args.refresh or not REDIRECTS_FILE.exists() else json.loads(REDIRECTS_FILE.read_text(encoding="utf-8"))
    todo = [u for u in urls if args.refresh or not (u in redirects or page_file(u).exists())]
    print(f"{len(urls)} כתובות, {len(urls) - len(todo)} כבר נשמרו, מורידים {len(todo)}")

    via, failed, saved_now, duplicates = Counter(), [], set(), 0
    for url in tqdm(todo, unit="page"):
        try:
            page = fetch_page(url)
            if page.url != url:
                redirects[url] = page.url
            if page.url in saved_now:
                duplicates += 1
            else:
                page_file(page.url).write_text(json.dumps(asdict(page), ensure_ascii=False, indent=1), encoding="utf-8")
                saved_now.add(page.url)
                via[page.via] += 1
        except PageGone:
            failed.append(f"{url}\tgone")
        except AllProvidersFailed as exc:
            failed.append(f"{url}\t{' | '.join(exc.errors)[:300]}")
        time.sleep(DELAY_SECONDS)

    REDIRECTS_FILE.write_text(json.dumps(redirects, ensure_ascii=False, indent=1), encoding="utf-8")
    (DATA_DIR / "crawl_failed.txt").write_text("\n".join(failed) + "\n" if failed else "", encoding="utf-8")
    print(f"\nנשמרו: {sum(via.values())} (לפי מקור: {dict(via)})")
    print(f"הפניות לכתובת אחרת: {len(redirects)}, מתוכן {duplicates} כפילויות שדולגו")
    print(f"נכשלו: {len(failed)}" + (" - פירוט ב-data/crawl_failed.txt" if failed else ""))
    print(f"סה״כ בתיקייה: {len(list(PAGES_DIR.glob('*.json')))} עמודים")


if __name__ == "__main__":
    main()
