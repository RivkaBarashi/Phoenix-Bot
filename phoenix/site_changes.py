"""בדיקת שינויים שבועית באתר - בלי לבנות את האינדקס מחדש ובלי לכתוב שום דבר.

לכל עמוד שנאסף (data/pages) מורידים את הגרסה של היום באותה דרך (חילוץ הטקסט של fetch.py), ובודקים מה שחשוב לבוט:
  - הוסר: משפטים שבאינדקס (data/chunks.jsonl) ולא מופיעים יותר בעמוד - מידע שהבוט עלול לצטט והשתנה.
  - נוסף: שורות תוכן חדשות (5 מילים ומעלה) שלא מופיעות באף עמוד שנאסף - מידע שהבוט עוד לא מכיר.
בלוקים מתחלפים ("כתבות נוספות" שמשתנות בכל טעינה) לא נספרים: הם לא באינדקס, והשורות שלהם קיימות בעמודים אחרים
(נבדק 20/10: בהשוואת הטקסט כולו 4 מתוך 5 עמודים "השתנו" ב-7%-40%, כולם מהבלוקים האלה).
התוצאה: דוח Markdown - אילו עמודים השתנו, אילו נעלמו (404) ואילו לא נבדקו (תקלה). האינדקס נבנה מחדש רק בהחלטה
של מפתחת הפרויקט (README: "רענון התוכן מהאתר"), אחרי שראתה מה השתנה.

הרצה:  python -m phoenix.site_changes [--out report.md] [--limit N]
ב-GitHub Actions (site-changes.yml) הדוח נכתב לסיכום של הריצה, ואם יש שינויים - נפתח Issue. בלי מפתחות: רק
הורדה ישירה של העמודים (Firecrawl - רק אם יש מפתח).
"""
import argparse
import glob
import json
import re
import sys
import time

from . import config
from .fallback import AllProvidersFailed
from .fetch import PageGone, fetch_page

DELAY_SECONDS = 1.0     # קצב מנומס, כמו באיסוף
MIN_REMOVED = 0.03      # פחות מ-3% מהמשפטים שבאינדקס (תאריך, ניסוח קטן) - לא נחשב שינוי
MIN_WORDS = 5           # שורה קצרה יותר (כותרת, כפתור) לא נחשבת תוכן
# שורות חדשות מסמנות עמוד רק מ-5 ומעלה: בלוק "כתבה נוספת" לכתבה שלא נאספה מוסיף 1-4 שורות (נבדק 20/10, 12 עמודים)
MIN_ADDED = 5
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MARKUP = re.compile(r"[#*_>|`]+")
_SPACES = re.compile(r"\s+")


def norm(line: str) -> str:
    return _SPACES.sub(" ", _MARKUP.sub(" ", _LINK.sub(r"\1", line))).strip()


def content_lines(text: str, skip: int = 0) -> list[str]:
    lines = [norm(line) for line in text.splitlines()[skip:]]
    return list(dict.fromkeys(line for line in lines if len(line.split()) >= MIN_WORDS))


def indexed_lines() -> dict[str, list[str]]:
    """לכל עמוד - המשפטים שבאינדקס (בלי שתי השורות הראשונות של כל קטע: כותרת העמוד והכותרת בתוכו)."""
    out: dict[str, list[str]] = {}
    for line in open(config.ROOT / "data" / "chunks.jsonl", encoding="utf-8"):
        chunk = json.loads(line)
        out.setdefault(chunk["url"], []).extend(content_lines(chunk["text"], skip=2))
    return {url: list(dict.fromkeys(lines)) for url, lines in out.items()}


PREFIX = 50  # בלוק "כתבה נוספת" = כותרת של עמוד אחר + תחילת הפתיח שלו, קטועה


def known_text(pages: list[dict]) -> set[str]:
    """מה שכבר נאסף: כל שורות התוכן, תחילת כל שורה (לפתיח קטוע) וכותרות העמודים."""
    lines = {line for page in pages for line in content_lines(page["text"])}
    titles = {norm(re.sub(r"\s*[|\-–]\s*הפניקס\s*$", "", page.get("title", ""))) for page in pages}
    return lines | {line[:PREFIX] for line in lines} | titles


def compare(indexed: list[str], live_text: str, known: set[str]) -> tuple[float, list[str]]:
    """(חלק המשפטים מהאינדקס שכבר לא בעמוד, שורות תוכן חדשות שלא מופיעות באף עמוד שנאסף)."""
    joined = " ".join(norm(line) for line in live_text.splitlines())
    removed = sum(1 for line in indexed if line not in joined) / len(indexed) if indexed else 0.0
    added = [line for line in content_lines(live_text) if line not in known and line[:PREFIX] not in known]
    return removed, added


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", help="קובץ Markdown לדוח (ועוד קובץ <out>.count עם מספר העמודים שהשתנו)")
    parser.add_argument("--limit", type=int, help="רק N העמודים הראשונים (לבדיקה מהירה)")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    pages = [json.load(open(path, encoding="utf-8"))
             for path in sorted(glob.glob(str(config.ROOT / "data" / "pages" / "*.json")))]
    known = known_text(pages)
    indexed = indexed_lines()
    pages = pages[: args.limit]
    changed, gone, failed = [], [], []
    for i, saved in enumerate(pages, 1):
        try:
            live = fetch_page(saved["url"])
            removed, added = compare(indexed.get(saved["url"], []), live.text, known)
            if removed >= MIN_REMOVED or len(added) >= MIN_ADDED:
                changed.append((removed, len(added), saved["url"], saved.get("fetched_at", "")[:10],
                                added[0][:80] if added else ""))
        except PageGone:
            gone.append(saved["url"])
        except AllProvidersFailed as exc:
            failed.append((saved["url"], " | ".join(exc.errors)[:150]))
        print(f"\r{i}/{len(pages)}", end="", flush=True)
        time.sleep(DELAY_SECONDS)
    print()

    lines = [f"# שינויים באתר מאז האיסוף ({time.strftime('%Y-%m-%d')})", "",
             f"נבדקו {len(pages)} עמודים: **{len(changed)} השתנו**, {len(gone)} נעלמו, {len(failed)} לא נבדקו.",
             "", "האינדקס לא נבנה מחדש. כדי לעדכן - README, \"רענון התוכן מהאתר\" (בהחלטה שלך).", ""]
    if changed:
        lines += ["## השתנו", "", "| הוסר מהאינדקס | שורות חדשות | עמוד | נאסף | דוגמה לשורה חדשה |",
                  "|---|---|---|---|---|"]
        lines += [f"| {removed:.0%} | {added} | {url} | {when} | {sample} |"
                  for removed, added, url, when, sample in sorted(changed, reverse=True)]
        lines.append("")
    if gone:
        lines += ["## נעלמו (404)", ""] + [f"- {url}" for url in gone] + [""]
    if failed:
        lines += ["## לא נבדקו (תקלה בהורדה)", ""] + [f"- {url} - {error}" for url, error in failed] + [""]
    report = "\n".join(lines)
    print(report)
    if args.out:  # קוד יציאה 0 תמיד: שינוי באתר הוא מידע, לא כישלון. ה-workflow מחליט לפי המספר אם לפתוח Issue
        open(args.out, "w", encoding="utf-8").write(report)
        open(args.out + ".count", "w").write(str(len(changed) + len(gone)))


if __name__ == "__main__":
    main()
