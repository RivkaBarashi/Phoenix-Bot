"""הורדת ה-sitemap של אתר הפניקס ובחירת העמודים החשובים.

הרצה:  .venv\\Scripts\\python -m phoenix.sitemap

יוצר:
  data/urls.txt            - כל הכתובות ב-sitemap (כולל sitemaps פנימיים)
  data/urls_important.txt  - רק תביעות, שאלות נפוצות, דירה, רכב, בריאות ויצירת קשר,
                             בלי עמודים ש-robots.txt אוסר לסרוק
"""
import re
import sys
from urllib.parse import unquote, urlparse
from xml.etree import ElementTree

import httpx

from . import config

SITE = "https://www.fnx.co.il"
DATA_DIR = config.ROOT / "data"
NS = "{http://www.sitemaps.org/schemas/sitemap/0.9}"

# קטגוריה -> תבנית על הנתיב. עמוד חשוב אם הוא מתאים לפחות לקטגוריה אחת.
CATEGORIES = {
    "claims": r"claim",  # כל אזור /claims-customer-service/ ועוד עמודי תביעה בודדים
    "faq": r"faq|questions",
    "apartment": r"apartment|appartment",  # באתר יש את שני האיותים
    "car": r"car-insurance",
    "health": r"health|dental",
    # לאן להפנות כשהתשובה לא באתר או כשמדובר בפוליסה אישית
    "contact": r"^/customer-service/$|contact|be-in-touch|call-back|/chat/|service-charter"
               r"|public-inquiries|your-request-for-info",
}
# לא שימושי ללקוחה: אנגלית, תגיות, הודעות על תביעות ייצוגיות, סטטיסטיקות
EXCLUDE = r"^/en/|^/tags/|^/404|/class-actions/|/claims-statistical-info/"


def fetch_sitemap_urls(url: str, seen: set[str] | None = None) -> list[str]:
    """כל הכתובות ב-sitemap. אם זה sitemapindex - נכנסים לכל sitemap פנימי."""
    seen = set() if seen is None else seen
    if url in seen:
        return []
    seen.add(url)
    r = httpx.get(url, headers={"User-Agent": config.USER_AGENT}, timeout=config.HTTP_TIMEOUT, follow_redirects=True)
    r.raise_for_status()
    root = ElementTree.fromstring(r.content.strip())
    # רק <loc> ישיר של <url>/<sitemap> - לא כתובות תמונות (image:loc)
    locs = [loc.text.strip() for entry in root for loc in entry.findall(f"{NS}loc") if loc.text]
    if root.tag == f"{NS}sitemapindex":
        return [u for child in locs for u in fetch_sitemap_urls(child, seen)]
    return locs


def robots_disallowed() -> list[re.Pattern]:
    """כללי Disallow מ-robots.txt עבור כל הבוטים (User-agent: *)."""
    r = httpx.get(f"{SITE}/robots.txt", headers={"User-Agent": config.USER_AGENT}, timeout=config.HTTP_TIMEOUT)
    r.raise_for_status()
    patterns, applies = [], False
    for line in r.text.splitlines():
        key, _, value = line.partition(":")
        key, value = key.strip().lower(), value.split("#")[0].strip()
        if key == "user-agent":
            applies = value == "*"
        elif key == "disallow" and applies and value:
            patterns.append(re.compile(re.escape(value).replace(r"\*", ".*").replace(r"\$", "$")))
    return patterns


def categories(url: str) -> list[str]:
    path = unquote(urlparse(url).path).lower()
    if re.search(EXCLUDE, path):
        return []
    return [name for name, pattern in CATEGORIES.items() if re.search(pattern, path)]


def is_allowed(url: str, disallowed: list[re.Pattern]) -> bool:
    parts = urlparse(url)
    target = parts.path + (f"?{parts.query}" if parts.query else "")
    return not any(p.match(target) for p in disallowed)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    DATA_DIR.mkdir(exist_ok=True)

    urls = sorted(set(fetch_sitemap_urls(f"{SITE}/sitemap.xml")))
    disallowed = robots_disallowed()
    blocked = [u for u in urls if not is_allowed(u, disallowed)]
    important = [u for u in urls if categories(u) and is_allowed(u, disallowed)]

    (DATA_DIR / "urls.txt").write_text("\n".join(urls) + "\n", encoding="utf-8")
    (DATA_DIR / "urls_important.txt").write_text("\n".join(important) + "\n", encoding="utf-8")

    print(f"data/urls.txt:           {len(urls)} כתובות")
    print(f"data/urls_important.txt: {len(important)} כתובות")
    print(f"  ({len(blocked)} כתובות ב-sitemap חסומות ב-robots.txt - לא נכנסו לרשימה החשובה)")
    print("לפי קטגוריה (עמוד יכול להיות ביותר מאחת):")
    for name in CATEGORIES:
        print(f"  {name:10} {sum(name in categories(u) for u in important)}")


if __name__ == "__main__":
    main()
