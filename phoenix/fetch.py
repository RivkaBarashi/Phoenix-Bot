"""הורדת עמוד מהאתר והפיכתו לטקסט נקי, עם גיבוי.

קודם מורידים ישירות (חינם ובלי מכסה). רק אם זה נכשל או שיצא כמעט ריק
(למשל עמוד שנבנה ב-JavaScript) - משתמשים ב-Firecrawl, ואז ב-Jina Reader.
"""
import datetime as dt
from dataclasses import dataclass

import httpx
from bs4 import BeautifulSoup

from . import config
from .fallback import EmptyResult, Provider, StopChain, run_chain

MIN_CHARS = 150  # פחות מזה - כנראה התוכן לא נטען
NOISE_TAGS = ["script", "style", "noscript", "template", "svg", "iframe", "nav", "header", "footer"]
BLOCK_TAGS = ["p", "div", "section", "article", "ul", "ol", "li", "table", "tr", "dt", "dd", "br",
              "h1", "h2", "h3", "h4", "h5", "h6"]


class PageGone(StopChain):
    """העמוד לא קיים (404/410) - אין טעם לנסות ספק אחר."""


@dataclass
class Page:
    url: str            # הכתובת הסופית (אחרי הפניות) - זו שהבוט מצטט
    requested_url: str  # הכתובת שביקשנו (מה-sitemap)
    title: str
    text: str        # Markdown נקי
    fetched_at: str  # מתי נאסף - כדי לדעת מה לרענן
    via: str         # direct / firecrawl / jina


def _enough(text: str | None) -> str:
    if not text or len(text.strip()) < MIN_CHARS:
        raise EmptyResult("too little text")
    return text.strip()


def html_to_text(html: str) -> tuple[str, str]:
    """(כותרת, טקסט) מתוך <main>, עם כותרות (#) ורשימות (-).

    לא משתמשים ב-trafilatura: הוא מוחק טלפונים ומיילים מעמודי יצירת קשר -
    בדיוק המידע שהבוט צריך כדי להפנות לשירות לקוחות.
    """
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.get_text(strip=True) if soup.title else ""
    root = soup.find("main") or soup.body or soup
    for tag in root(NOISE_TAGS):
        tag.decompose()
    for el in root.find_all(BLOCK_TAGS):
        if el.name[0] == "h" and el.name[1:].isdigit():
            el.insert_before("\n" + "#" * int(el.name[1]) + " ")
        elif el.name == "li":
            el.insert_before("\n- ")
        el.insert_after("\n")
    lines, prev = [], None
    for raw in root.get_text(" ").split("\n"):
        line = " ".join(raw.split())
        if line.strip("#- ") and line != prev:
            lines.append(line)
            prev = line
    return title, "\n".join(lines)


def _direct(url):
    r = httpx.get(url, headers={"User-Agent": config.USER_AGENT, "Accept-Language": "he-IL,he"},
                  follow_redirects=True, timeout=30)
    if r.status_code in (404, 410):
        raise PageGone(url)
    r.raise_for_status()
    title, text = html_to_text(r.text)
    return str(r.url), title, _enough(text)


def _firecrawl(url):
    r = httpx.post(
        "https://api.firecrawl.dev/v2/scrape",
        headers={"Authorization": f"Bearer {config.API_KEYS['firecrawl']}"},
        json={"url": url, "formats": ["markdown"], "onlyMainContent": True},
        timeout=90,
    )
    r.raise_for_status()
    data = r.json()["data"]
    meta = data.get("metadata", {})
    return meta.get("url") or meta.get("sourceURL") or url, meta.get("title", ""), _enough(data.get("markdown"))


def _jina(url):
    r = httpx.get(f"https://r.jina.ai/{url}", headers={"Accept": "application/json"}, timeout=60)
    r.raise_for_status()
    data = r.json()["data"]
    return data.get("url") or url, data.get("title", ""), _enough(data.get("content"))


_CALLS = {"direct": _direct, "firecrawl": _firecrawl, "jina": _jina}

PROVIDERS = [
    Provider(name, _CALLS[name], enabled=name != "firecrawl" or bool(config.API_KEYS["firecrawl"]))
    for name in config.FETCH_CHAIN
]


def fetch_page(url: str) -> Page:
    # בלי "מנוחה" לספקים: כשל בעמוד אחד לא אומר שהספק כולו לא עובד.
    (final_url, title, text), via = run_chain("fetch", PROVIDERS, url, retries=2, max_wait=30, cooldown=False)
    return Page(final_url, url, title, text, dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), via)
