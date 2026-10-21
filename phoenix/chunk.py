"""ניקוי העמודים שנאספו וחיתוך לקטעים (chunks).

הרצה:  .venv\\Scripts\\python -m phoenix.chunk

קורא את data/pages/*.json וכותב data/chunks.jsonl - קטע בכל שורה, עם הכתובת,
כותרת העמוד, הכותרת בתוך העמוד ותאריך האיסוף (בלי כתובת אין ציטוט מקור).

ניקוי (לפני החיתוך):
  1. בכתבות - חותכים מההצהרה המשפטית "החברה עושה מאמצים..." והלאה:
     שם מתחילות "כתבות נוספות" עם תקצירים של כתבות אחרות.
  2. מוחקים סעיפים שלמים של תבנית האתר ("כל הדרכים לדבר איתנו", "עוד בנושא").
  3. מוחקים שורות שחוזרות בהרבה עמודים: תפריטים, כפתורים, הצהרות, באנר עוגיות.
     תוכן אמיתי שחוזר (למשל תהליך הטיפול בתביעה) מופיע בעד ~12 עמודים, לכן הסף 14.
     בעמודי יצירת קשר לא מוחקים שורות עם מספרים או מיילים - זה בדיוק המידע שם.
  4. סעיף שלם (כותרת + תוכן) שחוזר זהה ב-5 עמודים ומעלה נמחק מכל העמודים. בפועל
     אלה כרטיסי "כתבות מומלצות" (כותרת + תקציר) - הכתבה המלאה נמצאת בעמוד משלה,
     ועותק של הכרטיס בעמוד הראשי היה גורם לציטוט של העמוד הלא נכון.
  5. עמודים שהטקסט שלהם זהה נשמרים פעם אחת (בכתובת הקצרה מביניהם).
  6. עמודים מחוץ ל-www.fnx.co.il (הפניה לתת-אתר, למשל דף שגיאה) לא נכנסים.

חיתוך:
  לפי הכותרות בעמוד; סעיפים קטנים מתאחדים עד ~800 מילים, סעיף ארוך מדי נחתך
  לפי משפטים. כל קטע (חוץ מהראשון בעמוד) מתחיל במשפט-שניים מסוף הקטע הקודם.
"""
import collections
import hashlib
import json
import random
import re
import sys
from dataclasses import asdict, dataclass, field
from urllib.parse import urlparse

from . import config
from .sitemap import categories

DATA_DIR = config.ROOT / "data"
CHUNKS_FILE = DATA_DIR / "chunks.jsonl"

MIN_WORDS, TARGET_WORDS, MAX_WORDS = 300, 500, 800
OVERLAP_MAX_WORDS = 50

ARTICLE_DISCLAIMER = "החברה עושה מאמצים בכדי לוודא"
DROP_SECTIONS = {"כל הדרכים לדבר איתנו", "עוד בנושא", "נשמח לשוחח"}
BOILERPLATE_MIN_PAGES = 14
SHORT_LINE_WORDS, SHORT_LINE_MIN_PAGES = 4, 6
REPEATED_SECTION_MIN_PAGES = 5
PDF_HEADER_MIN_PAGES = 3  # שורה שחוזרת ב-3 עמודים ומעלה של אותו מסמך PDF - כותרת עליונה/תחתונה


def is_pdf(url: str) -> bool:
    return ".pdf#page=" in url
MIN_PAGE_WORDS = 15

MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
MD_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|>~])")  # Firecrawl מחזיר \- \_ \* וכו'
HTML_BREAK = re.compile(r"<br\s*/?>", re.IGNORECASE)
SITE_SUFFIX = re.compile(r"\s*[|\-–]\s*הפניקס\s*$")  # "... | הפניקס" / "... - הפניקס"
HEADING = re.compile(r"^(#{1,6})\s*(.*)$")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
# תפריט נגישות, מפרידי טבלה של Markdown, תוויות של סרטונים, קישורי "למידע נוסף",
# ושמות של iframe (העמוד עצמו נטען בתוך iframe ואין בו טקסט)
JUNK = re.compile(r"ניגודיות עדינה|^[\s|:\-]+$|^סרטון סרטון|^(למידע|לפרטים) נוס(ף|פים|פות) על |iframe\\?_")
SITE_HOST = "www.fnx.co.il"


@dataclass
class Section:
    path: list[str]
    lines: list[str] = field(default_factory=list)

    @property
    def words(self) -> int:
        return sum(len(line.split()) for line in self.lines)

    @property
    def only_headings(self) -> bool:
        return all(HEADING.match(line) for line in self.lines)


@dataclass
class Chunk:
    id: str
    url: str
    title: str
    heading: str      # הכותרת בתוך העמוד שבה הקטע מתחיל, למשל "תביעות ביטוח דירה › נזקי צנרת"
    fetched_at: str
    chunk_index: int
    words: int        # מילים בגוף הקטע (בלי שורות הכותרת)
    text: str         # מה שנשמר באינדקס ומוצג למודל: כותרת עמוד, כותרת סעיף, גוף


def words(text: str) -> int:
    return len(text.split())


def clean_title(title: str) -> str:
    return " ".join(SITE_SUFFIX.sub("", title).split("|")[0].split())


def normalize(line: str) -> str:
    line = MD_LINK.sub(r"\1", MD_IMAGE.sub("", line))
    line = HTML_BREAK.sub(" ", MD_ESCAPE.sub(r"\1", line)).replace("**", "")
    return " ".join(line.split())


def repeats_title(line: str, title: str) -> bool:
    """שורה רגילה שהיא רק שם העמוד (לפעמים עם " - הפניקס") - אין בה מידע."""
    return not HEADING.match(line) and SITE_SUFFIX.sub("", line).strip() == title


def content_words(lines: list[str]) -> int:
    """מילים בשורות התוכן - בלי כותרות."""
    return sum(words(line) for line in lines if not HEADING.match(line))


def pre_clean(text: str) -> list[str]:
    """שלבים 1-2: חיתוך סוף כתבה ומחיקת סעיפי תבנית."""
    lines = [line for line in map(normalize, text.split("\n")) if line and not JUNK.search(line)]
    for i, line in enumerate(lines):
        if line.startswith(ARTICLE_DISCLAIMER):
            lines = lines[:i]
            break
    out, skip_level = [], None
    for line in lines:
        m = HEADING.match(line)
        if m:
            level, heading_text = len(m.group(1)), m.group(2).strip()
            if skip_level is not None and level <= skip_level:
                skip_level = None
            if skip_level is None and heading_text in DROP_SECTIONS:
                skip_level = level
                continue
        if skip_level is None:
            out.append(line)
    return out


def is_boilerplate(line: str, pages_with_line: int, contact_page: bool) -> bool:
    """שלב 3: שורה שחוזרת בהרבה עמודים."""
    if HEADING.match(line):
        return False
    if contact_page and re.search(r"\d|@", line):
        return False
    if pages_with_line >= BOILERPLATE_MIN_PAGES:
        return True
    return words(line) <= SHORT_LINE_WORDS and pages_with_line >= SHORT_LINE_MIN_PAGES


def split_sections(lines: list[str]) -> list[Section]:
    sections, stack = [], []  # stack: [(level, text)]
    current = Section(path=[])
    for line in lines:
        m = HEADING.match(line)
        if m:
            level, heading_text = len(m.group(1)), m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading_text))
            if current.lines:
                sections.append(current)
            current = Section(path=[t for _, t in stack if t], lines=[line])
        else:
            current.lines.append(line)
    if current.lines:
        sections.append(current)

    # כותרת בלי תוכן מתחברת לסעיף שאחריה, כדי שלא תישאר "תלויה" בסוף קטע
    merged, carry = [], []
    for sec in sections:
        if sec.only_headings:
            carry.extend(sec.lines)
            continue
        merged.append(Section(sec.path, carry + sec.lines))
        carry = []
    return merged


def sentences(lines: list[str]) -> list[str]:
    return [s for line in lines for s in SENTENCE_END.split(line) if s.strip()]


def split_long(sec: Section) -> list[list[str]]:
    """סעיף ארוך מ-MAX_WORDS -> חלקים של ~TARGET_WORDS, לפי גבולות משפטים."""
    pieces, cur = [], []
    for sent in sentences(sec.lines):
        if cur and words(" ".join(cur + [sent])) > TARGET_WORDS:
            pieces.append(cur)
            cur = []
        while words(sent) > MAX_WORDS:  # "משפט" ענק בלי סימני פיסוק
            head = sent.split()
            pieces.append([" ".join(head[:TARGET_WORDS])])
            sent = " ".join(head[TARGET_WORDS:])
        cur.append(sent)
    if cur:
        pieces.append(cur)
    return pieces


def group_sections(sections: list[Section]) -> list[tuple[list[str], list[str]]]:
    """מאחד סעיפים סמוכים עד MAX_WORDS. מחזיר [(path, lines)]."""
    groups, cur_path, cur, cur_words = [], [], [], 0

    def flush():
        nonlocal cur, cur_words
        if cur:
            groups.append((cur_path, cur))
        cur, cur_words = [], 0

    for sec in sections:
        if sec.words > MAX_WORDS:
            flush()
            groups.extend((sec.path, piece) for piece in split_long(sec))
            continue
        if cur_words + sec.words > MAX_WORDS:
            flush()
        if not cur:
            cur_path = sec.path
        cur.extend(sec.lines)
        cur_words += sec.words
    flush()

    # קטע קטן מאוד מתאחד עם השכן הקודם, אם יחד הם עדיין עד MAX_WORDS
    merged = []
    for path, lines in groups:
        n = sum(words(line) for line in lines)
        if merged and n < MIN_WORDS and sum(words(x) for x in merged[-1][1]) + n <= MAX_WORDS:
            merged[-1] = (merged[-1][0], merged[-1][1] + lines)
        else:
            merged.append((path, lines))
    return merged


def overlap_from(lines: list[str]) -> str:
    """משפט או שניים מסוף הקטע הקודם (עד OVERLAP_MAX_WORDS מילים)."""
    tail = []
    for sent in reversed(sentences([line for line in lines if not HEADING.match(line)])):
        if words(" ".join([sent] + tail)) > OVERLAP_MAX_WORDS or len(tail) == 2:
            break
        tail.insert(0, sent)
    if not tail and lines:  # המשפט האחרון ארוך מדי - לוקחים את סופו
        tail = ["… " + " ".join(lines[-1].split()[-(OVERLAP_MAX_WORDS - 10):])]
    return " ".join(tail)


def chunk_page(url: str, title: str, fetched_at: str, sections: list[Section]) -> list[Chunk]:
    chunks, prev_lines = [], None
    for i, (path, body_lines) in enumerate(group_sections(sections)):
        body = "\n".join(body_lines)
        if prev_lines is not None:
            # החפיפה לא מעבירה את הקטע מעל MAX_WORDS
            room = MAX_WORDS - words(body)
            overlap = overlap_from(prev_lines).split()
            if room > 0 and overlap:
                body = " ".join(overlap[-room:]) + "\n" + body
        heading = " › ".join(path)
        text = "\n".join(x for x in (title, heading) if x) + "\n\n" + body
        # המזהה נגזר מכל הטקסט שעובר הטמעה - כל שינוי (גם בכותרת) = קטע חדש באינדקס
        chunk_id = hashlib.sha1(f"{url}\n{text}".encode()).hexdigest()[:16]
        chunks.append(Chunk(chunk_id, url, title, heading, fetched_at, i, words(body), text))
        prev_lines = body_lines
    return chunks


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    pages = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((DATA_DIR / "pages").glob("*.json"))]
    print(f"עמודים שנאספו: {len(pages)}")
    external = [p["url"] for p in pages if urlparse(p["url"]).netloc != SITE_HOST]
    pages = [p for p in pages if urlparse(p["url"]).netloc == SITE_HOST]
    pages.sort(key=lambda p: len(p["url"]))  # בכפילויות - הכתובת הקצרה נשמרת

    pre = {p["url"]: pre_clean(p["text"]) for p in pages}
    # שורות תבנית - רק לפי עמודי האתר. עמוד PDF (תביעות, תקנון): עשר טבלאות המועדים כמעט זהות, ושורה כמו "30 ימים"
    # חוזרת בהן - אבל היא העובדה עצמה, לא תבנית. ב-PDF נמחקות רק שורות שחוזרות בהרבה עמודים של אותו מסמך (כותרות).
    df = collections.Counter(line for url, lines in pre.items() if not is_pdf(url) for line in set(lines))
    doc_df = collections.defaultdict(collections.Counter)
    for url, lines in pre.items():
        if is_pdf(url):
            doc_df[url.split("#")[0]].update(set(lines))
    contact = {p["url"]: "contact" in categories(p["url"]) for p in pages}

    removed, title_repeats = collections.Counter(), 0
    cleaned, seen_text, duplicates, empty = {}, {}, [], []
    for p in pages:
        url, title = p["url"], clean_title(p["title"])
        keep = []
        for line in pre[url]:
            if is_pdf(url):
                if doc_df[url.split("#")[0]][line] >= PDF_HEADER_MIN_PAGES:
                    removed[line] += 1
                    continue
            elif is_boilerplate(line, df[line], contact[url]):
                removed[line] += 1
                continue
            if repeats_title(line, title):
                title_repeats += 1
            else:
                keep.append(line)
        digest = hashlib.sha1("\n".join(keep).encode()).hexdigest()
        if digest in seen_text:
            duplicates.append((url, seen_text[digest]))
        elif content_words(keep) < MIN_PAGE_WORDS:
            empty.append(url)
        else:
            seen_text[digest] = url
            cleaned[url] = keep

    sections = {url: split_sections(lines) for url, lines in cleaned.items()}
    section_pages = collections.Counter(  # סעיפים חוזרים - רק בעמודי האתר (ב-PDF הטבלאות הדומות הן התוכן)
        sig for url, secs in sections.items() if not is_pdf(url) for sig in {"\n".join(sec.lines) for sec in secs})
    repeated = collections.defaultdict(list)  # סעיף -> כל העמודים שבהם הופיע
    for url in sections:
        keep = []
        for sec in sections[url]:
            sig = "\n".join(sec.lines)
            if not is_pdf(url) and section_pages[sig] >= REPEATED_SECTION_MIN_PAGES:
                repeated[sig].append(url)
            else:
                keep.append(sec)
        sections[url] = keep

    chunks, seen_body = [], set()
    for p in pages:
        if p["url"] not in sections:
            continue
        for c in chunk_page(p["url"], clean_title(p["title"]), p["fetched_at"], sections[p["url"]]):
            body_hash = hashlib.sha1(c.text.split("\n\n", 1)[-1].encode()).hexdigest()
            if body_hash not in seen_body:
                seen_body.add(body_hash)
                chunks.append(c)

    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    sizes = [c.words for c in chunks]
    print(f"עמודים מחוץ ל-{SITE_HOST} שדולגו: {len(external)} {external}")
    print(f"עמודים כפולים (אותו טקסט בכתובת אחרת) שדולגו: {len(duplicates)}")
    for url, kept in duplicates:
        print(f"   {url}  =  {kept}")
    print(f"שורות שהן רק שם העמוד שנמחקו: {title_repeats}")
    print(f"עמודים בלי תוכן אחרי ניקוי (פחות מ-{MIN_PAGE_WORDS} מילים מלבד כותרות): {len(empty)}")
    for url in empty:
        print(f"   {url}")
    print(f"שורות תבנית שנמחקו: {len(removed)} שורות שונות, {sum(removed.values())} מופעים. הנפוצות:")
    for line, n in removed.most_common(8):
        print(f"   {n:4} עמודים | {line[:90]}")
    print(f"סעיפים שחוזרים זהים ב-{REPEATED_SECTION_MIN_PAGES}+ עמודים (נמחקו מכל העמודים): {len(repeated)} סעיפים, "
          f"{sum(len(urls) for urls in repeated.values())} מופעים:")
    for sig, urls in repeated.items():
        print(f"   {len(urls):2} עמודים | {sig.split(chr(10))[0][:70]}")
    print(f"\nעמודים באינדקס: {len(cleaned)}")
    print(f"קטעים: {len(chunks)}  ->  {CHUNKS_FILE.relative_to(config.ROOT)}")
    print(f"מילים לקטע: ממוצע {sum(sizes) / len(sizes):.0f}, חציון {sorted(sizes)[len(sizes) // 2]}, "
          f"מינימום {min(sizes)}, מקסימום {max(sizes)}")
    print(f"   {sum(s < MIN_WORDS for s in sizes)} קצרים מ-{MIN_WORDS} (עמודים קצרים שאין מה לאחד איתם; "
          f"מתוכם {sum(s < 50 for s in sizes)} מתחת ל-50 מילים), "
          f"{sum(MIN_WORDS <= s <= MAX_WORDS for s in sizes)} בטווח {MIN_WORDS}-{MAX_WORDS}, "
          f"{sum(s > MAX_WORDS for s in sizes)} ארוכים מ-{MAX_WORDS}")

    print("\nהקטעים עם הכי מעט תוכן (מילים מלבד כותרות):")
    bodies = {c.id: c.text.split("\n\n", 1)[-1].split("\n") for c in chunks}
    for c in sorted(chunks, key=lambda c: content_words(bodies[c.id]))[:8]:
        print(f"   {content_words(bodies[c.id]):3} | {c.url.replace('https://' + SITE_HOST, '')} | "
              f"{' '.join(' '.join(bodies[c.id]).split())[:80]}")

    line_df = collections.Counter(
        line for c in chunks for line in set(c.text.split("\n\n", 1)[-1].split("\n")) if not HEADING.match(line))
    print("\nבדיקה: השורות שחוזרות בהכי הרבה קטעים אחרי הניקוי:")
    for line, n in line_df.most_common(5):
        print(f"   {n:3} קטעים | {line[:90]}")


if __name__ == "__main__":
    main()
