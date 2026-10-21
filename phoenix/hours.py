""""פתוח עכשיו" ליד כל הפניה למוקד (22/10, בקשת הבעלים): שעות הפעילות של כל מוקד נקראות מעמוד "צור קשר" שבאתר -
לא מוקלדות בקוד, כך שהן מתעדכנות עם האתר - ומחושב לפי שעון ישראל אם המוקד פתוח עכשיו, ומתי ייפתח.

  "🟢 פתוח עכשיו, עד 16:00" · "🔴 סגור כרגע, ייפתח מחר ב-8:30" · "🔴 סגור כרגע, ייפתח ביום א' ב-8:30"

רק בממשק: הטקסט של התשובה, החיפוש והאימות לא משתנים. שורת שעות שלא מובנת, או מספר שמופיע בכמה מוקדים עם שעות
שונות - בלי סטטוס (עדיף כלום מאשר "פתוח" שגוי). חגים לא מופיעים בעמוד "צור קשר" - לא נלקחים בחשבון.
"""
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

ISRAEL = ZoneInfo("Asia/Jerusalem")
DAYS = "אבגדהוש"  # א = ראשון ... ש = שבת
PYTHON_WEEKDAY = {d: (i + 6) % 7 for i, d in enumerate(DAYS)}  # datetime.weekday(): שני = 0 ... ראשון = 6
ALWAYS = "always"
SEGMENT = re.compile(r"(?:ימים|יום)\s+([א-ש])['׳]?(?:\s*-\s*([א-ש])['׳]?)?\s*,?\s*(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})")
ALWAYS_TEXT = re.compile(r"24\s*שעות\s*ביממה")
NUMBER = re.compile(r"\*?\d[\d\-]{2,}\d\*?")


@dataclass
class Center:
    name: str
    numbers: set[str]
    schedule: object  # ALWAYS, או {יום בשבוע (weekday): [(דקת פתיחה, דקת סגירה)]}


def digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def parse_schedule(text: str):
    """"שעות פעילות: ימים א'-ה', 8:30-16:00" -> {6: [(510, 960)], 0: [...], ...}; "24 שעות ביממה" -> ALWAYS; לא מובן -> None."""
    if ALWAYS_TEXT.search(text):
        return ALWAYS
    schedule: dict[int, list[tuple[int, int]]] = {}
    for first, last, h1, m1, h2, m2 in SEGMENT.findall(text):
        start, end = DAYS.index(first), DAYS.index(last or first)
        if end < start:
            return None
        for day in DAYS[start:end + 1]:
            schedule.setdefault(PYTHON_WEEKDAY[day], []).append((int(h1) * 60 + int(m1), int(h2) * 60 + int(m2)))
    return schedule or None


def parse_centers(contact_text: str) -> list[Center]:
    """כל מוקד בעמוד "צור קשר": כותרת (##), המספרים שמתחתיה, ושורת שעות הפעילות שלו."""
    centers, current = [], None
    for line in contact_text.splitlines():
        if line.startswith("## "):
            current = Center(line[3:].strip(), set(), None)
            centers.append(current)
        elif current is not None:
            if schedule := parse_schedule(line):  # שורת השעות - לא מספר טלפון ("8:30-16:00" הוא לא "3016")
                current.schedule = current.schedule or schedule
                continue
            current.numbers |= {digits(n) for n in NUMBER.findall(line) if len(digits(n)) >= 4}
    return [c for c in centers if c.numbers and c.schedule]


@lru_cache
def _centers() -> tuple[Center, ...]:
    from .answer import contact_text  # כאן ולא למעלה: הממשק טוען את המודול הזה גם בבדיקות בלי האינדקס
    return tuple(parse_centers(contact_text()))


def schedule_for(number: str, centers=None):
    """לוח השעות של מספר (רק אם בכל המוקדים שהוא מופיע בהם השעות זהות)."""
    found = [c.schedule for c in (centers if centers is not None else _centers()) if number in c.numbers]
    return found[0] if found and all(s == found[0] for s in found) else None


def _time(minutes: int) -> str:
    return f"{minutes // 60}:{minutes % 60:02d}"


def status(schedule, now: datetime) -> str:
    if schedule == ALWAYS:
        return "🟢 פתוח 24 שעות ביממה"
    minute = now.hour * 60 + now.minute
    for start, end in schedule.get(now.weekday(), []):
        if start <= minute < end:
            return f"🟢 פתוח עכשיו, עד {_time(end)}"
    for ahead in range(0, 8):
        day = (now + timedelta(days=ahead)).weekday()
        opening = [s for s, _ in schedule.get(day, []) if ahead or s > minute]
        if opening:
            when = "היום" if ahead == 0 else "מחר" if ahead == 1 else f"ביום {DAYS[(day + 1) % 7]}'"
            return f"🔴 סגור כרגע, ייפתח {when} ב-{_time(min(opening))}"
    return ""


def statuses_for(text: str, now: datetime | None = None, centers=None) -> list[str]:
    """הסטטוסים של המוקדים שהמספרים שלהם מופיעים בטקסט (בלי כפולים, לפי הסדר)."""
    now = now or datetime.now(ISRAEL)
    out = []
    text = re.sub(r"\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}", " ", text)  # שעות בתוך התשובה - לא מספרי טלפון
    for number in dict.fromkeys(digits(n) for n in NUMBER.findall(text)):
        if (schedule := schedule_for(number, centers)) and (line := status(schedule, now)) and line not in out:
            out.append(line)
    return out


WHATSAPP = "0549893455"  # הוואטסאפ של שירות הלקוחות - הכפתור "המשך עם נציג בוואטסאפ"
