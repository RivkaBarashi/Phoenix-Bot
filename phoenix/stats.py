"""סטטיסטיקה קטנה לזמנים: ממוצע, p90 (הזמן ש-90% מהשאלות מתחתיו) ומקסימום."""
import math


def p90(values: list[float]) -> float:
    """האחוזון ה-90 בשיטת "הדירוג הקרוב": ערך אמיתי מהרשימה, ש-90% מהערכים קטנים ממנו או שווים לו."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)] if ordered else 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def search_seconds(timings: dict[str, float]) -> float | None:
    """כמה החיפוש הוסיף לזמן התשובה: החיפוש המוקדם רץ במקביל להבנה, ולכן אז רק ההמתנה לו אחרי ההבנה
    (search_wait); בלי חיפוש מוקדם (או חיפוש שני) - וקטור + מילות מפתח + Rerank."""
    if "search_wait" in timings:
        return timings["search_wait"]
    parts = [timings[k] for k in ("vector", "bm25", "rerank") if k in timings]
    return sum(parts) if parts else None


def stage_table(timings: list[dict[str, float]],
                order: list[str] | None = None) -> list[tuple[str, int, float, float, float]]:
    """[(שלב, כמה שאלות, ממוצע, p90, מקסימום)] לכל שלב שמופיע ברשומות. השלב "search" מחושב (search_seconds)."""
    timings = [t | ({"search": s} if (s := search_seconds(t)) is not None else {}) for t in timings]
    stages = order or sorted({k for t in timings for k in t})
    rows = []
    for stage in stages:
        values = [t[stage] for t in timings if stage in t]
        if values:
            rows.append((stage, len(values), mean(values), p90(values), max(values)))
    return rows
