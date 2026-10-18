"""סטטיסטיקה קטנה לזמנים: ממוצע ו-p90 (הזמן ש-90% מהשאלות מתחתיו)."""
import math


def p90(values: list[float]) -> float:
    """האחוזון ה-90 בשיטת "הדירוג הקרוב": ערך אמיתי מהרשימה, ש-90% מהערכים קטנים ממנו או שווים לו."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)] if ordered else 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stage_table(timings: list[dict[str, float]], order: list[str] | None = None) -> list[tuple[str, int, float, float]]:
    """[(שלב, כמה שאלות, ממוצע, p90)] לכל שלב שמופיע ברשומות."""
    stages = order or sorted({k for t in timings for k in t})
    rows = []
    for stage in stages:
        values = [t[stage] for t in timings if stage in t]
        if values:
            rows.append((stage, len(values), mean(values), p90(values)))
    return rows
