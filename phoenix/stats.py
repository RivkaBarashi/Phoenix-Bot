"""סטטיסטיקה קטנה: זמנים (ממוצע, p90, מקסימום), וגבול עליון לשיעור טעויות במדגם קטן (כיול ספי ה-Rerank)."""
import math


def binomial_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) כש-X ~ Binomial(n, p)."""
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def clopper_pearson_upper(k: int, n: int, confidence: float = 0.95) -> float:
    """גבול עליון (חד-צדדי, Clopper-Pearson) לשיעור אמיתי של אירוע שקרה k פעמים מתוך n: השיעור הגבוה ביותר שעוד
    מתיישב עם התצפית ברמת ביטחון confidence. 0 מתוך 52 -> 5.6%; 1 מתוך 52 -> 8.8%. מדויק, בלי הנחת נורמליות."""
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    low, high = k / n, 1.0
    for _ in range(60):  # חיפוש בינארי: binomial_cdf יורד ככל ש-p עולה
        mid = (low + high) / 2
        if binomial_cdf(k, n, mid) > 1 - confidence:
            low = mid
        else:
            high = mid
    return high


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
