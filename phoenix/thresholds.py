"""מצב ספי ה-Rerank המכוילים (data/thresholds.json) - docs/before-paid-tests.md, סעיף 1.

הסף הוא "מסננת גסה": מתחתיו "לא מופיע באתר" בלי לשאול את מודל התשובה. ציון ה-Rerank מודד "הקטע באותו נושא", לא "הקטע
עונה על השאלה" (בריצה הסופית 21/10: ציונים של שאלות עם תשובה ובלי תשובה חופפים), ולכן הסף נבחר כך שכמעט אף שאלה עם
תשובה לא תיחסם (calibrate_thresholds.py), ואת "לא מופיע באתר" מעליו מחליטים המודל והבדיקות בקוד.

הקובץ שומר לכל מדרג: הסף, מודל הדירוג, גרסת האינדקס, המדגם והגבול העליון. מודל הדירוג או האינדקס השתנו - הסף "התיישן":
הבוט לא משנה התנהגות בזמן ריצה (רק אזהרה ביומן ובשורת מצב מפתחת), והבדיקה האוטומטית נכשלת עד כיול מחדש - כך סף ישן לא
עולה לאוויר בשקט.
"""
import json
import logging
import os

from . import answer_cache, config

log = logging.getLogger("phoenix")
ENV_NAMES = {"cohere": "RERANK_THRESHOLD", "local": "RERANK_THRESHOLD_LOCAL", "voyage": "RERANK_THRESHOLD_VOYAGE"}


def load() -> dict[str, dict]:
    try:
        data = json.loads(config.THRESHOLDS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {name: entry for name, entry in data.items() if isinstance(entry, dict)}


def status(ranker: str, entries: dict[str, dict] | None = None) -> str:
    """calibrated / stale:model / stale:index / uncalibrated (אין כיול בקובץ - ערך ברירת המחדל או משתנה סביבה)."""
    entry = (load() if entries is None else entries).get(ranker)
    if not entry:
        return "uncalibrated"
    if entry.get("model") != dict(config.RERANK_CHAIN).get(ranker):
        return "stale:model"
    if entry.get("index_version") != answer_cache.index_version():
        return "stale:index"
    return "calibrated"


def warnings() -> list[str]:
    """מה לא בסדר בספים: כיול שהתיישן, או משתנה סביבה שסותר את הכיול (הקובץ קובע)."""
    entries, out = load(), []
    for ranker, entry in entries.items():
        state = status(ranker, entries)
        if state != "calibrated":
            out.append(f"{ranker}: הסף {entry.get('threshold')} {state} - צריך להריץ calibrate_thresholds.py")
        env = os.getenv(ENV_NAMES.get(ranker, ""), "").strip()
        try:
            differs = bool(env) and abs(float(env) - float(entry.get("threshold", -1))) > 1e-9
        except ValueError:  # ערך לא מספרי בסביבה - config היה נכשל קודם; כאן רק לא להפיל את האזהרות
            differs = True
        if differs:
            out.append(f"{ranker}: {ENV_NAMES[ranker]}={env} ב-.env/בסביבה, אבל הכיול בקובץ ({entry.get('threshold')}) "
                       "הוא שקובע")
    return out


def summary() -> str:
    """לשורת מצב מפתחת: "ספים: cohere 0.82 (מכויל)" / "(התיישן: index)" / "(לא מכויל)"."""
    entries = load()
    names = {"calibrated": "מכויל", "stale:model": "התיישן: מודל", "stale:index": "התיישן: אינדקס",
             "uncalibrated": "לא מכויל"}
    return "ספים: " + " · ".join(f"{r} {config.RERANK_THRESHOLDS[r]} ({names[status(r, entries)]})"
                                 for r in config.RERANK_THRESHOLDS)


def log_warnings() -> None:
    for line in warnings():
        log.warning("rerank threshold: %s", line)
