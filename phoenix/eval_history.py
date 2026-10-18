"""היסטוריית ריצות הבדיקה: שורה ב-data/eval_history.csv בכל ריצה - תאריך, commit ומדדים.
כך רואים לאורך זמן מה כל שינוי עשה לתוצאות. commit עם "+dirty" = היו שינויים בקוד שעוד לא נכנסו ל-commit.
מדד בצורה "18/24" = 18 מתוך 24 שאלות מהסוג הזה.
"""
import csv
import subprocess
import time

from . import config

HISTORY_FILE = config.ROOT / "data" / "eval_history.csv"
FIELDS = ["date", "commit", "script", "questions", "selection", "index", "prompt", "threshold", "n",
          "answered", "correct", "partial", "wrong", "cited_correct", "not_on_site_ok", "personal_ok",
          "personal_correct", "greeting_ok", "off_topic_ok", "injection_blocked", "top1", "top3", "top5", "smoke_passed",
          "mean_total_s", "p90_total_s", "llm_calls", "llm_cache_hits", "stopped"]


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=config.ROOT, capture_output=True, text=True, timeout=15).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def git_commit() -> str:
    """ה-commit הנוכחי, ו-"+dirty" אם יש שינויים בקוד (קובצי התוצאות של הבדיקות לא נחשבים)."""
    commit = _git("rev-parse", "--short", "HEAD")
    dirty = _git("status", "--porcelain", "--untracked-files=no", "--", ".", ":!data/eval_*")
    return commit + ("+dirty" if commit and dirty else "")


def append(**row):
    """מוסיף שורה. עמודות שלא היו בקובץ (גרסה ישנה) מתווספות לכל השורות."""
    row = {"date": time.strftime("%Y-%m-%d %H:%M"), "commit": git_commit(), **row}
    rows = list(csv.DictReader(open(HISTORY_FILE, encoding="utf-8-sig"))) if HISTORY_FILE.exists() else []
    fields = FIELDS + [k for r in rows + [row] for k in r if k not in FIELDS]
    with open(HISTORY_FILE, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(fields)))
        writer.writeheader()
        writer.writerows(rows + [row])
    return HISTORY_FILE


def ratio(good: int, total: int) -> str:
    return f"{good}/{total}" if total else ""
