"""כמה קריאות יצאו לכל שירות בהרצה הנוכחית - מודפס בסוף כל הרצת בדיקות, כדי לשמור על המכסות.

קריאה = בקשה שיצאה לשירות בפועל, כולל כאלה שנכשלו (גם הן נספרות במגבלה לדקה).
תשובה מהמטמון (של המודל או של ה-Rerank) לא נספרת כקריאה - היא מופיעה בנפרד, כחיסכון.
"""
from . import config, llm, rerank
from .fallback import CALLS, FAILED, resting

SERVICES = {"gemini": "Gemini", "cohere": "Cohere", "groq": "Groq", "openrouter": "OpenRouter", "voyage": "Voyage"}
LLM_NAMES = {f"{p}:{m}" for p, m in config.LLM_CHAIN}
EMBED_NAMES = {f"{p}:{m}" for p, m in config.EMBED_CHAIN}
RERANK_NAMES = {f"{p}:{m}" for p, m in config.RERANK_CHAIN}


def _calls(name: str) -> int:
    """Rerank: רק בקשות HTTP אמיתיות (קטעים שכבר דורגו מגיעים מהמטמון בלי בקשה)."""
    return rerank.API_REQUESTS[name] if name in RERANK_NAMES else CALLS[name]


def report() -> str:
    names = set(CALLS) | set(rerank.API_REQUESTS) | set(llm.CACHE_HITS)
    lines = ["=== קריאות שיצאו לכל שירות (בהרצה הזו) ==="]
    for service, label in SERVICES.items():
        mine = sorted(n for n in names if n.startswith(service + ":"))
        parts = {purpose: sum(_calls(n) for n in mine if n in group)
                 for purpose, group in (("מודל שפה", LLM_NAMES), ("אמבדינגים", EMBED_NAMES), ("Rerank", RERANK_NAMES))}
        total = sum(parts.values())
        line = f"   {label:<11} {total:>4} קריאות"
        if total:
            line += " (" + ", ".join(f"{k} {v}" for k, v in parts.items() if v) + ")"
        if failed := sum(FAILED[n] for n in mine):
            line += f" | נכשלו {failed}"
        saved = [f"{sum(llm.CACHE_HITS[n] for n in mine)} תשובות מודל",
                 f"{sum(rerank.CACHE_HITS[n] for n in mine)} ציוני Rerank"]
        if any(not s.startswith("0 ") for s in saved):
            line += " | מהמטמון: " + ", ".join(s for s in saved if not s.startswith("0 "))
        if tokens := sum(llm.TOKENS[n] for n in mine):
            line += f" | {tokens:,} טוקנים"
        lines.append(line)
    detail = [f"{n} {_calls(n)}" + (f" ({FAILED[n]} נכשלו)" if FAILED[n] else "") for n in sorted(names) if _calls(n)]
    if detail:
        lines.append("   לפי מודל: " + " | ".join(detail))
    if rest := resting():
        lines.append("   נחים עכשיו (מכסה/עומס): " + " | ".join(f"{n} עוד {_duration(s)}" for n, s in sorted(rest.items())))
    return "\n".join(lines)


def _duration(seconds: float) -> str:
    if seconds == float("inf"):
        return "עד הפעלה מחדש"
    minutes = int(seconds // 60)
    return f"{minutes // 60}:{minutes % 60:02d} שעות" if minutes >= 60 else f"{max(minutes, 1)} דק'"
