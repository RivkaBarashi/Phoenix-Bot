"""חיפוש מילות מפתח (BM25) על הקטעים, עם טיפול בתחיליות בעברית.

בעברית אותיות השימוש (ו, ה, ב, ל, מ, ש, כ) נצמדות לתחילת המילה: "בהצפה", "והצפה"
ו"הצפה" הן אותה מילה. לכן כל מילה נכנסת גם בלי עד 3 אותיות כאלה בתחילתה
("בהצפה" -> "בהצפה", "הצפה", "צפה"), גם בקטעים וגם בשאלה - וכך הן מתאימות.
מילים של 3 אותיות ומטה לא נחתכות, כדי ש"שלי" לא יהפוך ל"לי".
חיפוש זה רץ מקומית, בלי API - הוא עובד גם כשכל הספקים חסומים.
"""
import json
import re
from functools import lru_cache

from rank_bm25 import BM25Okapi

from .chunk import CHUNKS_FILE

PREFIX_LETTERS = set("והבלמשכ")
MAX_PREFIXES = 3
WORD = re.compile(r"\w+(?:[\"'׳״]\w+)*")  # כולל ראשי תיבות כמו חו"ל, ת"ז
STOPWORDS = {
    "של", "את", "על", "עם", "אל", "אם", "או", "גם", "כי", "זה", "זו", "זאת", "הוא", "היא", "הם", "הן",
    "אני", "אתה", "את", "אנחנו", "אתם", "יש", "אין", "לא", "כן", "מה", "מי", "איך", "האם", "כל", "רק",
    "עוד", "כמו", "אבל", "לי", "לך", "לו", "לה", "לנו", "לכם", "להם", "שלי", "שלך", "שלו", "שלה",
    "שלנו", "שלכם", "הזה", "הזאת", "אחרי", "לפני", "בין", "תוך", "מאוד", "כך", "אז", "היה", "היו",
}


def variants(word: str) -> list[str]:
    """המילה, ואותה מילה בלי אות שימוש אחת, שתיים או שלוש בתחילתה."""
    out = [word]
    for _ in range(MAX_PREFIXES):
        if len(word) <= 3 or word[0] not in PREFIX_LETTERS:
            break
        word = word[1:]
        out.append(word)
    return out


def tokenize(text: str) -> list[str]:
    return [v for w in WORD.findall(text.lower()) for v in variants(w) if len(v) > 1 and v not in STOPWORDS]


@lru_cache
def _index() -> tuple[list[dict], BM25Okapi]:
    chunks = [json.loads(line) for line in open(CHUNKS_FILE, encoding="utf-8")]
    return chunks, BM25Okapi([tokenize(c["text"]) for c in chunks])


def search(query: str, n: int = 20) -> list[tuple[dict, float]]:
    """n הקטעים עם ציון BM25 הגבוה ביותר: [(קטע, ציון)]."""
    chunks, bm25 = _index()
    scores = bm25.get_scores(tokenize(query))
    best = sorted(range(len(chunks)), key=lambda i: -scores[i])[:n]
    return [(chunks[i], float(scores[i])) for i in best if scores[i] > 0]
