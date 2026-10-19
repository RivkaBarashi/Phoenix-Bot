"""הגדרות הפרויקט: מפתחות API מתוך .env, וסדר הספקים לכל מטרה.

כדי לשנות את סדר הגיבוי - משנים את הסדר ברשימות למטה.
"""
import os
from functools import lru_cache
from pathlib import Path

import truststore
from dotenv import load_dotenv

# הרשת כאן עוברת דרך סינון (Netspark) שמחליף תעודות HTTPS בחלק מהאתרים.
# התעודה שלו מותקנת ב-Windows, אז אומרים לפייתון להשתמש במאגר התעודות של Windows.
truststore.inject_into_ssl()

from google import genai  # noqa: E402 - אחרי truststore
from google.genai import types  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

API_KEYS = {
    "gemini": os.getenv("GEMINI_API_KEY", ""),
    "groq": os.getenv("GROQ_API_KEY", ""),
    "openrouter": os.getenv("OPENROUTER_API_KEY", ""),
    "cohere": os.getenv("COHERE_API_KEY", ""),
    "voyage": os.getenv("VOYAGE_API_KEY", ""),
    "firecrawl": os.getenv("FIRECRAWL_API_KEY", ""),
}
# מאגר מפתחות Gemini: כל מפתח מחשבון ומפרויקט נפרד, ולכן עם מכסה יומית משלו לכל מודל.
# המספר הוא המספר בשם המשתנה (GEMINI_API_KEY = 1, GEMINI_API_KEY_2 = 2 ...) - כך הוא מופיע בדוחות, בלי הערך.
GEMINI_KEYS: list[tuple[int, str]] = [
    (number, key) for number, key in
    [(1, API_KEYS["gemini"].strip())] + [(n, os.getenv(f"GEMINI_API_KEY_{n}", "").strip()) for n in range(2, 10)]
    if key
]

# מודל שפה לתשובות, לפי סדר. בחינמי של Gemini לכל מודל מכסה יומית נפרדת, ולמודלי Flash
# היא רק 20 בקשות ביום (נבדק 16/10, מתאפסת בחצות שעון פסיפיק) - לכן כמה דגמי Flash ברצף,
# ואחריהם Flash-Lite ו-Gemma. ספק שנגמרה לו המכסה "נח" עד האיפוס, והבא בתור עונה.
LLM_CHAIN = [
    ("gemini", "gemini-3.5-flash"),
    ("gemini", "gemini-3.6-flash"),
    ("gemini", "gemini-3.7-flash"),
    ("gemini", "gemini-3.8-flash"),
    ("gemini", "gemini-3.5-flash-lite"),
    ("gemini", "gemini-3.1-flash-lite"),
    ("gemini", "gemma-4-26b-a4b-it"),
    # Groq: 1,000 בקשות ביום, אבל עד 8K טוקנים לדקה - מקבל הקשר מקוצר (answer.py)
    ("groq", "openai/gpt-oss-120b"),
    # OpenRouter: 50 בקשות ביום לכל המודלים החינמיים יחד
    ("openrouter", "nvidia/nemotron-3-super-120b-a12b:free"),
    ("openrouter", "google/gemma-4-31b-it:free"),
    ("cohere", "command-a-plus-05-2026"),  # אחרון: חולק את מכסת ה-Trial עם ה-Rerank
]

# אמבדינגים: לכל ספק אינדקס משלו (וקטורים של מודלים שונים לא ניתנים להשוואה).
EMBED_CHAIN = [
    ("gemini", "gemini-embedding-2"),
    ("voyage", "voyage-4-lite"),
]
# באיזה אינדקס הבוט מחפש: gemini או voyage. אם הספק הזה לא זמין - עוברים לאינדקס של השני.
EMBEDDER = os.getenv("EMBEDDER", "gemini").strip().lower()
CHROMA_DIR = ROOT / "data" / "chroma"

RERANK_CHAIN = [
    ("cohere", "rerank-v4.0-pro"),
    ("voyage", "rerank-3"),
]
# שכבת הגנה ראשונה: ציון Rerank מינימלי כדי בכלל לנסות לענות. מתחת לזה - "לא מופיע באתר".
# הסף נקבע מול הציונים של Cohere, ולכן תקף רק כש-Cohere דירג. בלי Cohere ההחלטה היא של המודל.
RERANK_THRESHOLD = float(os.getenv("RERANK_THRESHOLD", "0.82"))
THRESHOLD_RERANKER = "cohere"

# איסוף עמודים: קודם ישירות (חינם, בלי מכסה), ורק אם נכשל - שירותים חיצוניים.
# jina לא צריך מפתח.
FETCH_CHAIN = ["direct", "firecrawl", "jina"]

USER_AGENT = "PhoenixHackathonBot/0.1 (student RAG project; polite crawl)"
HTTP_TIMEOUT = 30  # ספק שלא עונה תוך 30 שניות - עוברים לבא (בניית האינדקס, איסוף עמודים)

# בזמן שאלה (הצ'אט): כל ספק - מודל שפה, וקטור לשאלה, Rerank - מקבל עד QUERY_TIMEOUT שניות,
# ועל 429 או תקלה עוברים מיד לספק הבא, בלי לחכות. (Gemini לא מקבל פחות מ-10 שניות.)
# סקריפטי הבדיקות קוראים ל-patient() ומחכים יותר, כדי לא לעבור לגיבוי בגלל עומס רגעי.
QUERY_TIMEOUT = 15
QUERY_RETRIES = 0
QUERY_MAX_WAIT = 0.0
# זמן כולל לשאלה אחת בצ'אט - כל השלבים יחד (השלמת שאלת המשך, חיפוש, Rerank, מודל). כל ניסיון אצל ספק מקבל
# רק את מה שנשאר; כשנשארות פחות מ-3 שניות - הודעה ידידותית, בלי ניסיון נוסף. (הבדיקות לא מוגבלות.)
QUESTION_TIMEOUT = float(os.getenv("QUESTION_TIMEOUT", "20"))
# בצ'אט: כל שאלה נשלחת במקביל לשני מודלי Gemini שונים (במפתחות שונים), והתשובה הראשונה שעוברת את כל
# הבדיקות מוצגת. כפול קריאות למודל - false מכבה. הבדיקות (eval) תמיד מודל אחד, בזה אחר זה.
PARALLEL_LLM = os.getenv("PARALLEL_LLM", "true").strip().lower() in ("1", "true", "yes", "on")


def patient(timeout: float = 60, retries: int = 1, max_wait: float = 10):
    """לסקריפטים בלבד (eval_*.py): המתנה ארוכה יותר לכל ספק."""
    global QUERY_TIMEOUT, QUERY_RETRIES, QUERY_MAX_WAIT
    QUERY_TIMEOUT, QUERY_RETRIES, QUERY_MAX_WAIT = timeout, retries, max_wait


@lru_cache
def gemini_client(number: int = 1) -> genai.Client:
    """לקוח Gemini למפתח מספר number (1 = GEMINI_API_KEY, 2 = GEMINI_API_KEY_2 ...)."""
    return genai.Client(
        api_key=dict(GEMINI_KEYS).get(number, API_KEYS["gemini"]),
        http_options=types.HttpOptions(timeout=HTTP_TIMEOUT * 1000),
    )
