"""הגדרות הפרויקט: מפתחות API מתוך .env, וסדר הספקים לכל מטרה.

כדי לשנות את סדר הגיבוי - משנים את הסדר ברשימות למטה.
"""
import os
import threading
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
    [(1, API_KEYS["gemini"].strip())] + [(n, os.getenv(f"GEMINI_API_KEY_{n}", "").strip()) for n in range(2, 21)]
    if key
]

# מודל שפה לתשובות, לפי סדר. בחינמי של Gemini לכל מודל מכסה יומית נפרדת, ולמודלי Flash
# היא רק 20 בקשות ביום (נבדק 16/10, מתאפסת בחצות שעון פסיפיק) - לכן כמה דגמי Flash ברצף,
# ואחריהם Flash-Lite ו-Gemma. ספק שנגמרה לו המכסה "נח" עד האיפוס, והבא בתור עונה.
# בלי gemini-3.5-flash: מאחוריו עונה gemini-3.6-flash (נבדק 19/10 - model_version בתשובה, ומכסה משותפת),
# כך שהוא רק ניסיון כפול על אותה מכסה, ובבקשות במקביל - אותו מודל פעמיים.
LLM_CHAIN = [
    ("gemini", "gemini-3.6-flash"),
    ("gemini", "gemini-3.7-flash"),
    ("gemini", "gemini-3.8-flash"),
    # בלי Flash-Lite ו-Gemma: איכות לפני מהירות (19/10, בעומס: Flash-Lite ענה בטלפון של מוקד אחר ונתן "המלצה").
    # כשכל מודלי ה-Flash לא זמינים - הודעת התקלה עם ההפניה, ולא תשובה ממודל חלש.
    # Groq: 1,000 בקשות ביום, אבל עד 8K טוקנים לדקה - מקבל הקשר מקוצר (answer.py)
    ("groq", "openai/gpt-oss-120b"),
    # OpenRouter: 50 בקשות ביום לכל המודלים החינמיים יחד
    ("openrouter", "nvidia/nemotron-3-super-120b-a12b:free"),
    ("openrouter", "google/gemma-4-31b-it:free"),
    # בלי Cohere כמודל שפה: הוא חולק את מכסת ה-Trial (1,000 בחודש) עם ה-Rerank, ואם היא נגמרת - הסף נכבה
]

# שלב הבנת השאלה (understand.py): המודל הכי איכותי שיש לנו קודם (Pro לא זמין בחינמי), ואחריו השרשרת הרגילה
UNDERSTAND_TIMEOUT: float | None = float(os.getenv("UNDERSTAND_TIMEOUT", "8"))  # בצ'אט; הבדיקות - בלי מגבלה
UNDERSTAND_LLMS = [m.strip() for m in os.getenv("UNDERSTAND_LLMS", "gemini-3.8-flash,gemini-3.7-flash,gemini-3.6-flash").split(",")]

# אמבדינגים: לכל ספק אינדקס משלו (וקטורים של מודלים שונים לא ניתנים להשוואה).
EMBED_CHAIN = [
    ("gemini", "gemini-embedding-2"),
    ("voyage", "voyage-4-lite"),
]
# באיזה אינדקס הבוט מחפש: gemini או voyage. אם הספק הזה לא זמין - עוברים לאינדקס של השני.
EMBEDDER = os.getenv("EMBEDDER", "gemini").strip().lower()
CHROMA_DIR = ROOT / "data" / "chroma"

# Rerank: Cohere, ואם הוא לא זמין - Voyage (מהיר, בתוך 20 השניות גם בענן), ובמחשב שהמדרג המקומי מותקן בו - הוא
# אחרון (מודל ONNX ב-data/models, על המעבד: ~10 שניות לשאלה, ולכן לא בענן).
LOCAL_RERANKER = os.getenv("LOCAL_RERANKER", "bge-reranker-v2-m3")
RERANK_CHAIN = [
    ("cohere", "rerank-v4.0-pro"),
    ("voyage", "rerank-3"),
    ("local", LOCAL_RERANKER),
]
# שכבת הגנה ראשונה: ציון Rerank מינימלי כדי בכלל לנסות לענות. מתחת לזה - "לא מופיע באתר".
# לכל מדרג סקאלה משלו, ולכן סף משלו - כל אחד כויל על data/questions.csv. כשאף מדרג לא זמין - מצב זהיר:
# בלי סף, המודל מחליט והקוד מאמת, ואזהרה ביומן ובפרטים הטכניים.
RERANK_THRESHOLDS = {
    "cohere": float(os.getenv("RERANK_THRESHOLD", "0.82")),
    # כויל 19/10 (20 מועמדים, 384 טוקנים): ב-0.2 נשארו 21/23 שאלות עם תשובה, ו-20/20 בלי תשובה סורבו
    "local": float(os.getenv("RERANK_THRESHOLD_LOCAL", "0.2")),
}
RERANK_THRESHOLD = RERANK_THRESHOLDS["cohere"]  # הסף של המסלול הראשי (הבדיקות מכיילות מולו)

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
# 20 השניות הן מגבלה קשיחה. בצ'אט כל שאלה נשלחת במקביל לשני מודלי Flash חזקים (RACE_LLMS, כל אחד במפתח אחר),
# והתשובה הראשונה שעוברת את כל הבדיקות מוצגת - מודל עמוס או איטי לא מעכב את התשובה. שורף כפול מכסה בשלב
# התשובה. מודל מהזוג לא זמין - הבא בתור מ-LLM_CHAIN. הבדיקות (eval) תמיד מודל אחד.
PARALLEL_LLM = os.getenv("PARALLEL_LLM", "true").strip().lower() in ("1", "true", "yes", "on")
RACE_LLMS = [m.strip() for m in os.getenv("RACE_LLMS", "gemini-3.6-flash,gemini-3.8-flash").split(",") if m.strip()]
# בצ'אט: החיפוש על השאלה כמו שנכתבה מתחיל במקביל להבנת השאלה. ניסוח עצמאי שונה מהותית (פחות מ-70% מילים
# משותפות, סוג ביטוח אחר, שאלה מפוצלת, תרגום) - מחפשים שוב עליו. שאלת המשך שתלויה בשיחה - בלי חיפוש מוקדם.
PARALLEL_SEARCH = os.getenv("PARALLEL_SEARCH", "true").strip().lower() in ("1", "true", "yes", "on")
SAME_SEARCH_JACCARD = 0.7
# חשיבה: הרמה הנמוכה ביותר שכל מודל מקבל (נבדק 20/10: 3.6 - MINIMAL; 3.7 ו-3.8 דוחים MINIMAL - LOW). שלב ההבנה
# תמיד בה (3.8 בהבנה: 2.6 שניות במקום 14-23 בעומס הלילה). בתשובה - רק עם ANSWER_LOW_THINKING, אחרי בדיקת איכות.
LOW_THINKING = {"gemini-3.6-flash": "MINIMAL", "gemini-3.7-flash": "LOW", "gemini-3.8-flash": "LOW"}
ANSWER_LOW_THINKING = os.getenv("ANSWER_LOW_THINKING", "false").strip().lower() in ("1", "true", "yes", "on")


def patient(timeout: float = 60, retries: int = 1, max_wait: float = 10):
    """לסקריפטים בלבד (eval_*.py): המתנה ארוכה יותר לכל ספק, ועל עומס (503) - ניסיון חוזר באותו מודל במקום
    לדלג עליו 5 דקות (זיכרון העומס הוא לצ'אט - fallback.OVERLOAD_MEMORY)."""
    global QUERY_TIMEOUT, QUERY_RETRIES, QUERY_MAX_WAIT
    QUERY_TIMEOUT, QUERY_RETRIES, QUERY_MAX_WAIT = timeout, retries, max_wait
    from . import fallback  # כאן ולא למעלה: fallback לא תלוי ב-config, וזה המקום היחיד שמשנה אותו
    fallback.OVERLOAD_MEMORY = False
    global UNDERSTAND_TIMEOUT
    UNDERSTAND_TIMEOUT = None  # בבדיקות מודדים את הבנת השאלה עצמה, לא את הגיבוי


_gemini_clients: dict[int, genai.Client] = {}
_gemini_clients_lock = threading.Lock()


def gemini_client(number: int = 1) -> genai.Client:
    """לקוח Gemini למפתח מספר number (1 = GEMINI_API_KEY, 2 = GEMINI_API_KEY_2 ...). לקוח אחד לכל מפתח, גם כששני
    threads מבקשים אותו בפעם הראשונה באותו רגע (בקשות במקביל): לקוח כפול נמחק מיד, והספרייה סוגרת את החיבור
    שלו באמצע הבקשה - "Cannot send a request, as the client has been closed"."""
    with _gemini_clients_lock:
        if number not in _gemini_clients:
            _gemini_clients[number] = genai.Client(
                api_key=dict(GEMINI_KEYS).get(number, API_KEYS["gemini"]),
                http_options=types.HttpOptions(timeout=HTTP_TIMEOUT * 1000),
            )
        return _gemini_clients[number]
