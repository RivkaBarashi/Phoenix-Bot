"""הגדרות הפרויקט: מפתחות API מתוך .env, וסדר הספקים לכל מטרה.

כדי לשנות את סדר הגיבוי - משנים את הסדר ברשימות למטה.
"""
import contextlib
import contextvars
import os
import sys
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


def _streamlit_secrets_to_env():
    """Streamlit Cloud: הסודות (Secrets) בשורש מגיעים גם כמשתני סביבה. ליתר ביטחון - מה שחסר בסביבה נלקח
    מ-st.secrets (רק כשרצים בתוך Streamlit, ובלי לדרוס משתנה קיים). בלי קובץ סודות - שום דבר לא קורה."""
    if "streamlit" not in sys.modules:
        return
    try:
        import streamlit as st
        for name, value in st.secrets.items():
            if isinstance(value, (str, int, float, bool)) and name not in os.environ:
                os.environ[name] = str(value)
    except Exception:  # noqa: BLE001 - אין סודות (מקומית) - משתני הסביבה ו-.env
        pass


_streamlit_secrets_to_env()

# מאיזה אתר נכתבה כל שורה ביומן המשותף: SITE, או לפי הסביבה (Render מגדיר RENDER; Streamlit Cloud מריץ מ-/mount/src)
SITE = os.getenv("SITE") or ("render" if os.getenv("RENDER") else
                             "streamlit-cloud" if Path("/mount/src").exists() else "local")

API_KEYS = {
    "gemini": os.getenv("GEMINI_API_KEY", ""),
    "groq": os.getenv("GROQ_API_KEY", ""),
    "openrouter": os.getenv("OPENROUTER_API_KEY", ""),
    "cohere": os.getenv("COHERE_API_KEY", ""),
    "voyage": os.getenv("VOYAGE_API_KEY", ""),
    "firecrawl": os.getenv("FIRECRAWL_API_KEY", ""),
}
# Cohere כמודל שפה (גיבוי מחוץ ל-Google): רק חשבון Trial נפרד, COHERE_CHAT_API_KEY (החלטת הבעלים 21/10: המפתח של
# ה-Rerank, COHERE_API_KEY, לא עונה ולא מבין שאלות - המכסה החודשית שלו, 1,000, שמורה ל-Rerank). בלי המפתח - בלי גיבוי.
COHERE_CHAT_KEY = (os.getenv("COHERE_CHAT_API_KEY", "") or os.getenv("COHERE_API_KEY2", "")).strip()  # שני השמות - אותו חשבון
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
]
# מודל הגיבוי מחוץ ל-Google (21/10: gemini-3.8-flash החזיר 504 בכל 20 המפתחות, ו-38/81 שאלות בריצה הסופית נשארו בלי
# תשובה). מקבל את ההקשר המלא (5 קטעים) ואת אותן בדיקות. בצ'אט הוא המתחרה השני במירוץ התשובה (מול 3.6) ונכנס לגיבוי
# של הבנת השאלה; Google לא עונה - הוא עונה. נמדד 21/10 על פרומפט התשובה האמיתי: command-a-03-2025 - 10.6 ש',
# command-a-plus - 9.9 ש', command-r-plus - 46 ש'; שעה אחר כך command-a - 37-43 ש' (Trial בעומס) ו-command-a-plus - 8.6 ש'.
# לכן command-a-plus. Groq לא מתאים (8K טוקנים לדקה, ההקשר ~11K). ריק - בלי גיבוי.
BACKUP_LLM = os.getenv("BACKUP_LLM", "cohere:command-a-plus-05-2026").strip()
if BACKUP_LLM:
    LLM_CHAIN.append(tuple(BACKUP_LLM.split(":", 1)))

# שלב הבנת השאלה (understand.py): המודל הכי איכותי שיש לנו קודם (Pro לא זמין בחינמי), ואחריו השרשרת הרגילה
UNDERSTAND_TIMEOUT: float | None = float(os.getenv("UNDERSTAND_TIMEOUT", "8"))  # בצ'אט; הבדיקות - בלי מגבלה
UNDERSTAND_LLMS = [m.strip() for m in os.getenv("UNDERSTAND_LLMS", "gemini-3.8-flash,gemini-3.7-flash,gemini-3.6-flash").split(",")]
# בצ'אט - בקשת גיבוי להבנה: המודלים החזקים קודם; אין תשובה תוך UNDERSTAND_HEDGE_AFTER שניות (או כשל מהיר) - אותה
# בקשה גם ל-UNDERSTAND_HEDGE, והתשובה התקינה הראשונה קובעת. נבדק 20/10 בעומס אצל Google: 3.8 ו-3.7 - 504 אחרי 30
# שניות, 3.6 - 2.7 שניות. בלי הגיבוי כל שאלה חיכתה 8 שניות ועברה לכללי הקוד. ריק - בלי גיבוי.
UNDERSTAND_HEDGE = os.getenv("UNDERSTAND_HEDGE", "gemini-3.6-flash").strip()
UNDERSTAND_HEDGE_AFTER = float(os.getenv("UNDERSTAND_HEDGE_AFTER", "3"))
# יחד עם בקשת הגיבוי - גם למודל שמחוץ ל-Google (BACKUP_LLM), כך ש-Google שלא עונה לא משאיר את השאלה לכללי הקוד.
# הכלל (21/10): ההבנה שלו נבחרת רק אם 3.8 ו-3.6 נכשלו, או ש-3.6 לא ענה תוך UNDERSTAND_BACKUP_GRACE שניות מתחילת הגיבוי.
UNDERSTAND_BACKUP = BACKUP_LLM
UNDERSTAND_BACKUP_GRACE = float(os.getenv("UNDERSTAND_BACKUP_GRACE", "3"))

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
    # כויל 20/10 (20 מועמדים, כל קטע מקוצר לחלק הרלוונטי, 61 שאלות): השאלה עם תשובה הכי נמוכה 0.644, השאלה בלי
    # תשובה הכי גבוהה 0.636. נבחר 0.65 (החלטת מפתחת הפרויקט): לא להמציא חשוב יותר, ו-0.64 נשען על מרווח של 0.004.
    # ב-0.65: 30/31 שאלות עם תשובה נשארו (אחת סורבה בטעות, 0.644), 20/20 בלי תשובה סורבו.
    "voyage": float(os.getenv("RERANK_THRESHOLD_VOYAGE") or "0.65"),
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
# כלל המעבר ל-BACKUP_LLM בשלב התשובה (החלטת הבעלים, 21/10): Google החזיר תקלה (503, מכסה, 504/timeout) או שאין מפתח
# Gemini פנוי - הגיבוי מתחיל מיד; Google רק איטי - הגיבוי מתחיל במקביל אחרי BACKUP_AFTER_SECONDS. Gemini קודם: תשובה תקינה
# שלו עד GEMINI_PREFERRED_SECONDS (מתחילת שלב התשובה) - היא מוצגת; אחרי זה - התשובה התקינה הראשונה שיש.
BACKUP_AFTER_SECONDS = float(os.getenv("BACKUP_AFTER_SECONDS", "3"))
GEMINI_PREFERRED_SECONDS = float(os.getenv("GEMINI_PREFERRED_SECONDS", "12"))
PARALLEL_LLM = os.getenv("PARALLEL_LLM", "true").strip().lower() in ("1", "true", "yes", "on")
RACE_LLMS = [m.strip() for m in os.getenv("RACE_LLMS", "gemini-3.6-flash,gemini-3.8-flash").split(",") if m.strip()]
# בצ'אט: החיפוש על השאלה כמו שנכתבה מתחיל במקביל להבנת השאלה. ניסוח עצמאי שונה מהותית (פחות מ-70% מילים
# משותפות, סוג ביטוח אחר, שאלה מפוצלת, תרגום) - מחפשים שוב עליו. שאלת המשך שתלויה בשיחה - בלי חיפוש מוקדם.
PARALLEL_SEARCH = os.getenv("PARALLEL_SEARCH", "true").strip().lower() in ("1", "true", "yes", "on")
SAME_SEARCH_JACCARD = 0.7
# חשיבה: הרמה הנמוכה ביותר שכל מודל מקבל (נבדק 20/10: 3.6 - MINIMAL; 3.7 ו-3.8 דוחים MINIMAL - LOW). שלב ההבנה
# לפי UNDERSTAND_LOW_THINKING / UNDERSTAND_HEDGE_LOW_THINKING למטה. בתשובה - רק עם ANSWER_LOW_THINKING, אחרי בדיקת איכות.
LOW_THINKING = {"gemini-3.6-flash": "MINIMAL", "gemini-3.7-flash": "LOW", "gemini-3.8-flash": "LOW"}
ANSWER_LOW_THINKING = os.getenv("ANSWER_LOW_THINKING", "false").strip().lower() in ("1", "true", "yes", "on")
# שלב ההבנה: חשיבה נמוכה לכל מודל (UNDERSTAND_LOW_THINKING), או רק למודל הגיבוי (UNDERSTAND_HEDGE_LOW_THINKING).
# נבדק 20/10, הבנה בלבד על 75 השאלות, 3.6: נמוכה - שגויה 1/75, חציון 3.5 ש'; רגילה - שגויה 1/48, חציון 10.5 ש'.
# 3.8 לא נבדק בחשיבה נמוכה (עומס אצל Google כל 20/10), ולכן לפי החלטת הבעלים: 3.8 (המודל הראשי) בחשיבה רגילה, כדי לא
# לקחת סיכון על טעויות ניסוח בשאלות המנוסחות; מודל הגיבוי (3.6, אחרי 3 שניות) בחשיבה נמוכה - הוא שומר על הזמן.
UNDERSTAND_LOW_THINKING = os.getenv("UNDERSTAND_LOW_THINKING", "false").strip().lower() in ("1", "true", "yes", "on")
UNDERSTAND_HEDGE_LOW_THINKING = os.getenv("UNDERSTAND_HEDGE_LOW_THINKING", "true").strip().lower() in ("1", "true", "yes", "on")
# מדידת "לפני ואחרי" על אותו שרת (סעיף 6ה): ?dev=1&legacy=1 עונה בלי שיפורי הזמנים - בלי מרוץ, בלי חיפוש מוקדם,
# חשיבה רגילה בהבנה, תיקון כבר מ-5 שניות, ובלי מטמון. לשאלה אחת בלבד (contextvar) - לא משנה לאף אחד אחר.
LEGACY: contextvars.ContextVar[bool] = contextvars.ContextVar("legacy", default=False)


@contextlib.contextmanager
def legacy_mode(on: bool):
    token = LEGACY.set(bool(on))
    try:
        yield
    finally:
        LEGACY.reset(token)


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
