"""הגדרות הפרויקט: מפתחות API מתוך .env, וסדר הספקים לכל מטרה.

כדי לשנות את סדר הגיבוי - משנים את הסדר ברשימות למטה.
"""
import contextlib
import contextvars
import json
import os
import re
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
# כל מספר שהוא (21/10: נוספו 21-28; קודם נקראו רק 2-20).
def _gemini_keys() -> list[tuple[int, str]]:
    numbered = []
    for name, value in os.environ.items():
        match = re.fullmatch(r"GEMINI_API_KEY_(\d+)", name)
        if match and int(match.group(1)) >= 2:
            numbered.append((int(match.group(1)), value.strip()))
    return [(number, key) for number, key in [(1, API_KEYS["gemini"].strip())] + sorted(numbered) if key]


GEMINI_KEYS: list[tuple[int, str]] = _gemini_keys()
# הבוט - מפתח Gemini אחד, בלי סבב מפתחות, למודלי השפה ולווקטור לשאלה: GEMINI_API_KEY (מספר 1; אם הוא ריק - הראשון שיש).
# ביום הבדיקות משדרגים את הפרויקט של המפתח הזה לתשלום, ושום דבר בקוד לא משתנה. LLM_GEMINI_KEY=<מספר> - מפתח אחר (בדיקות
# על מפתחות חינמיים: eval_batches.py נותן לכל חבילה מפתח אחר). המאגר (GEMINI_KEYS) - רק לכלים: הבדיקות ובניית האינדקס.
LLM_GEMINI_KEY: int | None = (int(os.getenv("LLM_GEMINI_KEY")) if os.getenv("LLM_GEMINI_KEY", "").strip().isdigit()
                              else GEMINI_KEYS[0][0] if GEMINI_KEYS else None)

# כל מודלי השפה שהבוט מכיר. מי עונה בכל שלב ובאיזה סדר - ROUTES (למטה), דרך שער המודלים (gateway.py). השרשרת המלאה
# משמשת רק קריאה בלי נתיב (בדיקת ספקים), ו-Groq/OpenRouter - רק את השופט בבדיקות.
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
# תשובה). מקבל את ההקשר המלא (5 קטעים) ואת אותן בדיקות. בדרגת הגיבוי של נתיב התשובה ושל נתיב ההבנה (ROUTES);
# Google לא עונה - הוא עונה. נמדד 21/10 על פרומפט התשובה האמיתי: command-a-03-2025 - 10.6 ש',
# command-a-plus - 9.9 ש', command-r-plus - 46 ש'; שעה אחר כך command-a - 37-43 ש' (Trial בעומס) ו-command-a-plus - 8.6 ש'.
# לכן command-a-plus. Groq לא מתאים (8K טוקנים לדקה, ההקשר ~11K). ריק - בלי גיבוי.
BACKUP_LLM = os.getenv("BACKUP_LLM", "cohere:command-a-plus-05-2026").strip()
# command-a-plus "חושב" לפני התשובה: נמדד 21/10 - 4,082 טוקני חשיבה מול ~470 של התשובה, 12-25 ש' ולפעמים יותר מ-90.
# עם תקציב חשיבה של 300 טוקנים - 4-5 ש' ברוב השאלות, אותן תשובות. "disabled" שובר את ה-JSON (נבדק).
COHERE_THINKING_BUDGET = int(os.getenv("COHERE_THINKING_BUDGET", "300"))
# שלב ההבנה (סיווג וניסוח - JSON קצר): תקציב קטן יותר. נמדד 22/10 05:20 על פרומפט ההבנה: 300 - 4.3-5.1 ש' (ולפעמים יותר
# מ-8, מגבלת ההבנה, והשאלה עברה לכללי הקוד); 100 - 2.5 ש', JSON תקין. רק בהבנה (low_thinking), לא בתשובה.
COHERE_LOW_THINKING_BUDGET = int(os.getenv("COHERE_LOW_THINKING_BUDGET", "100"))
COHERE_THINKING_MODELS = ("command-a-plus", "command-a-reasoning")
if BACKUP_LLM:
    LLM_CHAIN.append(tuple(BACKUP_LLM.split(":", 1)))

# שער המודלים (gateway.py): לכל שלב נתיב - מודל ראשי, ואחריו גיבויים לפי הסדר. "|" - מכאן דרגת גיבוי: תשובה שלהם מחכה
# למודלים שלפני ה-"|" (PREFER_SECONDS). "~low" - חשיבה נמוכה (LOW_THINKING; ב-Cohere - COHERE_LOW_THINKING_BUDGET).
# "BACKUP" - BACKUP_LLM (ריק - בלי). כל נתיב אפשר להחליף במשתנה סביבה.
# - הבנת השאלה: 3.8 בחשיבה רגילה (החלטת הבעלים 20/10 - פחות טעויות ניסוח); תקלה או 3 ש' בלי תשובה - 3.6 בחשיבה נמוכה
#   (2.7 ש' בעומס, כש-3.8 החזיר 504 אחרי 30 ש') ומודל הגיבוי מחוץ ל-Google, יחד.
# - תשובה: 3.6 (המודל שנמדד באיכות); תקלה או 3 ש' בלי תשובה - 3.8 ומודל הגיבוי מחוץ ל-Google, יחד. תשובה של 3.8
#   מנצחת מיד (Gemini), ושל Cohere מחכה ל-Gemini עד 12 ש' (החלטת הבעלים 23/10: Gemini שני לפני Cohere).
# - קריאות עזר (בדיקת נושא, השלמת שאלת המשך, ניסוח מחדש, ניסיון התיקון): Gemini בלבד, כמו קודם.
ROUTES = {
    "understand": os.getenv("UNDERSTAND_ROUTE", "gemini:gemini-3.8-flash, gemini:gemini-3.6-flash~low | BACKUP~low"),
    "answer": os.getenv("ANSWER_ROUTE", "gemini:gemini-3.6-flash, gemini:gemini-3.8-flash | BACKUP"),
    "helper": os.getenv("HELPER_ROUTE", "gemini:gemini-3.6-flash, gemini:gemini-3.8-flash"),
    # בדיקת העיגון (GROUNDING): 3.8 - מודל אחר מזה שבדרך כלל עונה (3.6), כך שהבודק לא "מאשר את עצמו"
    "verify": os.getenv("VERIFY_ROUTE", "gemini:gemini-3.8-flash, gemini:gemini-3.6-flash | BACKUP"),
}


def route(name: str) -> list[tuple[str, bool, int]]:
    """[(ספק או מודל, חשיבה נמוכה, דרגה)] לפי הסדר. נקרא בכל קריאה - שינוי של BACKUP_LLM (גם בבדיקות) נכנס מיד."""
    lanes = []
    for tier, group in enumerate(ROUTES[name].split("|")):
        for part in group.split(","):
            target, _, extra = part.strip().partition("~")
            target = BACKUP_LLM if target == "BACKUP" else target
            if target:
                lanes.append((target, extra.strip() == "low", tier))
    return lanes


# בצ'אט: הראשי לבד; אחרי HEDGE_AFTER_SECONDS בלי תשובה, או מיד כשהוא נתקל בתקלה (שגיאה, תשובה שנפסלה, מפסק פתוח) - כל
# שאר הנתיב מתחיל יחד.
HEDGE_AFTER_SECONDS = float(os.getenv("HEDGE_AFTER_SECONDS", "3"))
# תשובה טובה של גיבוי מחכה למודל מוקדם יותר בנתיב עד כמה שניות מההתחלה (Cohere לבד - 2 שגויות ב-45 שאלות, Gemini - 0):
# בתשובה - 12 ש' (21/10); בהבנה - 6 ש' (3 עד הגיבוי + 3 חסד, 21/10). ולא קרוב יותר מ-PREFER_MARGIN לסוף הזמן.
PREFER_SECONDS = {"answer": float(os.getenv("ANSWER_PREFER_SECONDS", "12")),
                  "understand": float(os.getenv("UNDERSTAND_PREFER_SECONDS", "6"))}
PREFER_MARGIN = {"answer": 1.5, "understand": 0.5}
# ניסיון חוזר באותו מודל - רק על תקלה מהירה (429 לדקה, 503), ורק אם ההמתנה עד LLM_MAX_WAIT (בצ'אט המודל הבא כבר רץ).
LLM_RETRIES = 1
LLM_MAX_WAIT = 2.0

# ב2 (23/10) - זיכרון של כל השיחה. קודם שלב ההבנה ראה רק את 6 ההודעות האחרונות, וסוג הביטוח נשכח ("ואבא שלי בן 70, הוא
# חייב למלא טופס?" אחרי שאלה על טוטאל לוס - החיפוש הביא תביעות פטירה). דלוק:
# 1. שלב ההבנה מקבל את כל השיחה (understand.conversation_text): כל הודעות הלקוחה, ותשובות הבוט מקוצרות; מעל
#    CONVERSATION_CHARS - קודם מתקצרות תשובות ישנות של הבוט, וההודעה הראשונה של הלקוחה נשארת תמיד.
# 2. זיכרון נושא (answer.next_memory): סוג הביטוח והנושא נשמרים לאורך השיחה, בתוך ההודעות עצמן (שורד רענון).
# 3. כללים בקוד: שאלת המשך שהמודל השמיט ממנה את הסוג - יורשת אותו; סוג אחר בהודעה מחליף אותו; ניסיון לשנות את
#    ההוראות לא נשאר בשיחה; אין שתי שאלות הבהרה ברצף. כבוי עד A/B על סט השיחות (data/conversations.json, כלל ג2).
CONVERSATION_MEMORY = os.getenv("CONVERSATION_MEMORY", "false").strip().lower() in ("1", "true", "yes", "on")
CONVERSATION_CHARS = int(os.getenv("CONVERSATION_CHARS", "20000"))  # ~8K טוקנים; שיחה של 40 שאלות נכנסת כמעט כולה

# שלב הבנת השאלה (understand.py): עד UNDERSTAND_TIMEOUT שניות בצ'אט, ואחריו כללי הקוד; הבדיקות - בלי מגבלה
UNDERSTAND_TIMEOUT: float | None = float(os.getenv("UNDERSTAND_TIMEOUT", "8"))

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
# ספים מכוילים (calibrate_thresholds.py -> data/thresholds.json, 26/10): הקובץ קובע, לפני משתני הסביבה ולפני ערכי ברירת
# המחדל שלמעלה - כך סף ישן מ-.env לא עוקף בשקט כיול חדש (thresholds.warnings מתריע כשהם סותרים). בלי הקובץ - כמו קודם.
# הקובץ שומר גם על איזה אינדקס ואיזה מודל דירוג הסף כויל (thresholds.status): השתנו - אזהרה, והבדיקה האוטומטית נכשלת.
THRESHOLDS_FILE = ROOT / "data" / "thresholds.json"


def _calibrated_thresholds() -> dict[str, float]:
    try:
        data = json.loads(THRESHOLDS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {name: float(entry["threshold"]) for name, entry in data.items()
            if isinstance(entry, dict) and isinstance(entry.get("threshold"), (int, float))}


RERANK_THRESHOLDS.update(_calibrated_thresholds())
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
# הווקטור לשאלה (Gemini embeddings) - בדרך כלל 0.2-1 ש' (בשאלה הראשונה אחרי הפעלה - עד ~4.4). נבדק 21/10 עם Google תקוע:
# הוא חיכה 16 ש' לפני המעבר לחיפוש מילות מפתח בלבד, ולא נשאר זמן לתשובה. בצ'אט - עד EMBED_QUERY_TIMEOUT שניות (שעון
# קיר; ה-timeout של הבקשה נשאר QUERY_TIMEOUT, כי Gemini לא מקבל פחות מ-10), ואחריו מילות מפתח בלבד.
EMBED_QUERY_TIMEOUT = float(os.getenv("EMBED_QUERY_TIMEOUT", "6"))
QUERY_RETRIES = 0
QUERY_MAX_WAIT = 0.0
# זמן כולל לשאלה אחת בצ'אט - כל השלבים יחד (השלמת שאלת המשך, חיפוש, Rerank, מודל). כל ניסיון אצל ספק מקבל
# רק את מה שנשאר; כשנשארות פחות מ-3 שניות - הודעה ידידותית, בלי ניסיון נוסף. (הבדיקות לא מוגבלות.)
QUESTION_TIMEOUT = float(os.getenv("QUESTION_TIMEOUT", "20"))
# 20 השניות הן מגבלה קשיחה. בצ'אט כל קריאה למודל עוברת בנתיב שלה במקביל מדורג (HEDGE_AFTER_SECONDS, למעלה): מודל עמוס
# או איטי לא מעכב את התשובה. false - בזה אחר זה גם בצ'אט (מתג חירום). הבדיקות (eval) תמיד בזה אחר זה.
PARALLEL_LLM = os.getenv("PARALLEL_LLM", "true").strip().lower() in ("1", "true", "yes", "on")
# "עובדות קודם, ואז התשובה" (21/10, לפי ההנחיה של Anthropic: לחלץ קודם ציטוטים מילה במילה, ולענות רק מהם). נמצא
# בבדיקה: ב-33 מתוך 35 תשובות חלקיות הפרט החסר היה בקטעים שהמודל קיבל - המודל השמיט אותו ("עד 5 משפטים"). בגרסה
# הזו המודל מעתיק קודם כל משפט רלוונטי (evidence), ואז כותב תשובה שכוללת את כולם; והקוד מוסיף עובדה מאומתת עם
# מספר שחסרה בתשובה. false - ההוראות הקודמות. נמדד 22/10 (A/B מזווג: כל שאלה בשתי הגרסאות ברצף, אותו חיפוש, 3.7,
# 53 שאלות עם תשובה באתר ואישיות): מלאות 31 -> 36, חלקיות 19 -> 14, שגויות 1 -> 1 (אותה שאלה). דלוק (כלל ההחלטה של הבעלים).
ANSWER_FACTS = os.getenv("ANSWER_FACTS", "true").strip().lower() in ("1", "true", "yes", "on")
# ב3 (23/10): מספר שהלקוחה כתבה ("אבא שלי בן 70", כשבאתר כתוב 67) נפסל בבדיקת המספרים, והתשובה הפכה ל"לא הצלחתי לאמת".
# הבדיקה לא מקלה - כל מספר בתשובה חייב להופיע בקטעים, גם אם הלקוחה כתבה אותו: אחרת המודל יכול "לאשר" מספר שגוי שהלקוחה
# הציעה ("נכון שההחזר 500 ש"ח?" -> "כן, 500"). במקום זה, כלל 12 בהוראות: לא לחזור על מספר של הלקוחה (להתייחס אליו
# במילים) ולא לאשר אותו אם הוא לא בקטעים. ניסיון התיקון יודע תמיד שהמספר הוא של הלקוחה. כבוי עד A/B (כלל ג2).
CUSTOMER_NUMBERS_RULE = os.getenv("CUSTOMER_NUMBERS_RULE", "false").strip().lower() in ("1", "true", "yes", "on")
# ב3 - חור שקיים מ-16/10: מספר בתשובה עובר אם הוא מופיע באחד מ-5 הקטעים, גם בקטע שהתשובה לא מבוססת עליו ("נכון שההחזר
# 500 ש"ח?" עובר אם 500 מופיע בקטע אחר). דלוק: כל מספר בתשובה חייב להופיע בקטעים שהמודל ציין ב-sources (או בפרטי
# ההתקשרות). כבוי עד A/B - יכול לפסול תשובה שמשלבת מספר מקטע שהמודל שכח לציין.
NUMBERS_FROM_CITED = os.getenv("NUMBERS_FROM_CITED", "false").strip().lower() in ("1", "true", "yes", "on")
# ב1 (23/10) - תשובות חלקיות. "עובדות קודם" טיפל במודל שמשמיט פרט מהקטעים; כאן - פרט שלא הגיע אליו בכלל:
# - CONTEXT_FLEX: עד CONTEXT_MAX_CHUNKS קטעים במקום 5 - רק קטעים שעברו את סף ה-Rerank (רלוונטיים כמו שאנחנו דורשים
#   מהראשון), אחרי 5 הראשונים ("Lost in the Middle": הרלוונטיים ביותר בהתחלה).
# - CONTEXT_PAGES (small-to-big): קטעים מאותו עמוד - בלוק אחד, לפי הסדר בעמוד, ועוד הקטע הסמוך (לפני ואחרי) בעמוד
#   מחולק (211 עמודים, 508 קטעים): ההסתייגות או המספר שחסרים בדרך כלל צמודים לקטע שנמצא.
# שניהם בתוך CONTEXT_CHARS (5 הקטעים הראשונים - תמיד). יותר קטעים = יותר מספרים "מותרים" בבדיקת המספרים - לכן כדאי
# להפעיל איתם את NUMBERS_FROM_CITED. ההקשר הקבוע לכל קטע (שם העמוד ושרשרת הכותרות בתחילת הקטע - "Contextual Retrieval"
# בגרסה בלי מודל) קיים כבר מבניית האינדקס (chunk.py). כבויים עד A/B (כלל ג2).
CONTEXT_FLEX = os.getenv("CONTEXT_FLEX", "false").strip().lower() in ("1", "true", "yes", "on")
CONTEXT_MAX_CHUNKS = int(os.getenv("CONTEXT_MAX_CHUNKS", "8"))
CONTEXT_PAGES = os.getenv("CONTEXT_PAGES", "false").strip().lower() in ("1", "true", "yes", "on")
CONTEXT_CHARS = int(os.getenv("CONTEXT_CHARS", "24000"))  # היום: 5 קטעים ~11K תווים
# ב4 (23/10): ניסיון עקיפה נתפס בכללים בקוד לפני כל מודל (answer.INJECTION_RULES) - רצפה גם כשהמודלים למטה. דלוק: הכללים
# דטרמיניסטיים ונמדדו בלי API - 0 חסימות שגויות על 194 השאלות הלגיטימיות שיש לנו ועל רשימת ניסוחים מסוכנים בבדיקות.
INJECTION_RULES = os.getenv("INJECTION_RULES", "true").strip().lower() in ("1", "true", "yes", "on")
# ב5 + עיגון (23/10). GROUNDING: בדיקת עיגון לכל טענה (grounding.py) - כל משפט בתשובה צריך ציטוט מהאתר, והקוד מאמת את
# הציטוט. משפט לא נתמך - יורד, ונאמר במפורש שחלק מהתשובה לא אומת ("תשובה חלקית מאומתת", במקום "לא הצלחתי לאמת"); גם
# תשובה שהבדיקות בקוד פסלו (מספר / ציטוט) - מה שמאומת בה מוצג. קריאה נוספת (~3-5 ש', ~סנט); פחות מ-GROUNDING_MIN_SECONDS
# לשאלה - בלי הבדיקה (נרשם). ANSWER_STRUCTURE: המשפט הראשון עונה ישירות (כן/לא, סכום, זמן, פעולה). כבויים עד A/B.
GROUNDING = os.getenv("GROUNDING", "false").strip().lower() in ("1", "true", "yes", "on")
GROUNDING_MIN_SECONDS = float(os.getenv("GROUNDING_MIN_SECONDS", "5"))
ANSWER_STRUCTURE = os.getenv("ANSWER_STRUCTURE", "false").strip().lower() in ("1", "true", "yes", "on")
# אימות קפדני (23/10, תיקון חורים שנמצאו בבדיקה): ציטוט "כמעט מילה במילה" נבדק במשפט אחד ולפי הסדר, ומילים שהופכות
# משמעות ("לא", "רק", "עד") זהות בציטוט ובאתר (קודם "מכוסים" עבר מול "לא מכוסים"); ומספר שכתוב במילים ("שלושים") נבדק
# כמו מספר בספרות. דלוק: סוגר חורים שבהם תשובה שגויה הוצגה כמאומתת. false - ההתאמה הישנה (למדידה מול הגרסה הקודמת).
STRICT_VERIFICATION = os.getenv("STRICT_VERIFICATION", "true").strip().lower() in ("1", "true", "yes", "on")
# המונח בקטעים שצוטטו (26/10, docs/before-paid-tests.md סעיף 2): אחרי הסריקה המלאה כמעט כל מילה "נמצאת איפשהו באתר"
# ("טרקטורון" ברשימת ספורט מסוכן), ולכן הבדיקה לכל האתר כמעט לא חוסמת. דלוק: המונח ששלב ההבנה הוציא (key_term) חייב
# להופיע בקטע שהתשובה מצטטת - כל מילות התוכן שלו באותו קטע, בצורה המתוקנת או כמו שהלקוחה כתבה. רק מחמיר: הבדיקה לכל
# האתר נשארת לפניה. נפסלה - ניסיון תיקון אחד עם הסיבה; לא תוקנה - "לא מופיע באתר". כבוי עד A/B (כלל ג2).
TERMS_IN_CITED = os.getenv("TERMS_IN_CITED", "false").strip().lower() in ("1", "true", "yes", "on")
# ציטוט לפי מספר משפט (26/10, docs/before-paid-tests.md סעיף 3): כל משפט בקטעים ממוספר ([2.5] - החמישי בקטע 2), המודל
# מחזיר ב-evidence מספרים במקום להעתיק טקסט, והקוד שולף את המשפטים מהקטעים - טעות העתקה (שני משפטים מחוברים, מילה
# שהשתנתה) בלתי אפשרית, כמו ב-Citations API של Anthropic. טקסט במקום מספרים - האימות הרגיל. כבוי עד A/B (כלל ג2).
EVIDENCE_BY_ID = os.getenv("EVIDENCE_BY_ID", "false").strip().lower() in ("1", "true", "yes", "on")
# כמה שאלות בהודעה אחת (26/10, docs/before-paid-tests.md סעיף 4): שלב ההבנה כבר מפרק אותן (sub_questions), אבל עונים רק
# על הראשונה והשאר כפתורים. דלוק: עד MULTI_MAX_PARTS שאלות במקביל - לכל אחת חיפוש, סף, מודל תשובה, כל הבדיקות ועיגון
# משלה - ותשובה אחת עם חלק לכל שאלה. חלק שלא הסתיים בזמן, או שחסר בו סוג ביטוח - כפתור, כמו היום. כבוי עד A/B (כלל ג2).
MULTI_ANSWER = os.getenv("MULTI_ANSWER", "false").strip().lower() in ("1", "true", "yes", "on")
MULTI_MAX_PARTS = int(os.getenv("MULTI_MAX_PARTS", "3"))
# כל המתגים שמשנים תשובה - חלק מגרסת הקוד במטמון התשובות המאומתות (answer_cache): תשובה מגרסה אחת לא מוצגת בשנייה.
BEHAVIOR_FLAGS = ("ANSWER_FACTS", "ANSWER_LOW_THINKING", "SITE_QUOTE", "CUSTOMER_NUMBERS_RULE", "NUMBERS_FROM_CITED",
                  "INJECTION_RULES", "GROUNDING", "ANSWER_STRUCTURE", "STRICT_VERIFICATION", "TERMS_IN_CITED",
                  "EVIDENCE_BY_ID", "MULTI_ANSWER", "MULTI_MAX_PARTS", "CONVERSATION_MEMORY", "CONTEXT_FLEX",
                  "CONTEXT_MAX_CHUNKS", "CONTEXT_PAGES", "CONTEXT_CHARS")
# הציטוט מהאתר (כשאף מודל לא ענה) - רק כשמודל הבין את השאלה. בלי הבנה (כללי הקוד) - העמוד והפניה: בריצה של 21/10
# ציטוט כזה "ענה" על ניסיון injection ועל שאלה בלי תשובה באתר.
SITE_QUOTE_ONLY_UNDERSTOOD = os.getenv("SITE_QUOTE_ONLY_UNDERSTOOD", "true").strip().lower() in ("1", "true", "yes", "on")
# הציטוט מהאתר כשאף מודל לא ענה - כבוי (החלטת הבעלים 21/10, "רק אם זה משפר את האמינות"): על 43 ציטוטים מהלילה (הריצה
# הסופית, הסט הנסתר וה-A/B) - 7 טעויות חמורות (שגויה / המציא / injection שלא נחסם), ובמקומם "העמוד והפניה" - 0. המחיר:
# 6 מלאות ו-30 חלקיות נהיות "לא הצלחתי לאמת". אין מודל - תמיד העמוד הרלוונטי והפניה לשירות הלקוחות.
SITE_QUOTE = os.getenv("SITE_QUOTE", "false").strip().lower() in ("1", "true", "yes", "on")
# בצ'אט: החיפוש על השאלה כמו שנכתבה מתחיל במקביל להבנת השאלה. ניסוח עצמאי שונה מהותית (פחות מ-70% מילים
# משותפות, סוג ביטוח אחר, שאלה מפוצלת, תרגום) - מחפשים שוב עליו. שאלת המשך שתלויה בשיחה - בלי חיפוש מוקדם.
PARALLEL_SEARCH = os.getenv("PARALLEL_SEARCH", "true").strip().lower() in ("1", "true", "yes", "on")
SAME_SEARCH_JACCARD = 0.7
# חשיבה: הרמה הנמוכה ביותר שכל מודל מקבל (נבדק 20/10: 3.6 - MINIMAL; 3.7 ו-3.8 דוחים MINIMAL - LOW). שלב ההבנה
# לפי "~low" בנתיב שלו (ROUTES). בתשובה - רק עם ANSWER_LOW_THINKING, אחרי בדיקת איכות.
# נבדק 20/10, הבנה בלבד על 75 השאלות, 3.6: נמוכה - שגויה 1/75, חציון 3.5 ש'; רגילה - שגויה 1/48, חציון 10.5 ש'.
LOW_THINKING = {"gemini-3.6-flash": "MINIMAL", "gemini-3.7-flash": "LOW", "gemini-3.8-flash": "LOW"}
ANSWER_LOW_THINKING = os.getenv("ANSWER_LOW_THINKING", "false").strip().lower() in ("1", "true", "yes", "on")
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
    global QUERY_TIMEOUT, QUERY_RETRIES, QUERY_MAX_WAIT, EMBED_QUERY_TIMEOUT, LLM_RETRIES, LLM_MAX_WAIT
    QUERY_TIMEOUT, QUERY_RETRIES, QUERY_MAX_WAIT = timeout, retries, max_wait
    LLM_RETRIES, LLM_MAX_WAIT = retries, max_wait
    EMBED_QUERY_TIMEOUT = timeout
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
