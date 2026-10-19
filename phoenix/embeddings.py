"""אמבדינגים: Gemini ו-Voyage.

חשוב: וקטורים של מודלים שונים לא ניתנים להשוואה. לכן לכל ספק יש אינדקס משלו
(בונים ב-phoenix/index.py), ושאלה מוטמעת תמיד באותו מודל שבנה את האינדקס שבו מחפשים.

מסמנים לכל ספק אם הטקסט הוא שאלה או מסמך - כל ספק בדרך שלו:
  - Voyage: הפרמטר input_type.
  - gemini-embedding-2: מתעלם מהפרמטר task_type (נבדק: אותו וקטור בדיוק). לפי התיעוד
    מסמנים בתחילית בתוך הטקסט: "task: search result | query: ..." / "title: ... | text: ...".
    בלי זה השאלה והמסמכים הוטמעו באותה צורה, והחיפוש הווקטורי מצא רק 5/20.
"""
import httpx
from google.genai import types

from . import config
from .fallback import Provider, run_chain

# איך הטקסט נשלח להטמעה. index.py שומר את זה באינדקס, ואינדקס שנבנה בפורמט אחר נבנה מחדש.
INPUT_FORMAT = {"gemini": "prefix-v1", "voyage": "input-type"}
LEGACY_FORMAT = "input-type"  # אינדקסים שנבנו לפני שהפורמט נשמר


def _same_length(vectors: list, texts: list[str]) -> list:
    if len(vectors) != len(texts):
        raise ValueError(f"got {len(vectors)} vectors for {len(texts)} texts")
    return vectors


def gemini_input(text: str, is_query: bool) -> str:
    """התחילית ש-gemini-embedding-2 מצפה לה. קטע = שורות כותרת, שורה ריקה, גוף."""
    if is_query:
        return f"task: search result | query: {text}"
    header, sep, body = text.partition("\n\n")
    if not sep:
        return f"title: none | text: {text}"
    title = " › ".join(line for line in header.split("\n") if line) or "none"
    return f"title: {title} | text: {body}"


def _gemini(model, number=1):
    uses_prefix = model.startswith("gemini-embedding-2")  # gemini-embedding-001 עובד עם task_type

    def call(texts, is_query, timeout=None):
        if uses_prefix:
            inputs, cfg = [gemini_input(t, is_query) for t in texts], None
        else:
            inputs = texts
            cfg = types.EmbedContentConfig(task_type="RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT")
        if timeout:
            cfg = (cfg or types.EmbedContentConfig()).model_copy(
                update={"http_options": types.HttpOptions(timeout=int(timeout * 1000))})
        resp = config.gemini_client(number).models.embed_content(
            model=model,
            # gemini-embedding-2 מאחד רשימת מחרוזות לוקטור אחד - לכן כל טקסט כ-Content נפרד
            contents=[types.Content(parts=[types.Part(text=t)]) for t in inputs],
            config=cfg,
        )
        return _same_length([e.values for e in resp.embeddings], texts)
    return call


def _voyage(model):
    def call(texts, is_query, timeout=None):
        r = httpx.post(
            "https://api.voyageai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {config.API_KEYS['voyage']}"},
            json={"model": model, "input": texts, "input_type": "query" if is_query else "document"},
            timeout=timeout or config.HTTP_TIMEOUT,
        )
        r.raise_for_status()
        return _same_length([d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"])], texts)
    return call


_FACTORIES = {"gemini": _gemini, "voyage": _voyage}

PROVIDERS = [
    # Gemini: אותו מודל בכל המפתחות - אותם וקטורים, אותו אינדקס (fnx_gemini). מכסה של מפתח נגמרה - המפתח הבא
    provider
    for p, m in config.EMBED_CHAIN
    for provider in ([Provider(f"gemini:{m}#{n}", _gemini(m, n), group=f"gemini:{m}", key=f"gemini#{n}")
                      for n, _ in config.GEMINI_KEYS] if p == "gemini"
                     else [Provider(f"{p}:{m}", _FACTORIES[p](m), enabled=bool(config.API_KEYS[p]))])
]


def embed_query(text: str, providers: list[str] | None = None) -> tuple[list[float], str]:
    """וקטור לשאלה מהספק הראשון שעובד. providers: שמות ספקים לפי סדר עדיפות (ברירת מחדל: כולם)."""
    order = providers or [p.name for p in PROVIDERS]
    providers = sorted((p for p in PROVIDERS if p.name in order), key=lambda p: order.index(p.name))
    vectors, provider = run_chain("embed_query", providers, [text], True, timeout=config.QUERY_TIMEOUT,
                                  retries=config.QUERY_RETRIES, max_wait=config.QUERY_MAX_WAIT,
                                  deadline=config.QUERY_TIMEOUT)
    return vectors[0], provider
