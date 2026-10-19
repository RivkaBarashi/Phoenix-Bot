"""ממשק צ'אט לבוט של אתר הפניקס (Streamlit).

הרצה:  .venv\\Scripts\\streamlit run app.py   (או לחיצה כפולה על run_chat.bat)

הממשק רק מציג את השיחה וקורא ל-phoenix.answer.answer. כל ההחלטות - חיפוש, סף, ההוראות
למודל והבדיקות בקוד - נשארות בקוד של הבוט ולא משתנות מכאן.

בדיקות ממשק בלי לבזבז את המכסה של Gemini: CHAT_LLM_PROVIDERS=groq,openrouter ב-.env.
ריק או לא מוגדר = שרשרת המודלים הרגילה (Gemini קודם). ההגדרה משפיעה רק על הממשק -
הבדיקות (eval_*.py) לא קוראות אותה.

שאלות המשך: השיחה עד עכשיו (בלי הודעת הפתיחה ובלי הודעות שגיאה) נשלחת לבוט עם כל שאלה. הבוט
משתמש בה רק כשהשאלה תלויה בשיחה ("ותוך כמה זמן?") - ראו answer.needs_context.

יומן ומשוב: כל שאלה נרשמת ב-logs/conversations.jsonl, ובענן גם ב-Upstash - משותף לשני האתרים (report.py
מסכם; --cloud - היומן שבענן). מתחת לכל תשובה 👍/👎 שנשמר ביומן.

שמירת השיחה ברענון: לכל שיחה מזהה בכתובת (?c=...), והשיחה נשמרת ב-Upstash ל-24 שעות (מספרים ארוכים שהלקוחה
כתבה - מוסתרים, כמו ביומן). רענון הדף - השיחה חוזרת. בלי Upstash - הבוט עובד כרגיל, רק בלי לשמור. בתשובה אישית, "לא מופיע באתר" או תקלה - כפתור להמשך עם נציג בוואטסאפ, עם הודעה מוכנה.
"פרטים טכניים" מתחת לכל תשובה (החלטה, ציון, מודל, זמנים) - רק במצב מפתחת: ?dev=1 בכתובת.

שאלת הבהרה: שאלה על ביטוח בלי סוג ("איך מגישים תביעה?") מקבלת "על איזה ביטוח מדובר?" עם כפתורים.
לחיצה שולחת את השאלה המקורית + הסוג ("איך מגישים תביעה בביטוח רכב?"), כאילו הלקוחה כתבה אותה.

חיווי בשלבים: בזמן התשובה הבועה של הבוט מראה את השלב האמיתי - חיפוש, דירוג, ניסוח, אימות. answer.answer
קורא ל-on_stage בתחילת כל שלב, ולכן שלב מתחלף רק כשהקודם באמת הסתיים (בלי טיימרים). ברכה ושאלת הבהרה -
בלי שלבים; תשובה מהמטמון - בלי ניסוח ואימות. מעבר למודל גיבוי לא מוצג ללקוחה, רק ב"פרטים טכניים" וביומן.
"""
import html
import json
import logging
import os
import re
import time
import uuid
from functools import partial
from urllib.parse import quote

import streamlit as st

from phoenix import answer as bot
from phoenix import bm25, config, conversation_log, embeddings, index, llm, store

log = logging.getLogger("phoenix.app")

TEST_PROVIDERS = [p.strip() for p in os.getenv("CHAT_LLM_PROVIDERS", "").split(",") if p.strip()]
EXAMPLES = [
    "יש לי נזילה בבית, איך מגישים תביעה?",
    "תוך כמה זמן מטפלים בתביעת מחלות קשות?",
    "מה המספר של מוקד תביעות רכב ורכוש?",
]
WELCOME = ("שלום! אני הבוט של אתר הפניקס. אני עונה רק לפי המידע שבאתר fnx.co.il, ומצרף לכל תשובה "
           "קישור לעמוד המקור. כשהמידע לא מופיע באתר - אגיד זאת ואפנה לשירות הלקוחות. "
           "אין לי גישה למידע האישי שלכם (כיסוי, יתרה או תביעות). אפשר לשאול למשל:")
CONTACT_TITLE = "כל הדרכים ליצור איתנו קשר"
STAGE_NAMES = {"understand": "הבנת השאלה", "condense": "השלמת שאלה", "vector": "חיפוש וקטורי", "bm25": "מילות מפתח", "rerank": "Rerank",
               "search_wait": "המתנה לחיפוש המוקדם", "llm": "מודל", "repair": "תיקון", "total": "סה\"כ"}
EARLY_SEARCH = {"reused": "החיפוש המוקדם (במקביל להבנה) שימש לתשובה",
                "redone": "הניסוח העצמאי שונה מהותית - חיפוש שני עליו"}
# החיווי בשלבים: שורה ראשית ללקוחה, ומתחתיה המונח המקצועי באפור. pages = כמה עמודים שונים נמצאו בחיפוש.
STAGE_LINES = {
    "understand": ("💭 קורא את השאלה...", "הבנת השאלה: מודל שפה"),
    "search": ("🔍 מחפש את התשובה באתר הפניקס...", "חיפוש משולב: משמעות + מילות מפתח"),
    "rerank": ("📑 נמצאו {pages} עמודים מתאימים. בודק איזה מהם עונה בדיוק על השאלה...", "דירוג רלוונטיות (Rerank)"),
    "compose": ("✍️ מנסח תשובה מתוך המידע שנמצא...", "מודל שפה, רק מתוך הקטעים מהאתר"),
    "verify": ("✅ מוודא שכל פרט בתשובה כתוב באתר...", "אימות ציטוט ומספרים מול המקור"),
    "below_threshold": ("🔎 בדקתי את העמודים הרלוונטיים, והמידע הזה לא מופיע באתר", "ציון הרלוונטיות מתחת לסף"),
}
# העברה לנציג אנושי: הוואטסאפ של שירות הלקוחות (054-9893455, מעמוד צור קשר באתר)
WHATSAPP_URL = "https://wa.me/972549893455"
HANDOFF_LABEL = "💬 המשך עם נציג בוואטסאפ"
OTHER_TYPE_LABEL = "אחר"
OTHER_TYPE_TEXT = "איזה ביטוח? כתבו את סוג הביטוח בתיבה למטה (למשל: ביטוח משכנתא, ביטוח לעסק)."
OTHER_TYPE_MAX_WORDS = 4  # אחרי "אחר": תשובה קצרה = סוג הביטוח; ארוכה יותר = שאלה חדשה
STREAK_NOTICE = "לא מצאתי באתר תשובה לשתי השאלות האחרונות. נציג אנושי יוכל לעזור - ההודעה בוואטסאפ כבר מוכנה:"
ERROR_MESSAGE = ("משהו השתבש ולא הצלחתי לענות כרגע. אפשר לנסות שוב בעוד רגע, "
                 "או לפנות לשירות הלקוחות של הפניקס בוואטסאפ 054-9893455.")
LOAD_ERROR = ("הבוט לא הצליח להיטען. צריך לוודא שהאינדקס נבנה (python -m phoenix.chunk ואז "
              "python -m phoenix.index) ולהפעיל את הממשק מחדש.")

RTL_CSS = """
<style>
.stApp, [data-testid="stChatMessage"], [data-testid="stChatInput"] { direction: rtl; text-align: right; }
[data-testid="stChatInput"] textarea { direction: rtl; text-align: right; }
.answer { line-height: 1.7; }
.answer bdi { white-space: nowrap; }  /* טלפון לא נשבר בין שורות אחרי המקף ("073-" ו-"2076100") */
.sources { font-size: 0.9rem; border-top: 1px solid rgba(128, 128, 128, 0.3); margin-top: 0.5rem; padding-top: 0.4rem; }
.stage-main { line-height: 1.7; animation: stage-pulse 1.6s ease-in-out infinite; }  /* הבוט עובד - לא תקוע */
.stage-term, .checked { font-size: 0.8rem; opacity: 0.65; }  /* אפור, גם בערכת צבעים כהה */
.checked { margin-top: 0.2rem; }
@keyframes stage-pulse { 50% { opacity: 0.55; } }
@media (prefers-reduced-motion: reduce) { .stage-main { animation: none; } }
</style>
"""

# טלפונים, קישורים, מיילים וקודי כוכבית מוצגים משמאל לימין בתוך הטקסט העברי, כדי שלא יתהפכו.
# מייל - רק אותיות לטיניות: ב"ל-Healthd@Fnx.co.il" ה-"ל-" שייכת לעברית ונשארת מימין למייל.
LTR_TOKEN = re.compile(
    r"(?P<url>https?://[^\s<>\"']*[^\s<>\"'.,;:!?)\]])"
    r"|(?P<email>[A-Za-z0-9][A-Za-z0-9._%+-]*@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)"
    r"|(?P<star>\*\d{3,5}\*?(?!\w)|\b\d{3,5}\*)"
    r"|(?P<phone>\b0\d{1,2}-?\d{3}-?\d{4}\b)"
)
BIDI_MARKS = re.compile("[\u200e\u200f\u2066-\u2069]")  # סימני כיוון שכבר בטקסט - כאן bdi עושה את העבודה
LRI, PDI = "\u2066", "\u2069"  # בידוד משמאל לימין בטקסט רגיל (בזמן ההקלדה) - כמו bdi ב-HTML
WORD_JOINER = "\u2060"  # אחרי המקף בטלפון: בלי שבירת שורה בזמן ההקלדה (בגרסה הסופית זה ב-CSS)


@st.cache_resource(show_spinner="טוען את הבוט...")
def load_bot() -> dict:
    """נטען פעם אחת לכל השרת - לא בכל שאלה ולא לכל משתמש: מסנן המודלים לבדיקות, אינדקס
    מילות המפתח, אינדקסי הווקטורים, פרטי ההתקשרות ולקוח Gemini. משתמש בפונקציות הטעינה של הבוט."""
    if TEST_PROVIDERS:  # מצב בדיקות ממשק: רק המודלים שנבחרו (לא נוגע בקוד של הבוט)
        llm.PROVIDERS[:] = [p for p in llm.PROVIDERS if p.name.split(":")[0] in TEST_PROVIDERS]
    chunks, _ = bm25._index()
    bot._site_prefixes()
    bot.contact_text()
    vectors = {}
    for p in embeddings.PROVIDERS:
        col = index.get_collection(p.name) if p.enabled else None
        vectors[p.name] = col.count() if col is not None else 0
    for number, _ in config.GEMINI_KEYS:  # לקוח לכל מפתח Gemini
        config.gemini_client(number)
    return {"chunks": len(chunks), "vectors": vectors, "llms": [p.name for p in llm.PROVIDERS if p.enabled]}


def ltr_segments(text: str):
    """הטקסט בקטעים: (None, טקסט רגיל) או (סוג, ערך) לקישור, מייל, קוד כוכבית או טלפון.
    קוד כוכבית מוצג כמו שמחייגים: "5432*" ו-"*5432*" -> "*5432"."""
    text = BIDI_MARKS.sub("", text)
    last = 0
    for m in LTR_TOKEN.finditer(text):
        yield None, text[last:m.start()]
        yield m.lastgroup, "*" + m.group(0).strip("*") if m.lastgroup == "star" else m.group(0)
        last = m.end()
    yield None, text[last:]


def to_html(text: str) -> str:
    """טקסט -> HTML בטוח: הכול עובר escape, קישורים נפתחים בלשונית חדשה, וטלפונים, קישורים, מיילים
    וקודי כוכבית עטופים ב-bdi כדי שיוצגו נכון ובסדר הנכון בתוך עברית."""
    parts = []
    for kind, value in ltr_segments(text):
        if kind is None:
            parts.append(html.escape(value))
        elif kind == "url":
            href = html.escape(value, quote=True)
            parts.append(f'<a href="{href}" target="_blank" rel="noopener noreferrer"><bdi dir="ltr">{html.escape(value)}</bdi></a>')
        else:
            parts.append(f'<bdi dir="ltr">{html.escape(value)}</bdi>')
    return "".join(parts).replace("\n", "<br>")


def chat_history(messages: list[dict]) -> list[dict]:
    """השיחה עד עכשיו, לשאלות המשך: שאלות ותשובות בלבד. הודעת הפתיחה לא נשמרת בשיחה, והודעות שגיאה
    ("משהו השתבש...") לא עוזרות להבין את השאלה הבאה. הודעה ארוכה מדי ובקשת הקיצור שאחריה - גם לא: היא לא נענתה."""
    return [{"role": m["role"], "content": m["content"]} for m in messages
            if (len(m["content"]) <= bot.MAX_QUESTION_CHARS if m["role"] == "user"
                else (m.get("meta") or {}).get("kind") not in ("error", "too_long"))]


def message_from_answer(a: bot.Answer, question: str = "") -> dict:
    """תוצאת הבוט -> הודעה לתצוגה: הטקסט בלי שורת המקור, והמקורות כקישורים עם שם העמוד."""
    text = a.text.rsplit("\n\nהמקור", 1)[0]  # format_sources מוסיף "המקור:" / "המקורות:" בסוף
    titles = {h.url: h.title for h in a.hits} | {bot.CONTACT_URL: CONTACT_TITLE}
    meta = {"kind": a.kind, "reason": a.reason, "llm": a.llm, "reranker": a.reranker, "top_score": a.top_score,
            "timings": a.timings, "question": question, "race": a.race, "pages": a.pages, "warning": a.warning,
            "early_search": a.early_search, "cached": a.cached,
            "understanding": a.understanding, "more_questions": a.more_questions,
            "standalone": a.question if a.question and question and a.question != question else ""}
    return {"role": "assistant", "content": text,
            "sources": [(titles.get(url, url), url) for url in a.sources], "meta": meta}


def error_message(reason: str, question: str = "") -> dict:
    return {"role": "assistant", "content": ERROR_MESSAGE, "sources": [(CONTACT_TITLE, bot.CONTACT_URL)],
            "meta": {"kind": "error", "reason": reason, "llm": None, "reranker": "-", "top_score": None,
                     "question": question}}


def _isolated(kind: str | None, value: str) -> str:
    """קטע בזרם ההקלדה: טלפון, קוד, מייל או קישור מבודדים משמאל לימין, וטלפון לא נשבר אחרי המקף."""
    if kind is None:
        return value
    if kind == "phone":
        value = value.replace("-", "-" + WORD_JOINER)
    return f"{LRI}{value}{PDI}"


def stream_words(text: str):
    """הצגה הדרגתית של תשובה שכבר עברה את כל הבדיקות. לא מזרימים ישר מהמודל: אחרי שהוא מסיים
    הקוד עוד בודק את הציטוט והמספרים, ותשובה שנפסלה לא אמורה להופיע על המסך אפילו לרגע.
    טלפונים וקודי כוכבית מבודדים משמאל לימין כבר בזמן ההקלדה, ולא רק בגרסה הסופית."""
    shown = "".join(_isolated(kind, value) for kind, value in ltr_segments(text))
    for piece in re.split(r"(\s+)", shown):
        yield piece.replace("*", "\\*").replace("_", "\\_")  # בלי עיצוב Markdown באמצע ההקלדה
        if piece.strip():
            time.sleep(0.03)


def render_type_buttons(message: dict):
    """כפתורי סוג הביטוח מתחת ל"על איזה ביטוח מדובר?" - רק בהודעה האחרונה, כדי שכפתור ישן לא ישלח שוב."""
    original = message["meta"].get("question", "")
    # שורה גמישה: כל כפתור ברוחב הטקסט שלו, ובמסך צר - ממשיכים בשורה הבאה (בעמודות "אחר" נחתך ל"א…")
    with st.container(horizontal=True, wrap=True, gap="small"):
        for label in list(bot.INSURANCE_TYPES):
            st.button(label, key=f"type-{message.get('id')}-{label}", on_click=choose_type, args=(original, label))
        st.button(OTHER_TYPE_LABEL, key=f"type-{message.get('id')}-other", on_click=ask_other_type, args=(original,))


def render(message: dict, stream: bool = False, last: bool = False):
    with st.chat_message(message["role"]):
        render_body(message, stream, last)


def render_body(message: dict, stream: bool = False, last: bool = False):
    """תוכן ההודעה, בתוך בועה שכבר פתוחה (של render, או הבועה שבה הוצג החיווי בשלבים)."""
    meta = message.get("meta") or {}
    body = f'<div class="answer">{to_html(message["content"])}</div>'
    if stream:
        placeholder = st.empty()
        with placeholder:
            st.write_stream(stream_words(message["content"]))
        placeholder.markdown(body, unsafe_allow_html=True)  # הגרסה הסופית: קישורים וטלפונים משמאל לימין
    else:
        st.markdown(body, unsafe_allow_html=True)
    if message.get("sources"):
        label = "המקור" if len(message["sources"]) == 1 else "המקורות"
        links = " · ".join(
            f'<a href="{html.escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">{html.escape(title)}</a>'
            for title, url in message["sources"])
        st.markdown(f'<div class="sources">📄 {label}: {links}</div>', unsafe_allow_html=True)
        if meta.get("pages") and meta.get("kind") != "error":  # מול כמה עמודים נבדקה התשובה, ותוך כמה זמן
            seconds = (meta.get("timings") or {}).get("total", 0)
            st.markdown(f'<div class="checked">נבדק מול {meta["pages"]} עמודים באתר · {seconds:.1f} שניות</div>',
                        unsafe_allow_html=True)
    if last and meta.get("kind") == "clarify":
        render_type_buttons(message)
    if last and meta.get("more_questions"):  # כמה שאלות בהודעה אחת: עניתי על הראשונה, והשאר - בלחיצה
        st.caption("שאלת גם:")
        for i, more in enumerate(meta["more_questions"]):
            st.button(more, key=f"more-{message.get('id')}-{i}", on_click=ask, args=(more,))
    if handoff := message.get("handoff"):
        if handoff["streak"]:
            st.info(STREAK_NOTICE)
        st.link_button(HANDOFF_LABEL, handoff["url"], type="primary" if handoff["streak"] else "secondary")
    if message.get("id"):  # 👍/👎 - נשמר ביומן, מקושר לתשובה
        st.feedback("thumbs", key=f"feedback-{message['id']}", on_change=save_feedback, args=(message["id"],))
    if meta and dev_mode():  # "פרטים טכניים" - רק במצב מפתחת
        score = f"{meta['top_score']:.3f}" if meta.get("top_score") is not None else "-"
        with st.expander("פרטים טכניים"):
            st.caption(f"החלטה: {meta['kind']} ({meta['reason']}) · ציון Rerank: {score} ({meta['reranker']}) · "
                       f"מודל: {llm.describe(meta['llm'])}")
            if u := meta.get("understanding"):  # מה שלב הבנת השאלה החליט
                st.caption(f"הבנת השאלה ({llm.describe(u.get('llm'))}): {u['intent']} · סוג: {u['insurance_type'] or '-'} · "
                           f"מונח: {u['key_term'] or '-'} · נושא חדש: {'כן' if u['topic_changed'] else 'לא'} · "
                           f"שפה: {u['language']}")
            elif meta.get("kind") not in ("greeting", "too_long", "error"):
                st.caption("הבנת השאלה לא הייתה זמינה - כללי הקוד (גיבוי)")
            if meta.get("warning"):  # מצב זהיר: אף מדרג לא היה זמין
                st.caption(f"⚠️ {meta['warning']}")
            if meta.get("race"):  # בקשות במקביל ומעבר לגיבוי - רק כאן וביומן, לא ללקוחה
                st.caption(f"{meta['race']} · ענה: {llm.describe(meta['llm'])}")
            if meta.get("early_search"):
                st.caption(EARLY_SEARCH[meta["early_search"]])
            if meta.get("standalone"):
                st.caption(f"שאלת המשך - הבוט חיפש: {meta['standalone']}")
            if meta.get("cached"):
                st.caption("התשובה מהמטמון של התשובות המאומתות")
            if meta.get("timings"):
                st.caption("זמנים: " + " · ".join(f"{STAGE_NAMES.get(k, k)} {v:.1f} ש'"
                                                  for k, v in meta["timings"].items()))
            # שורה אחת לקריאה במכונה - הבדיקה בדפדפן (e2e_cloud.py) קוראת ממנה החלטה, מודל וזמנים
            st.caption("META " + json.dumps({k: meta.get(k) for k in DEV_META}, ensure_ascii=False))


DEV_META = ("kind", "reason", "llm", "reranker", "top_score", "timings", "early_search", "cached", "race")
CONVERSATION_TTL = 24 * 3600
CONVERSATION_ID = re.compile(r"[0-9a-f]{12}")


def conversation_id() -> str:
    """מזהה השיחה בכתובת (?c=...): רענון הדף שומר את הכתובת, ולפיה השיחה חוזרת מ-Upstash."""
    cid = st.query_params.get("c", "")
    if not CONVERSATION_ID.fullmatch(cid):
        cid = uuid.uuid4().hex[:12]
        st.query_params["c"] = cid
    return cid


def restore_conversation(cid: str) -> list[dict]:
    raw = store.read("GET", f"conv:{cid}")
    try:
        messages = json.loads(raw) if raw else []
    except ValueError:
        return []
    for m in messages:  # JSON מחזיר רשימות במקום זוגות - המקורות חוזרים לזוגות (שם, כתובת)
        m["sources"] = [tuple(source) for source in m.get("sources") or []]
    return messages


def save_conversation(cid: str, messages: list[dict]):
    """ברקע. בהודעות של הלקוחה מספרים ארוכים (ת"ז, פוליסה, טלפון) מוסתרים - כמו ביומן."""
    saved = [m | {"content": conversation_log.hide_numbers(m["content"])} if m["role"] == "user" else m
             for m in messages]
    store.write(["SET", f"conv:{cid}", json.dumps(saved, ensure_ascii=False, default=str), "EX", CONVERSATION_TTL])


def dev_mode() -> bool:
    """מצב מפתחת: ?dev=1 בכתובת. רק בו מוצגים "פרטים טכניים" (החלטה, ציון, מודל, זמנים) - הלקוחה לא רואה אותם."""
    return st.query_params.get("dev") == "1"


def show_stage(slot, stage: str, **info):
    """החיווי בשלבים: השלב שהבוט התחיל עכשיו מחליף את הקודם, באותו מקום בבועה. נקרא מתוך answer.answer
    (on_stage), ולכן מתחלף רק כשהשלב הקודם באמת הסתיים."""
    if stage not in STAGE_LINES or (stage == "rerank" and not info.get("pages")):  # אין מועמדים - אין מה לדרג
        return
    line, term = STAGE_LINES[stage]
    slot.markdown(f'<div class="stage"><div class="stage-main">{html.escape(line.format(**info))}</div>'
                  f'<div class="stage-term">{html.escape(term)}</div></div>', unsafe_allow_html=True)


def save_feedback(entry_id: str):
    value = st.session_state.get(f"feedback-{entry_id}")  # 1 = 👍, 0 = 👎, None = בוטל
    conversation_log.log_feedback(entry_id, {1: "up", 0: "down"}.get(value))


def handoff_for(reply: dict, earlier: list[dict]) -> dict | None:
    """כפתור "המשך עם נציג בוואטסאפ" לתשובה personal, not_on_site או תקלה, עם הודעה מוכנה שמסכמת את השאלה
    (בשאלת המשך - השאלה העצמאית, שמובנת גם בלי השיחה). אחרי 2 "לא מופיע באתר" ברצף - הודעה בולטת,
    וההודעה המוכנה מסכמת את שתי השאלות. earlier: ההודעות שלפני התשובה הזו."""
    meta = reply["meta"]
    if meta["kind"] not in ("personal", "not_on_site", "unverified", "error"):
        return None
    question = (meta.get("standalone") or meta.get("question") or "")[:200]
    answers = [m for m in earlier if m["role"] == "assistant"]
    previous = answers[-1]["meta"] if answers and answers[-1].get("meta") else {}
    streak = meta["kind"] == "not_on_site" and previous.get("kind") == "not_on_site"
    if streak:
        first = (previous.get("standalone") or previous.get("question") or "")[:200]
        text = f"שלום, שאלתי את הבוט באתר הפניקס ולא מצאתי באתר תשובה לשתי שאלות: 1) {first} 2) {question}"
    elif meta["kind"] == "personal":
        text = f"שלום, יש לי שאלה על המידע האישי שלי: {question}"
    elif meta["kind"] == "unverified":
        text = f"שלום, שאלתי את הבוט באתר הפניקס והוא מצא מידע קרוב, אבל לא תשובה מדויקת לשאלה: {question}"
    elif meta["kind"] == "error":
        text = f"שלום, ניסיתי לשאול את הבוט באתר הפניקס והייתה תקלה. השאלה שלי: {question}"
    else:
        text = f"שלום, שאלתי את הבוט באתר הפניקס ולא מצאתי באתר תשובה לשאלה: {question}"
    return {"streak": streak, "url": f"{WHATSAPP_URL}?text={quote(text)}"}


def ask(question: str):
    st.session_state.pending = question


def choose_type(original: str, kind: str):
    """כפתור סוג ביטוח: השאלה המקורית + הסוג נשלחת לבוט, ומופיעה בצ'אט כאילו הלקוחה כתבה אותה."""
    st.session_state.pop("awaiting_type", None)
    st.session_state.pending = bot.with_insurance_type(original, kind)


def ask_other_type(original: str):
    """"אחר": מבקשים לכתוב את סוג הביטוח. מה שייכתב (אם זה קצר) יתחבר לשאלה המקורית."""
    st.session_state.awaiting_type = original
    st.session_state.messages.append({"role": "assistant", "content": OTHER_TYPE_TEXT,
                                      "meta": {"kind": "clarify_other", "reason": "other_type", "llm": None,
                                               "reranker": "-", "top_score": None}})


def new_chat():
    st.session_state.messages = []
    st.session_state.pop("awaiting_type", None)
    st.session_state.session_id = uuid.uuid4().hex[:12]  # שיחה חדשה - מזהה חדש בכתובת
    st.query_params["c"] = st.session_state.session_id


def main():
    st.set_page_config(page_title="הבוט של אתר הפניקס", page_icon="💬", layout="centered")
    st.markdown(RTL_CSS, unsafe_allow_html=True)
    if "session_id" not in st.session_state:  # כניסה ראשונה או רענון: המזהה מהכתובת, והשיחה - מ-Upstash
        st.session_state.session_id = conversation_id()  # מזהה אנונימי של השיחה - ליומן ולשמירה
        st.session_state.messages = restore_conversation(st.session_state.session_id)
    st.session_state.setdefault("messages", [])

    title_col, button_col = st.columns([4, 1], vertical_alignment="bottom")
    title_col.title("שאלו את הבוט של הפניקס")
    button_col.button("שיחה חדשה", on_click=new_chat, use_container_width=True)
    if TEST_PROVIDERS:
        st.caption(f"מצב בדיקות ממשק - מודלים: {', '.join(TEST_PROVIDERS)}")

    try:
        load_bot()
    except Exception:
        log.exception("loading the bot failed")
        st.error(LOAD_ERROR)
        st.stop()

    typed = st.chat_input("כתבו שאלה...")
    if typed is not None and not typed.strip():
        st.toast("כתבו שאלה ואז שלחו")
    question = (typed or "").strip() or st.session_state.pop("pending", None)
    if typed and typed.strip() and (original := st.session_state.pop("awaiting_type", None)):
        # אחרי "אחר": תשובה קצרה היא סוג הביטוח, ומתחברת לשאלה המקורית; ארוכה יותר - שאלה חדשה
        if len(typed.split()) <= OTHER_TYPE_MAX_WORDS:
            question = bot.with_insurance_type(original, typed)

    with st.chat_message("assistant"):
        st.markdown(f'<div class="answer">{to_html(WELCOME)}</div>', unsafe_allow_html=True)
        if not st.session_state.messages and not question:
            for i, example in enumerate(EXAMPLES):
                st.button(example, key=f"example-{i}", on_click=ask, args=(example,))

    for i, message in enumerate(st.session_state.messages):  # כפתורי סוג ביטוח - רק בהודעה האחרונה שעוד מחכה
        render(message, last=i == len(st.session_state.messages) - 1 and not question)

    if question:
        user_message = {"role": "user", "content": question}
        st.session_state.messages.append(user_message)
        render(user_message)
        with st.chat_message("assistant"):  # החיווי בשלבים, ואחריו התשובה - באותה בועה
            status = st.empty()
            session = st.session_state.session_id
            try:
                history = chat_history(st.session_state.messages[:-1])  # בלי השאלה הנוכחית
                # מצב מפתחת: בלי מטמון (מודדים את הבוט); ?legacy=1 - בלי שיפורי הזמנים, למדידת "לפני"
                with config.legacy_mode(dev_mode() and st.query_params.get("legacy") == "1"):
                    result = bot.answer(question, history, on_stage=partial(show_stage, status), cache=not dev_mode())
                reply = message_from_answer(result, question)
                reply["handoff"] = handoff_for(reply, st.session_state.messages)
                reply["id"] = conversation_log.log_answer(
                    session, question, result, handoff=bool(reply["handoff"]), race=result.race,
                    early_search=result.early_search, cached=result.cached,
                    legacy=dev_mode() and st.query_params.get("legacy") == "1")
            except Exception as exc:  # תקלה לא צפויה (רשת, מכסה, באג) - הודעה ידידותית, הפירוט רק ביומן
                log.exception("answering failed")
                reply = error_message(type(exc).__name__, question)
                reply["handoff"] = handoff_for(reply, st.session_state.messages)
                reply["id"] = conversation_log.log_error(session, question, type(exc).__name__)
            st.session_state.messages.append(reply)
            save_conversation(session, st.session_state.messages)
            with status.container():  # התשובה במקום החיווי: מבנה כמו בהיסטוריה, בלי רכיב כפול בשאלה הבאה
                render_body(reply, stream=True, last=True)


if __name__ == "__main__":
    main()
