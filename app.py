"""ממשק צ'אט לבוט של אתר הפניקס (Streamlit).

הרצה:  .venv\\Scripts\\streamlit run app.py   (או לחיצה כפולה על run_chat.bat)

הממשק רק מציג את השיחה וקורא ל-phoenix.answer.answer. כל ההחלטות - חיפוש, סף, ההוראות
למודל והבדיקות בקוד - נשארות בקוד של הבוט ולא משתנות מכאן.

בדיקות ממשק בלי לבזבז את המכסה של Gemini: CHAT_LLM_PROVIDERS=groq,openrouter ב-.env.
ריק או לא מוגדר = שרשרת המודלים הרגילה (Gemini קודם). ההגדרה משפיעה רק על הממשק -
הבדיקות (eval_*.py) לא קוראות אותה.

שאלות המשך: בינתיים כל שאלה נשלחת לבוט לבד, בלי השיחה הקודמת. ההיסטוריה מוצגת על
המסך, והבנת ההקשר משאלה קודמת תיבנה בהמשך.
"""
import html
import logging
import os
import re

import streamlit as st

from phoenix import answer as bot
from phoenix import bm25, config, embeddings, index, llm

log = logging.getLogger("phoenix.app")

TEST_PROVIDERS = [p.strip() for p in os.getenv("CHAT_LLM_PROVIDERS", "").split(",") if p.strip()]
EXAMPLES = [
    "יש לי נזילה בבית, איך מגישים תביעה?",
    "תוך כמה זמן מטפלים בתביעת מחלות קשות?",
    "מה המספר של מוקד תביעות רכב ורכוש?",
]
WELCOME = ("שלום! אני הבוט של אתר הפניקס. אני עונה רק לפי המידע שבאתר fnx.co.il, ומצרף לכל תשובה "
           "קישור לעמוד המקור. כשהמידע לא מופיע באתר - אגיד זאת ואפנה לשירות הלקוחות. "
           "אין לי גישה לפוליסה האישית שלכם. אפשר לשאול למשל:")
CONTACT_TITLE = "כל הדרכים ליצור איתנו קשר"
ERROR_MESSAGE = ("משהו השתבש ולא הצלחתי לענות כרגע. אפשר לנסות שוב בעוד רגע, "
                 "או לפנות לשירות הלקוחות של הפניקס בוואטסאפ 054-9893455.")
LOAD_ERROR = ("הבוט לא הצליח להיטען. צריך לוודא שהאינדקס נבנה (python -m phoenix.chunk ואז "
              "python -m phoenix.index) ולהפעיל את הממשק מחדש.")

RTL_CSS = """
<style>
.stApp, [data-testid="stChatMessage"], [data-testid="stChatInput"] { direction: rtl; text-align: right; }
[data-testid="stChatInput"] textarea { direction: rtl; text-align: right; }
.answer { line-height: 1.7; }
.sources { font-size: 0.9rem; border-top: 1px solid rgba(128, 128, 128, 0.3); margin-top: 0.5rem; padding-top: 0.4rem; }
</style>
"""

# טלפונים, קישורים ומיילים מוצגים משמאל לימין בתוך הטקסט העברי, כדי שלא יתהפכו
LTR_TOKEN = re.compile(
    r"(?P<url>https?://[^\s<>\"']*[^\s<>\"'.,;:!?)\]])"
    r"|(?P<email>[\w.+-]+@[\w-]+(?:\.[\w-]+)+)"
    r"|(?P<star>\*\d{3,5}\b|\b\d{3,5}\*)"
    r"|(?P<phone>\b0\d{1,2}-?\d{3}-?\d{4}\b)"
)


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
    config.gemini_client()
    return {"chunks": len(chunks), "vectors": vectors, "llms": [p.name for p in llm.PROVIDERS if p.enabled]}


def to_html(text: str) -> str:
    """טקסט -> HTML בטוח: הכול עובר escape, קישורים נפתחים בלשונית חדשה, וטלפונים/קישורים/מיילים
    עטופים ב-bdi כדי שיוצגו נכון בתוך עברית. קודי כוכבית ("5432*") מוצגים כמו שמחייגים: ‎*5432."""
    parts, last = [], 0
    for m in LTR_TOKEN.finditer(text):
        parts.append(html.escape(text[last:m.start()]))
        value = m.group(0)
        if m.lastgroup == "url":
            href = html.escape(value, quote=True)
            parts.append(f'<a href="{href}" target="_blank" rel="noopener noreferrer"><bdi dir="ltr">{html.escape(value)}</bdi></a>')
        else:
            if m.lastgroup == "star" and value.endswith("*"):
                value = "*" + value[:-1]
            parts.append(f'<bdi dir="ltr">{html.escape(value)}</bdi>')
        last = m.end()
    parts.append(html.escape(text[last:]))
    return "".join(parts).replace("\n", "<br>")


def message_from_answer(a: bot.Answer) -> dict:
    """תוצאת הבוט -> הודעה לתצוגה: הטקסט בלי שורת המקור, והמקורות כקישורים עם שם העמוד."""
    text = a.text.rsplit("\n\nהמקור", 1)[0]  # format_sources מוסיף "המקור:" / "המקורות:" בסוף
    titles = {h.url: h.title for h in a.hits} | {bot.CONTACT_URL: CONTACT_TITLE}
    meta = {"kind": a.kind, "reason": a.reason, "llm": a.llm, "reranker": a.reranker, "top_score": a.top_score}
    return {"role": "assistant", "content": text,
            "sources": [(titles.get(url, url), url) for url in a.sources], "meta": meta}


def error_message(reason: str) -> dict:
    return {"role": "assistant", "content": ERROR_MESSAGE, "sources": [(CONTACT_TITLE, bot.CONTACT_URL)],
            "meta": {"kind": "error", "reason": reason, "llm": None, "reranker": "-", "top_score": None}}


def render(message: dict):
    with st.chat_message(message["role"]):
        st.markdown(f'<div class="answer">{to_html(message["content"])}</div>', unsafe_allow_html=True)
        if message.get("sources"):
            label = "המקור" if len(message["sources"]) == 1 else "המקורות"
            links = " · ".join(
                f'<a href="{html.escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">{html.escape(title)}</a>'
                for title, url in message["sources"])
            st.markdown(f'<div class="sources">📄 {label}: {links}</div>', unsafe_allow_html=True)
        if message.get("meta"):
            meta = message["meta"]
            score = f"{meta['top_score']:.3f}" if meta.get("top_score") is not None else "-"
            with st.expander("פרטים טכניים"):
                st.caption(f"החלטה: {meta['kind']} ({meta['reason']}) · ציון Rerank: {score} ({meta['reranker']}) · "
                           f"מודל: {meta['llm'] or '-'}")


def ask(question: str):
    st.session_state.pending = question


def new_chat():
    st.session_state.messages = []


def main():
    st.set_page_config(page_title="הבוט של אתר הפניקס", page_icon="💬", layout="centered")
    st.markdown(RTL_CSS, unsafe_allow_html=True)
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

    with st.chat_message("assistant"):
        st.markdown(f'<div class="answer">{to_html(WELCOME)}</div>', unsafe_allow_html=True)
        if not st.session_state.messages and not question:
            for i, example in enumerate(EXAMPLES):
                st.button(example, key=f"example-{i}", on_click=ask, args=(example,))

    for message in st.session_state.messages:
        render(message)

    if question:
        user_message = {"role": "user", "content": question}
        st.session_state.messages.append(user_message)
        render(user_message)
        with st.spinner("מחפש באתר..."):
            try:
                reply = message_from_answer(bot.answer(question))
            except Exception as exc:  # תקלה לא צפויה (רשת, מכסה, באג) - הודעה ידידותית, הפירוט רק ביומן
                log.exception("answering failed")
                reply = error_message(type(exc).__name__)
        st.session_state.messages.append(reply)
        render(reply)


if __name__ == "__main__":
    main()
