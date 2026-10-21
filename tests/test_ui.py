"""בדיקות ממשק (Streamlit AppTest): האפליקציה רצה בלי דפדפן, והבוט מזויף - בלי API.

הרצה:  .venv\\Scripts\\python -m pytest tests/test_ui.py
"""
from pathlib import Path

import json
import pytest
from streamlit.testing.v1 import AppTest

from phoenix import answer as bot
from phoenix import conversation_log

APP = str(Path(__file__).resolve().parent.parent / "app.py")
CONTACT = f"\n\nהמקור: {bot.CONTACT_URL}"


@pytest.fixture
def app(monkeypatch):
    """האפליקציה עם בוט מזויף: fake(question) -> Answer, או חריגה. היומן לא נכתב לקובץ. הבנת השאלה לא זמינה
    (בלי API) - הבוט האמיתי (real_answer) עובד לפי כללי הקוד."""
    monkeypatch.setattr(conversation_log, "_append", lambda record: None)
    monkeypatch.setattr(bot, "try_understand", lambda *args, **kwargs: None)

    def start(fake, dev: bool = False) -> AppTest:
        monkeypatch.setattr(bot, "answer", lambda question, history=(), on_stage=None, cache=True: fake(question))
        at = AppTest.from_file(APP, default_timeout=120)
        if dev:
            at.query_params["dev"] = "1"
        return at.run()
    return start


def regular(question):
    return bot.Answer(f"תשובה.{CONTACT}", "answer", "verified", [bot.CONTACT_URL], 0.9, "cohere:x",
                      llm="fake:model", question=question, timings={"total": 1.0}, pages=3)


def ask(at: AppTest, question: str) -> AppTest:
    return at.chat_input[0].set_value(question).run()


@pytest.mark.parametrize("dev", [False, True])
def test_technical_details_only_in_developer_mode(app, dev):
    at = ask(app(regular, dev=dev), "מה המספר של מוקד תביעות רכב?")
    assert not at.exception
    assert [e.label for e in at.expander] == (["פרטים טכניים"] if dev else [])


def test_whatsapp_button_after_a_malfunction(app):
    def broken(question):
        raise RuntimeError("Boom")
    at = ask(app(broken), "מה המספר של מוקד תביעות רכב?")
    links = [b.proto for b in at.get("link_button")]
    assert not at.exception and at.session_state["messages"][-1]["meta"]["kind"] == "error"
    assert len(links) == 1 and links[0].url.startswith("https://wa.me/972549893455?text=")


def test_too_long_message_is_answered_with_a_request_to_shorten(app):
    long = "איך מגישים תביעה? " * 70  # ~1,250 תווים
    at = ask(app(real_answer), long)
    reply = at.session_state["messages"][-1]
    assert not at.exception and reply["meta"]["kind"] == "too_long" and "ארוכה מדי" in reply["content"]
    assert at.session_state["messages"][-2]["content"] == long.strip()  # השאלה נשמרת כמו שהיא, בלי חיתוך


def test_clarification_has_a_pension_and_savings_button(app):
    at = ask(app(real_answer), "איך מגישים תביעה?")
    labels = [b.label for b in at.button]
    assert not at.exception and "פנסיה וחיסכון" in labels and labels.count("אחר") == 1


real_answer = bot.answer  # לפני שהבדיקות מחליפות אותו: הודעה ארוכה ושאלת הבהרה לא קוראות ל-API


def test_conversation_comes_back_after_refresh_and_numbers_are_hidden(app, monkeypatch):
    """רענון: המזהה בכתובת (?c=...) מחזיר את השיחה מ-Upstash. בשמירה - מספר ארוך שהלקוחה כתבה מוסתר."""
    from phoenix import store
    earlier = [{"role": "user", "content": "שאלה קודמת"},
               {"role": "assistant", "content": "תשובה קודמת", "sources": [["צור קשר", bot.CONTACT_URL]], "meta": {}}]
    written = []
    monkeypatch.setattr(store, "read", lambda *cmd: json.dumps(earlier, ensure_ascii=False)
                        if cmd == ("GET", "conv:abcdef123456") else None)
    monkeypatch.setattr(store, "write", lambda *commands, **kw: written.extend(commands))
    monkeypatch.setattr(bot, "answer", lambda question, history=(), on_stage=None, cache=True: regular(question))
    at = AppTest.from_file(APP, default_timeout=120)
    at.query_params["c"] = "abcdef123456"
    at.run()
    assert [m["content"] for m in at.session_state["messages"]] == ["שאלה קודמת", "תשובה קודמת"]
    assert at.session_state["messages"][1]["sources"] == [("צור קשר", bot.CONTACT_URL)]
    at.chat_input[0].set_value("ת\"ז 123456789, איך מגישים תביעה בביטוח רכב?").run()
    saved = json.loads(next(c for c in written if c[0] == "SET" and c[1] == "conv:abcdef123456")[2])
    assert len(saved) == 4 and "123456789" not in saved[2]["content"] and "[מספר הוסתר]" in saved[2]["content"]


def test_clarification_shows_a_button_for_every_insurance_type(app):
    at = ask(app(real_answer), "איך מגישים תביעה?")
    labels = [b.label for b in at.button]
    assert not at.exception
    for kind in bot.INSURANCE_TYPES:
        assert labels.count(kind) == 1, kind
    assert {"חיות מחמד", "משכנתא", "אובדן כושר עבודה"} <= set(labels)


# ---------------------------------------------------------------- העיצוב (design-references/FINAL-DESIGN-INSTRUCTIONS.md)
def page_html(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


def test_design_header_greeting_and_input(app):
    """הפס העליון, כרטיס הברכה עם "הפניקס" בכתום, ותיבת הכתיבה - כמו בהוראות העיצוב."""
    at = app(regular)
    html_text = page_html(at)
    assert not at.exception
    assert "הבוט האישי שלך" in html_text and 'class="px-close"' in html_text
    assert ('שלום! 👋<br>אני הבוט של <span class="px-orange">הפניקס</span>,<br>ואני עונה על כל שאלה<br>לפי המידע שבאתר.'
            in html_text)
    assert "app/static/robot.png" in html_text
    assert at.chat_input[0].placeholder == "כתבו לי שאלה..."


def test_design_messages_are_bubbles_with_time_and_the_source_inside(app):
    at = ask(app(regular), "מה המספר של מוקד תביעות רכב?")
    html_text = page_html(at)
    assert 'class="px-row px-user"' in html_text and 'class="px-row px-bot"' in html_text
    assert html_text.count('class="px-meta"') == 1 and html_text.count('class="px-time"') == 1  # שעה ו-✓✓ לשאלה, שעה לתשובה
    assert "המקור: <a href=" in html_text and bot.CONTACT_URL in html_text
    messages = at.session_state["messages"]
    assert all(len(m["time"]) == 5 and m["time"][2] == ":" for m in messages)


def test_design_keeps_every_feature_and_hides_only_technical_details(app):
    """החלטת הבעלים: כל הפיצ'רים נשארים (במראה של העיצוב), ורק פרטים טכניים מוסתרים - ללקוחה."""
    at = app(regular)
    labels = [b.label for b in at.button]
    assert len(labels) == 3 and all("?" in label for label in labels)  # כפתורי הדוגמה, לפני השאלה הראשונה
    assert "אני עונה רק לפי המידע שבאתר" in page_html(at)             # טקסט הפתיחה, עם הדוגמאות
    at = ask(at, "מה המספר של מוקד תביעות רכב?")
    html_text = page_html(at)
    assert [b.label for b in at.button] == []                          # אחרי השאלה הראשונה - בלי דוגמאות
    assert len(at.get("feedback")) == 1                                 # 👍/👎 נשאר
    assert "נבדק מול 3 עמודים באתר · 1.0 שניות" in html_text            # והשורה "נבדק מול N עמודים"
    assert "המקור: <a href=" in html_text and not at.expander           # שורת המקור בבועה; בלי "פרטים טכניים" ללקוחה
    assert "אני עונה רק לפי המידע שבאתר" not in html_text              # טקסט הפתיחה רק בהתחלה: אחריו - כמו בתמונה
    for forbidden in ("פרטים טכניים", "Powered by", "Privacy", "By chatting"):
        assert forbidden not in html_text
    assert "rerank" not in html_text.lower() and "cohere" not in html_text.lower()


def test_design_colors_are_only_the_palette():
    from phoenix import chat_design
    import re
    colors = set(re.findall(r"#[0-9A-Fa-f]{6}\b", chat_design.css()))
    assert {c.upper() for c in colors} <= {"#111C3F", "#FF5A23", "#F7F8FB", "#8A8FA0", "#FFFFFF"}


@pytest.mark.parametrize("dev", [False, True])
def test_developer_mode_shows_what_this_server_loaded(app, dev):
    """?dev=1: כמה קטעים באינדקס ואילו מפתחות Gemini נטענו (המספרים בלבד) - לבדיקת פריסה. ללקוחה - לא."""
    at = app(regular, dev=dev)
    lines = [c.value for c in at.caption if c.value.startswith("נטען:")]
    assert not at.exception
    if not dev:
        assert lines == []
    else:
        assert len(lines) == 1 and "קטעים" in lines[0] and "מפתחות Gemini:" in lines[0] and "AIza" not in lines[0]
        from phoenix import answer_cache
        assert f"קוד {answer_cache.code_version()}" in lines[0] and f"אינדקס {answer_cache.index_version()}" in lines[0]


@pytest.mark.parametrize("dev", [False, True])
def test_header_new_chat_link_and_a_close_button_that_only_closes(app, dev):
    """בקשת הבעלים (21/10): ✕ סוגר את החלון (כפתור, בלי ניווט - השיחה נשארת), "שיחה חדשה" לידו - קישור למזהה שיחה חדש
    (מצב מפתחת נשמר). הסקריפט של הסגירה, האייקון וצליל השליחה - בעמוד פעם אחת."""
    import re
    at = app(regular, dev=dev)
    html_text = page_html(at)
    assert not at.exception
    assert re.search(r'<button class="px-close" type="button"', html_text) and 'href' not in html_text.split('class="px-close"')[1][:40]
    link = re.search(r'<a class="px-new" href="([^"]+)" target="_self">שיחה חדשה</a>', html_text)
    import html as html_lib
    href = html_lib.unescape(link.group(1)) if link else ""  # "&amp;" ב-HTML - "&" בדפדפן
    assert re.search(r"[?&]c=[0-9a-f]{12}(?![0-9a-f])", href) and (("dev=1" in href) == dev)
    assert href.split("c=")[1][:12] != at.session_state["session_id"]  # מזהה אחר - שיחה חדשה
    scripts = [h.proto.body for h in at.html]
    assert len(scripts) == 1 and "__pxUI" in scripts[0] and "app/static/launcher.png" in scripts[0]


def test_the_launcher_icon_is_served_as_the_owner_sent_it():
    from PIL import Image
    root = Path(APP).parent
    icon, original = Image.open(root / "static" / "launcher.png"), Image.open(root / "design-references" / "launcher-icon.png")
    assert icon.size == (192, 192) and icon.mode == "RGBA" and original.size == (1024, 1024)
    assert icon.getpixel((2, 2))[3] == 0 and icon.getpixel((96, 96))[3] == 255  # רקע שקוף מחוץ לעיגול, כמו המקור



def test_the_script_has_no_tag_like_text_that_the_sanitizer_drops():
    """Streamlit מוחק סקריפט שיש בו "<" ואחריו אות או "/" (נבדק 21/10): הסקריפט לא היה רץ בכלל."""
    import re
    from phoenix import chat_design
    body = chat_design.script_html()
    assert body.startswith("<script>") and body.endswith("</script>")
    assert not re.search(r"<[/A-Za-z!]", body[len("<script>"):-len("</script>")])


def test_bubble_time_is_israel_time_even_on_a_utc_server(monkeypatch):
    """בענן השרתים ב-UTC: השעה בבועות - שעון ישראל."""
    import sys
    import time as time_module
    from datetime import datetime
    from zoneinfo import ZoneInfo
    sys.path.insert(0, str(Path(APP).parent))
    import app as app_module
    monkeypatch.setenv("TZ", "UTC")
    if hasattr(time_module, "tzset"):
        time_module.tzset()
    expected = {datetime.now(ZoneInfo("Asia/Jerusalem")).strftime("%H:%M")}
    assert app_module.clock() in expected | {datetime.now(ZoneInfo("Asia/Jerusalem")).strftime("%H:%M")}

