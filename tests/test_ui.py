"""בדיקות ממשק (Streamlit AppTest): האפליקציה רצה בלי דפדפן, והבוט מזויף - בלי API.

הרצה:  .venv\\Scripts\\python -m pytest tests/test_ui.py
"""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from phoenix import answer as bot
from phoenix import conversation_log

APP = str(Path(__file__).resolve().parent.parent / "app.py")
CONTACT = f"\n\nהמקור: {bot.CONTACT_URL}"


@pytest.fixture
def app(monkeypatch):
    """האפליקציה עם בוט מזויף: fake(question) -> Answer, או חריגה. היומן לא נכתב לקובץ."""
    monkeypatch.setattr(conversation_log, "_append", lambda record: None)

    def start(fake, dev: bool = False) -> AppTest:
        monkeypatch.setattr(bot, "answer", lambda question, history=(), on_stage=None: fake(question))
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
