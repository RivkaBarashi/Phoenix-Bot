"""בדיקות היחידה רצות בלי API: כל ניסיון לקרוא למודל או לשלוח בקשת HTTP נכשל מיד."""
import httpx
import pytest

from phoenix import llm


@pytest.fixture(autouse=True)
def no_api(monkeypatch):
    def blocked(*args, **kwargs):
        raise RuntimeError("קריאת API בבדיקת יחידה")
    monkeypatch.setattr(llm, "chat", blocked)
    monkeypatch.setattr(httpx.Client, "send", blocked)


@pytest.fixture(autouse=True)
def no_cloud_store(monkeypatch):
    """בלי Upstash בבדיקות - גם כשב-.env יש כתובת: שום דבר לא נשמר בענן, והבוט עובד כמו בלי אחסון."""
    from phoenix import store
    monkeypatch.setattr(store, "URL", "")


@pytest.fixture(autouse=True)
def no_backup_llm(monkeypatch):
    """בלי מודל הגיבוי מחוץ ל-Google (Cohere) - הבדיקות לא תלויות במפתח שב-.env. בדיקות הגיבוי מדליקות אותו בעצמן."""
    from phoenix import config
    monkeypatch.setattr(config, "BACKUP_LLM", "")
    monkeypatch.setattr(config, "UNDERSTAND_BACKUP", "")
