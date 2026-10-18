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
