"""מודל שפה עם גיבוי אוטומטי.

    from phoenix.llm import chat
    result = chat([{"role": "user", "content": "שאלה"}], system="הוראות")
    result.text, result.provider

אם הספק הראשון ב-config.LLM_CHAIN נחסם או נופל, עוברים לבא בתור.
"""
from dataclasses import dataclass
from typing import Any, Callable

import httpx
from google.genai import types

from . import config
from .fallback import EmptyResult, Provider, run_chain

OPENAI_COMPATIBLE_URLS = {
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
}


@dataclass
class ChatResult:
    text: str
    provider: str     # הספק שענה בפועל, למשל "gemini:gemini-3.5-flash"
    data: Any = None  # התוצאה של parse (אם ביקשו), למשל JSON מפוענח


def _non_empty(text: str | None) -> str:
    if not text or not text.strip():
        raise EmptyResult("empty response")
    return text.strip()


def _with_system(messages: list[dict], system: str) -> list[dict]:
    return ([{"role": "system", "content": system}] if system else []) + messages


def _gemini(model):
    # Gemma דרך אותו API לא תומך בהוראות מערכת ובמצב JSON - ההוראות נכנסות להודעה הראשונה
    is_gemma = model.startswith("gemma")

    def call(messages, system, temperature, json_mode=False):
        if is_gemma and system:
            messages = [{"role": "user", "content": f"{system}\n\n{messages[0]['content']}"}, *messages[1:]]
        contents = [
            types.Content(role="model" if m["role"] == "assistant" else "user",
                          parts=[types.Part(text=m["content"])])
            for m in messages
        ]
        resp = config.gemini_client().models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=None if is_gemma else (system or None),
                temperature=temperature,
                response_mime_type="application/json" if json_mode and not is_gemma else None,
            ),
        )
        return _non_empty(resp.text)
    return call


def _openai_compatible(provider, model):
    def call(messages, system, temperature, json_mode=False):  # JSON: לפי ההוראות בפרומפט
        r = httpx.post(
            OPENAI_COMPATIBLE_URLS[provider],
            headers={"Authorization": f"Bearer {config.API_KEYS[provider]}"},
            json={"model": model, "temperature": temperature, "messages": _with_system(messages, system)},
            timeout=config.HTTP_TIMEOUT,
        )
        r.raise_for_status()
        return _non_empty(r.json()["choices"][0]["message"]["content"])
    return call


def _cohere(model):
    def call(messages, system, temperature, json_mode=False):  # JSON: לפי ההוראות בפרומפט
        r = httpx.post(
            "https://api.cohere.com/v2/chat",
            headers={"Authorization": f"Bearer {config.API_KEYS['cohere']}"},
            json={"model": model, "temperature": temperature, "messages": _with_system(messages, system)},
            timeout=config.HTTP_TIMEOUT,
        )
        r.raise_for_status()
        return _non_empty("".join(part.get("text", "") for part in r.json()["message"]["content"]))
    return call


def _make(provider, model):
    if provider == "gemini":
        return _gemini(model)
    if provider == "cohere":
        return _cohere(model)
    return _openai_compatible(provider, model)


PROVIDERS = [
    Provider(f"{p}:{m}", _make(p, m), enabled=bool(config.API_KEYS[p]))
    for p, m in config.LLM_CHAIN
]


def chat(messages: list[dict], system: str = "", temperature: float = 0.2, json_mode: bool = False,
         parse: Callable[[str], Any] | None = None, only: list[str] | None = None) -> ChatResult:
    """messages: [{"role": "user" | "assistant", "content": "..."}]

    json_mode: לבקש JSON (ב-Gemini גם ברמת ה-API; אצל השאר לפי הפרומפט).
    parse: מפענח/בודק את התשובה. אם הוא זורק ValueError (למשל JSON שבור) - התשובה נחשבת
           פגומה ועוברים לספק הבא.
    only: רק ספקים מהסוגים האלה, למשל ["groq", "openrouter"].
    """
    providers = [p for p in PROVIDERS if not only or p.name.split(":")[0] in only]

    def checked(p: Provider) -> Provider:
        def call(messages, system, temperature):
            text = p.call(messages, system, temperature, json_mode=json_mode)
            if parse is None:
                return text, None
            try:
                return text, parse(text)
            except ValueError as exc:
                raise EmptyResult(f"unusable response: {exc}") from exc
        return Provider(p.name, call, p.enabled)

    # הלקוחה מחכה לתשובה - לא מחכים הרבה על ספק עמוס, עוברים מהר לבא.
    (text, data), provider = run_chain("llm", [checked(p) for p in providers], messages, system, temperature,
                                       retries=1, max_wait=3)
    return ChatResult(text, provider, data)
