"""מודל שפה עם גיבוי אוטומטי.

    from phoenix.llm import chat
    result = chat([{"role": "user", "content": "שאלה"}], system="הוראות")
    result.text, result.provider

אם הספק הראשון ב-config.LLM_CHAIN נחסם או נופל, עוברים לבא בתור.

מטמון (חיסכון במכסה): תשובה של מודל נשמרת ב-data/cache/llm.json לפי ההוראות (system), ההודעות (השאלה
והקטעים) וההגדרות. אותה בקשה בדיוק - התשובה מהמטמון, בלי לשלוח שוב. כל שינוי בהוראות או בקטעים הוא
מפתח אחר, כלומר "גרסה חדשה", ונשלח למודל מחדש. CACHE_MODE: on (ברירת מחדל) / write (תמיד לשאול
ולשמור - לבדיקה "טרייה") / off.
"""
import collections
import hashlib
import json
import logging
import os
import threading
import time
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
log = logging.getLogger("phoenix")

CACHE_FILE = config.ROOT / "data" / "cache" / "llm.json"
CACHE_MODE = "on"
CACHE_HITS: collections.Counter = collections.Counter()  # תשובות שהגיעו מהמטמון, לפי המודל שענה במקור
TOKENS: collections.Counter = collections.Counter()      # טוקנים שהספקים דיווחו (קלט + פלט), לפי מודל
_cache: dict[str, dict] | None = None
_cache_lock = threading.Lock()


@dataclass
class ChatResult:
    text: str
    provider: str         # הספק שענה בפועל, למשל "gemini:gemini-3.5-flash"
    data: Any = None      # התוצאה של parse (אם ביקשו), למשל JSON מפוענח
    cached: bool = False  # התשובה הגיעה מהמטמון, בלי קריאה למודל


def _read_cache_file() -> dict[str, dict]:
    try:
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _cache_get(key: str) -> dict | None:
    global _cache
    with _cache_lock:
        if _cache is None:
            _cache = _read_cache_file()
        return _cache.get(key)


def _cache_put(key: str, entry: dict):
    """שמירה בטוחה גם כשהצ'אט והבדיקות רצים יחד: ממזגים עם מה שבקובץ ומחליפים אותו בפעולה אחת."""
    global _cache
    with _cache_lock:
        merged = _read_cache_file() | (_cache or {})
        merged[key] = entry
        _cache = merged
        try:
            CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp = CACHE_FILE.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, CACHE_FILE)
        except OSError as exc:  # המטמון הוא רק חיסכון - תקלה בו לא עוצרת תשובה
            log.warning("llm cache not saved: %s", exc)


def cache_key(messages: list[dict], system: str, temperature: float, json_mode: bool, only: list[str] | None) -> str:
    payload = json.dumps([system, messages, temperature, json_mode, sorted(only or [])], ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _count_tokens(name: str, tokens) -> None:
    if isinstance(tokens, int) and tokens > 0:
        TOKENS[name] += tokens


def _non_empty(text: str | None) -> str:
    if not text or not text.strip():
        raise EmptyResult("empty response")
    return text.strip()


def _with_system(messages: list[dict], system: str) -> list[dict]:
    return ([{"role": "system", "content": system}] if system else []) + messages


def _gemini(model):
    # Gemma דרך אותו API לא תומך בהוראות מערכת ובמצב JSON - ההוראות נכנסות להודעה הראשונה
    is_gemma = model.startswith("gemma")

    def call(messages, system, temperature, json_mode=False, timeout=None):
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
                http_options=types.HttpOptions(timeout=int((timeout or config.HTTP_TIMEOUT) * 1000)),
            ),
        )
        _count_tokens(f"gemini:{model}", getattr(resp.usage_metadata, "total_token_count", None))
        return _non_empty(resp.text)
    return call


def _openai_compatible(provider, model):
    def call(messages, system, temperature, json_mode=False, timeout=None):  # JSON: לפי ההוראות בפרומפט
        r = httpx.post(
            OPENAI_COMPATIBLE_URLS[provider],
            headers={"Authorization": f"Bearer {config.API_KEYS[provider]}"},
            json={"model": model, "temperature": temperature, "messages": _with_system(messages, system)},
            timeout=timeout or config.HTTP_TIMEOUT,
        )
        r.raise_for_status()
        body = r.json()
        _count_tokens(f"{provider}:{model}", (body.get("usage") or {}).get("total_tokens"))
        return _non_empty(body["choices"][0]["message"]["content"])
    return call


def _cohere(model):
    def call(messages, system, temperature, json_mode=False, timeout=None):  # JSON: לפי ההוראות בפרומפט
        r = httpx.post(
            "https://api.cohere.com/v2/chat",
            headers={"Authorization": f"Bearer {config.API_KEYS['cohere']}"},
            json={"model": model, "temperature": temperature, "messages": _with_system(messages, system)},
            timeout=timeout or config.HTTP_TIMEOUT,
        )
        r.raise_for_status()
        body = r.json()
        used = (body.get("usage") or {}).get("tokens") or {}
        _count_tokens(f"cohere:{model}", int(used.get("input_tokens") or 0) + int(used.get("output_tokens") or 0))
        return _non_empty("".join(part.get("text", "") for part in body["message"]["content"]))
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
    key = cache_key(messages, system, temperature, json_mode, only)
    if CACHE_MODE == "on" and (hit := _cache_get(key)):
        try:
            data = parse(hit["text"]) if parse else None
        except ValueError:  # הבודק השתנה מאז ששמרנו - שואלים את המודל מחדש
            pass
        else:
            CACHE_HITS[hit["provider"]] += 1
            return ChatResult(hit["text"], hit["provider"], data, cached=True)

    providers = [p for p in PROVIDERS if not only or p.name.split(":")[0] in only]

    def checked(p: Provider) -> Provider:
        def call(messages, system, temperature):
            text = p.call(messages, system, temperature, json_mode=json_mode, timeout=config.QUERY_TIMEOUT)
            if parse is None:
                return text, None
            try:
                return text, parse(text)
            except ValueError as exc:
                raise EmptyResult(f"unusable response: {exc}") from exc
        return Provider(p.name, call, p.enabled)

    # הלקוחה מחכה לתשובה: מודל שלא עונה תוך QUERY_TIMEOUT או חסום - עוברים מיד לבא (config).
    (text, data), provider = run_chain("llm", [checked(p) for p in providers], messages, system, temperature,
                                       retries=config.QUERY_RETRIES, max_wait=config.QUERY_MAX_WAIT,
                                       deadline=config.QUERY_TIMEOUT)
    if CACHE_MODE != "off":
        _cache_put(key, {"text": text, "provider": provider, "saved": time.strftime("%Y-%m-%d %H:%M")})
    return ChatResult(text, provider, data)
