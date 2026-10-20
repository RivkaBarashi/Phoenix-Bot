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
from .fallback import EmptyResult, Provider, is_resting, run_chain

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
    provider: str         # הספק שענה בפועל, למשל "gemini:gemini-3.6-flash"
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


def cache_key(messages: list[dict], system: str, temperature: float, json_mode: bool, only: list[str] | None,
              low_thinking: bool = False) -> str:
    """חשיבה נמוכה - מפתח אחר (תשובה אחרת); בלי - אותו מפתח כמו תמיד, כך שהמטמון הקיים נשאר תקף."""
    payload = json.dumps([system, messages, temperature, json_mode, sorted(only or [])] + (["low"] if low_thinking else []),
                         ensure_ascii=False)
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


def _gemini(model, number=1):
    # Gemma דרך אותו API לא תומך בהוראות מערכת ובמצב JSON - ההוראות נכנסות להודעה הראשונה
    is_gemma = model.startswith("gemma")

    def call(messages, system, temperature, json_mode=False, timeout=None, low_thinking=False):
        if is_gemma and system:
            messages = [{"role": "user", "content": f"{system}\n\n{messages[0]['content']}"}, *messages[1:]]
        contents = [
            types.Content(role="model" if m["role"] == "assistant" else "user",
                          parts=[types.Part(text=m["content"])])
            for m in messages
        ]
        resp = config.gemini_client(number).models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=None if is_gemma else (system or None),
                temperature=temperature,
                response_mime_type="application/json" if json_mode and not is_gemma else None,
                # הרמה הנמוכה שהמודל מקבל (config.LOW_THINKING); בלי - ברירת המחדל של המודל
                thinking_config=types.ThinkingConfig(thinking_level=config.LOW_THINKING.get(model, "LOW"))
                if low_thinking and not is_gemma else None,
                http_options=types.HttpOptions(timeout=int((timeout or config.HTTP_TIMEOUT) * 1000)),
            ),
        )
        _count_tokens(f"gemini:{model}#{number}", getattr(resp.usage_metadata, "total_token_count", None))
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


def _providers() -> list[Provider]:
    """השרשרת לפי הסדר ב-config.LLM_CHAIN. כל מודל של Gemini - בכל המפתחות (קודם כל המפתחות של המודל
    הראשון, אחר כך של השני): מכסה של מפתח אחד נגמרה - ממשיכים באותו מודל במפתח הבא."""
    providers = []
    for p, m in config.LLM_CHAIN:
        if p == "gemini":
            providers += [Provider(f"gemini:{m}#{n}", _gemini(m, n), group=f"gemini:{m}", key=f"gemini#{n}")
                          for n, _ in config.GEMINI_KEYS]
        else:
            providers.append(Provider(f"{p}:{m}", _make(p, m), enabled=bool(config.API_KEYS[p])))
    return providers


PROVIDERS = _providers()
ALL_NAMES = [p.name for p in PROVIDERS]  # לפני כל סינון (ממשק בדיקות, smoke --providers) - לדוחות


def describe(name: str | None) -> str:
    """שם ספק לתצוגה: "gemini:gemini-3.6-flash#2" -> "gemini-3.6-flash (מפתח 2)". רק מספר המפתח, לא הערך."""
    if not name:
        return "-"
    base, cached = name.removesuffix(" (cache)"), name.endswith(" (cache)")
    service, _, model = base.partition(":")
    model, _, number = model.partition("#")
    text = f"{model} (מפתח {number})" if number else f"{service}: {model}" if model else service
    return text + (" (מהמטמון)" if cached else "")


def _from_cache(key: str, parse: Callable[[str], Any] | None) -> ChatResult | None:
    if not (hit := _cache_get(key)):
        return None
    try:
        data = parse(hit["text"]) if parse else None
    except ValueError:  # הבודק השתנה מאז ששמרנו - שואלים את המודל מחדש
        return None
    CACHE_HITS[hit["provider"]] += 1
    return ChatResult(hit["text"], hit["provider"], data, cached=True)


def lookup(messages: list[dict], system: str, temperature: float, json_mode: bool, only: list[str] | None,
           parse: Callable[[str], Any] | None = None, low_thinking: bool = False) -> ChatResult | None:
    """תשובה מהמטמון בלבד (בלי קריאה), עם אותו מפתח כמו chat - לפני ששולחים לשני מודלים במקביל."""
    key = cache_key(messages, system, temperature, json_mode, only, low_thinking)
    return _from_cache(key, parse) if CACHE_MODE == "on" else None


def remember(messages: list[dict], system: str, temperature: float, json_mode: bool, only: list[str] | None,
             result: ChatResult, low_thinking: bool = False):
    """שומר במטמון את התשובה שנבחרה (בבקשות במקביל - רק של המנצח)."""
    if CACHE_MODE != "off":
        _cache_put(cache_key(messages, system, temperature, json_mode, only, low_thinking),
                   {"text": result.text, "provider": result.provider, "saved": time.strftime("%Y-%m-%d %H:%M")})


def race_pair(models: list[str] | None = None, service: str = "gemini") -> list[list[str]]:
    """שני מודלים שונים שאליהם נשלחת אותה שאלה במקביל: לכל אחד - כל המפתחות שלו לפי הסדר, ובמודל השני
    מתחילים ממפתח אחר מהראשון (אם יש). רק צירופים שלא "נחים". רק מודל אחד זמין (השאר נחים בגלל עומס) - אותו
    מודל בשני מפתחות שונים במקביל, כך ש-503 בודד או תשובה איטית במפתח אחד לא מעכבים. אין אף מפתח פנוי שני - רשימה ריקה.
    models: המודלים המועדפים, לפי הסדר (config.RACE_LLMS); מודל מהם לא זמין - הבא בתור מהשרשרת."""
    groups: dict[str, list[Provider]] = {}
    for p in PROVIDERS:
        if p.name.startswith(service + ":") and p.enabled and not is_resting(p):
            groups.setdefault(p.group or p.name, []).append(p)
    if len(groups) == 1:
        keys = next(iter(groups.values()))
        return [[p.name for p in keys], [p.name for p in keys[1:] + keys[:1]]] if len(keys) > 1 else []
    if len(groups) < 2:
        return []
    preferred = [g for g in (f"{service}:{m}" for m in models or []) if g in groups]
    first, second = [groups[g] for g in preferred + [g for g in groups if g not in preferred]][:2]
    if len(second) > 1 and second[0].key == first[0].key:
        second = second[1:] + second[:1]
    return [[p.name for p in first], [p.name for p in second]]


def chat(messages: list[dict], system: str = "", temperature: float = 0.2, json_mode: bool = False,
         parse: Callable[[str], Any] | None = None, only: list[str] | None = None,
         names: list[str] | None = None, cache: bool = True, low_thinking: bool = False) -> ChatResult:
    """messages: [{"role": "user" | "assistant", "content": "..."}]

    json_mode: לבקש JSON (ב-Gemini גם ברמת ה-API; אצל השאר לפי הפרומפט).
    parse: מפענח/בודק את התשובה. אם הוא זורק ValueError (למשל JSON שבור) - התשובה נחשבת
           פגומה ועוברים לספק הבא.
    only: רק ספקים מהסוגים האלה, למשל ["groq", "openrouter"].
    names: רק הספקים האלה, בסדר הזה (בקשה אחת מתוך שתיים במקביל).
    cache: False - בלי לקרוא מהמטמון ובלי לכתוב אליו (בבקשות במקביל רק המנצח נשמר, ב-remember).
    low_thinking: ב-Gemini - רמת החשיבה הנמוכה שהמודל מקבל (config.LOW_THINKING). לשאר הספקים אין משמעות.
    """
    key = cache_key(messages, system, temperature, json_mode, only, low_thinking)
    if cache and CACHE_MODE == "on" and (hit := _from_cache(key, parse)):
        return hit

    if names is not None:
        by_name = {p.name: p for p in PROVIDERS}
        providers = [by_name[n] for n in names if n in by_name]
    else:
        providers = [p for p in PROVIDERS if not only or p.name.split(":")[0] in only]

    def checked(p: Provider) -> Provider:
        def call(messages, system, temperature):
            extra = {"low_thinking": True} if low_thinking and p.name.startswith("gemini:") else {}
            text = p.call(messages, system, temperature, json_mode=json_mode, timeout=config.QUERY_TIMEOUT, **extra)
            if parse is None:
                return text, None
            try:
                return text, parse(text)
            except ValueError as exc:
                raise EmptyResult(f"unusable response: {exc}") from exc
        return Provider(p.name, call, p.enabled, p.group, p.key)  # עם הקבוצות: עומס נח במודל, מפתח לא תקין - במפתח

    # הלקוחה מחכה לתשובה: מודל שלא עונה תוך QUERY_TIMEOUT או חסום - עוברים מיד לבא (config).
    (text, data), provider = run_chain("llm", [checked(p) for p in providers], messages, system, temperature,
                                       retries=config.QUERY_RETRIES, max_wait=config.QUERY_MAX_WAIT,
                                       deadline=config.QUERY_TIMEOUT)
    if cache and CACHE_MODE != "off":
        _cache_put(key, {"text": text, "provider": provider, "saved": time.strftime("%Y-%m-%d %H:%M")})
    return ChatResult(text, provider, data)
