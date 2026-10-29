"""מודל שפה עם גיבוי אוטומטי.

    from phoenix.llm import chat
    result = chat([{"role": "user", "content": "שאלה"}], system="הוראות")
    result.text, result.provider

כל קריאה עוברת בשער המודלים (gateway.py): לכל שלב נתיב משלו (config.ROUTES) - מודל ראשי ואחריו גיבויים,
עם מפסק זרם לכל מודל. מפתח Gemini ראשי (config.LLM_GEMINI_KEY); הבא בתור - רק כשהמכסה היומית נגמרה (gemini_keys.py).

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

from . import config, fallback, gateway, gemini_keys
from .fallback import EmptyResult, Provider

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
    route: str = ""       # כשנשאל יותר ממודל אחד: מי נשאל ("gemini-3.6-flash (מפתח 1) מול cohere: ...")


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


def _gemini(model):
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

        def ask(number):
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
        return gemini_keys.call(model, ask)  # המפתח הראשי; נגמרה לו המכסה היומית - הבא בתור
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
    def call(messages, system, temperature, json_mode=False, timeout=None, low_thinking=False):
        from . import store  # כאן ולא למעלה: store לא נדרש לשאר הספקים
        store.count_call("cohere_chat")  # מונה חודשי נפרד מה-Rerank (calls:cohere_chat:<חודש> ב-Upstash)
        r = httpx.post(
            "https://api.cohere.com/v2/chat",
            headers={"Authorization": f"Bearer {config.COHERE_CHAT_KEY}"},
            json={"model": model, "temperature": temperature, "messages": _with_system(messages, system)}
            | ({"response_format": {"type": "json_object"}} if json_mode else {})
            | ({"thinking": {"type": "enabled", "token_budget": config.COHERE_LOW_THINKING_BUDGET if low_thinking
                                                                  else config.COHERE_THINKING_BUDGET}}
               if model.startswith(config.COHERE_THINKING_MODELS) else {}),
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
    """הספקים לפי הסדר ב-config.LLM_CHAIN: מודל Gemini אחד לכל גרסה, על המפתח הראשי (config.LLM_GEMINI_KEY) - ועל הבא
    בתור רק כשהמכסה היומית נגמרה (gemini_keys.py). השער (gateway.py) בוחר ביניהם לפי הנתיב של כל שלב."""
    providers = []
    for p, m in config.LLM_CHAIN:
        if p == "gemini":
            n = config.LLM_GEMINI_KEY
            providers.append(Provider(f"gemini:{m}#{n or 1}", _gemini(m), enabled=n is not None,
                                      group=f"gemini:{m}", key=f"gemini#{n or 1}"))
        else:
            key = config.COHERE_CHAT_KEY if p == "cohere" else config.API_KEYS[p]
            providers.append(Provider(f"{p}:{m}", _make(p, m), enabled=bool(key)))
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


def lookup(messages: list[dict], system: str, temperature: float, json_mode: bool, only: list[str] | None = None,
           parse: Callable[[str], Any] | None = None, low_thinking: bool = False, **_) -> ChatResult | None:
    """תשובה מהמטמון בלבד (בלי קריאה), עם אותו מפתח כמו chat - כדי לדעת מראש אם יהיה שלב "מנסח תשובה"."""
    key = cache_key(messages, system, temperature, json_mode, only, low_thinking)
    return _from_cache(key, parse) if CACHE_MODE == "on" else None


def _lanes(route: str | None, names: list[str] | None, only: list[str] | None,
           low_thinking: bool) -> list[tuple[Provider, bool, int]]:
    """[(ספק, חשיבה נמוכה, דרגה)] לפי הסדר. names (מודל קבוע בבדיקות) קודם ל-route; בלי שניהם - only, או כל השרשרת
    בלי מודל הגיבוי (הקריאות שלו שמורות להבנה ולתשובה). ספק בלי מפתח - לא בנתיב."""
    if names is not None:
        by_name = {p.name: p for p in PROVIDERS}
        lanes = [(by_name[n], low_thinking, 0) for n in names if n in by_name]
    elif route is not None:
        lanes = [(p, low_thinking or low, tier) for target, low, tier in config.route(route)
                 for p in PROVIDERS if target in (p.name, p.group)]
    elif only:
        lanes = [(p, low_thinking, 0) for p in PROVIDERS if p.name.split(":")[0] in only]
    else:
        lanes = [(p, low_thinking, 0) for p in PROVIDERS if p.name != config.BACKUP_LLM]
    return [lane for lane in lanes if lane[0].enabled]


def policy(route: str | None, hedge: bool) -> gateway.Policy:
    """בצ'אט (hedge) - לפי הנתיב: אחרי כמה שניות מתחיל הבא בתור, ועד מתי תשובה של המודל הראשי עדיפה. בסקריפטים -
    בזה אחר זה, בסבלנות (config.patient)."""
    return gateway.Policy(hedge_after=config.HEDGE_AFTER_SECONDS if hedge else None,
                          prefer_until=config.PREFER_SECONDS.get(route or "", 0.0) if hedge else 0.0,
                          margin=config.PREFER_MARGIN.get(route or "", 0.5), timeout=config.QUERY_TIMEOUT,
                          retries=config.LLM_RETRIES, max_wait=config.LLM_MAX_WAIT)


def chat(messages: list[dict], system: str = "", temperature: float = 0.2, json_mode: bool = False,
         parse: Callable[[str], Any] | None = None, only: list[str] | None = None,
         names: list[str] | None = None, cache: bool = True, low_thinking: bool = False,
         route: str | None = None, hedge: bool | None = None,
         rank: Callable[[ChatResult], int] | None = None) -> ChatResult:
    """messages: [{"role": "user" | "assistant", "content": "..."}]. כל הקריאות עוברות בשער המודלים (gateway.py).

    json_mode: לבקש JSON (ב-Gemini גם ברמת ה-API; אצל השאר לפי הפרומפט).
    parse: מפענח/בודק את התשובה. אם הוא זורק ValueError (למשל JSON שבור) - התשובה פגומה, והמודל הבא בנתיב.
    route: שם הנתיב ב-config.ROUTES ("understand", "answer", "helper") - המודלים, לפי הסדר.
    only: רק ספקים מהסוגים האלה, למשל ["groq", "openrouter"] (השופט בבדיקות).
    names: רק הספקים האלה, בסדר הזה - מודל קבוע בבדיקות (eval_answers.pin_models); קודם ל-route.
    hedge: True - הראשי לבד, ושאר הנתיב יחד אחרי HEDGE_AFTER_SECONDS בלי תשובה או מיד אחרי תקלה; False - בזה אחר זה.
           ברירת המחדל: בצ'אט (יש מגבלת זמן לשאלה) - True; בסקריפטים - False.
    rank(תשובה) -> 0 טובה, יותר - פחות טובה: תשובה שנפסלה בבדיקות מפעילה את הבא בנתיב, והטובה ביותר נבחרת.
    cache: False - בלי לקרוא מהמטמון ובלי לכתוב אליו. נשמרת רק התשובה שנבחרה.
    low_thinking: ב-Gemini - רמת החשיבה הנמוכה שהמודל מקבל (config.LOW_THINKING); ב-Cohere - תקציב חשיבה קטן.
    """
    key = cache_key(messages, system, temperature, json_mode, only, low_thinking)
    if cache and CACHE_MODE == "on" and (hit := _from_cache(key, parse)):
        return hit
    if hedge is None:
        hedge = fallback.time_left() is not None and config.PARALLEL_LLM

    def lane(p: Provider, low: bool, tier: int) -> gateway.Lane:
        def call() -> ChatResult:
            extra = {"low_thinking": True} if low and p.name.startswith(("gemini:", "cohere:")) else {}
            text = p.call(messages, system, temperature, json_mode=json_mode, timeout=config.QUERY_TIMEOUT, **extra)
            # המפתח שענה בפועל (אחרי שהמכסה של הראשי נגמרה - לא הראשי): "gemini:gemini-3.6-flash#4"
            number = gemini_keys.answered_by() if p.name.startswith("gemini:") else None
            try:
                return ChatResult(text, f"{p.group}#{number}" if number else p.name, parse(text) if parse else None)
            except ValueError as exc:
                raise EmptyResult(f"unusable response: {exc}") from exc
        return gateway.Lane(p.name, call, tier)

    lanes = [lane(*spec) for spec in _lanes(route, names, only, low_thinking)]
    picked = gateway.run(route or "llm", lanes, policy(route, hedge), rank=rank)
    result = picked.value
    if len(picked.started) > 1:
        result.route = " מול ".join(describe(n) for n in picked.started)
    if cache and CACHE_MODE != "off":
        _cache_put(key, {"text": result.text, "provider": result.provider, "saved": time.strftime("%Y-%m-%d %H:%M")})
    return result
