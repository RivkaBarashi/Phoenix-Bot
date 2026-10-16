"""בדיקה שכל הספקים עובדים ושמנגנון הגיבוי עובר לספק הבא.

הרצה:  .venv\\Scripts\\python check_providers.py
"""
import logging
import sys
import time

import httpx

from phoenix import embeddings, fetch, llm, rerank
from phoenix.fallback import Provider, run_chain

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.WARNING, format="     [%(levelname)s] %(message)s")

QUESTION = [{"role": "user", "content": "בשני משפטים: מה זה ביטוח דירה?"}]
QUERY = "איך מגישים תביעה על נזקי מים?"
DOCS = [
    "ביטוח רכב חובה ומקיף - מה ההבדל ביניהם",
    "הגשת תביעה על נזקי מים בביטוח דירה: מצלמים את הנזק ופונים למוקד התביעות",
    "דמי ניהול בקרן פנסיה",
]
TEST_URL = "https://www.fnx.co.il/"


def check(label, fn):
    start = time.time()
    try:
        result = fn()
        print(f"OK   {label} ({time.time() - start:.1f}s): {result}")
    except Exception as exc:
        print(f"FAIL {label}: {type(exc).__name__}: {str(exc)[:250]}")


def short(text, n=110):
    return " ".join(text.split())[:n]


print("\n== מודל שפה - כל ספק בנפרד ==")
for p in llm.PROVIDERS:
    check(p.name, lambda p=p: short(p.call(QUESTION, "ענה בעברית.", 0.2)))

print("\n== אמבדינגים - כל ספק בנפרד ==")
for p in embeddings.PROVIDERS:
    check(p.name, lambda p=p: f"dim {len(p.call([QUERY], True)[0])}")

print("\n== Rerank - כל ספק בנפרד (הקטע הנכון הוא 1) ==")
for p in rerank.PROVIDERS:
    check(p.name, lambda p=p: [(i, round(s, 3)) for i, s in p.call(QUERY, DOCS, 3)])

print("\n== איסוף עמוד - כל ספק בנפרד ==")
for p in fetch.PROVIDERS:
    check(p.name, lambda p=p: (lambda t: f"title={t[0]!r}, {len(t[1])} chars")(p.call(TEST_URL)))

print("\n== הדגמת גיבוי: ספק ראשון 'נחסם' (429), צריך לעבור לבא ==")


def blocked(*_):
    req = httpx.Request("POST", "https://example.invalid")
    raise httpx.HTTPStatusError("429 simulated", request=req, response=httpx.Response(429, request=req))


chain = [Provider("simulated:blocked", blocked), *llm.PROVIDERS]
check("fallback", lambda: "answered by " + run_chain("llm-demo", chain, QUESTION, "", 0.2, retries=0)[1])

print("\n== הפונקציות שהבוט ישתמש בהן ==")
check("chat()", lambda: (lambda r: f"[{r.provider}] {short(r.text)}")(llm.chat(QUESTION)))
check("embed_query()", lambda: (lambda v: f"[{v[1]}] dim {len(v[0])}")(embeddings.embed_query(QUERY)))
check("rerank()", lambda: rerank.rerank(QUERY, DOCS, 2))
check("fetch_page()", lambda: (lambda pg: f"[{pg.via}] {pg.title!r} {len(pg.text)} chars")(fetch.fetch_page(TEST_URL)))
