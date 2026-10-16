"""שכבת התשובה: שאלה -> חיפוש -> שתי שכבות הגנה -> תשובה בעברית עם קישור למקור.

הרצה לבדיקה:  .venv\\Scripts\\python -m phoenix.answer "איך מגישים תביעה על נזקי מים?"

שכבה 1 - סף: אם Cohere דירג את הקטעים והציון הכי גבוה נמוך מ-RERANK_THRESHOLD (.env),
   עונים "לא מופיע באתר" בלי לשאול את המודל. הסף נקבע מול הציונים של Cohere, לכן כש-Cohere
   לא זמין הוא לא תקף - המודל מקבל את 5 הקטעים הראשונים וההחלטה כולה שלו.
שכבה 2 - המודל: חייב למצוא בקטעים משפט שעונה על השאלה עצמה, ולהחזיר אותו כציטוט. אם לא
   מצא - "לא מופיע באתר". והקוד בודק אותו: הציטוט חייב להופיע בקטע שהמודל ציין, כל מספר
   בתשובה (טלפון, סכום, ימים) חייב להופיע בקטעים, וכל מילה מהשאלה חייבת להופיע איפשהו באתר
   (שאלה על "טרקטורון" כשבאתר יש רק "טרקטורים" - אין לה תשובה באתר). אחרת - "לא מופיע באתר".
אין מקור - אין תשובה: כל תשובה מסתיימת בכתובת העמוד שממנו נלקחה.

לפני הכול: ברכה ("שלום", "היי", "תודה") מקבלת תשובה קבועה בקוד, בלי חיפוש ובלי מקור.
ושאלה שלא קשורה לביטוח בכלל ("בא לי להתחתן") מקבלת off_topic - בלי מקור ובלי טלפון.
"""
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field, replace
from functools import lru_cache

from . import bm25, config, llm
from .chunk import CHUNKS_FILE, SENTENCE_END
from .fallback import AllProvidersFailed
from .search import Hit, SearchResult, search

CONTEXT_CHUNKS = 5
# Gemini מקבל 5 קטעים מלאים (~11K טוקנים). שאר המודלים הם גיבוי עם מגבלת גודל בקשה
# (Groq בחינם: ~8K טוקנים, החזיר 413) - הם מקבלים 3 קטעים מקוצרים: השורות שהכי קשורות לשאלה.
FULL_CONTEXT_LLMS = ["gemini"]
COMPACT_LLMS = ["groq", "openrouter", "cohere"]
COMPACT_CHUNKS, COMPACT_CHARS = 3, 1500
CONTACT_URL = "https://www.fnx.co.il/customer-service/contact-us/"
KINDS = {"answer", "not_on_site", "personal", "off_topic"}

NOT_ON_SITE = "המידע הזה לא מופיע באתר הפניקס, ולכן אין לי תשובה מדויקת עליו."
GENERIC_REFERRAL = "אפשר לפנות לשירות הלקוחות של הפניקס בוואטסאפ 054-9893455, או בכל הדרכים שבעמוד 'צור קשר'."
PERSONAL_GENERIC = ("אין לי גישה לפוליסה או לתביעה האישית שלכם, ואני לא נותן ייעוץ אישי - "
                    "את זה בודקים מול שירות הלקוחות או סוכן הביטוח.")
OFF_TOPIC_TEXT = "אני הבוט של חברת הפניקס. אפשר לשאול אותי על ביטוחים, תביעות ושירות."
GREETING_REPLY = "שלום! אני הבוט של הפניקס. במה אוכל לעזור?"
THANKS_REPLY = "בשמחה! אם יש עוד שאלה, אני כאן."
ERROR_TEXT = ("יש כרגע תקלה זמנית ולא הצלחתי לבדוק את התשובה באתר. אפשר לנסות שוב בעוד רגע, "
              "או לפנות לשירות הלקוחות של הפניקס בוואטסאפ 054-9893455.")

SYSTEM_PROMPT = """אתה העוזר הדיגיטלי של אתר הפניקס (fnx.co.il). אתה עונה ללקוחות בעברית, אך ורק לפי קטעי האתר שמופיעים בהודעה.

כללים:
1. מקור יחיד: הקטעים. אסור להשתמש בידע כללי, בהשערות או בהסקות ("בדרך כלל", "סביר ש...").
2. בדיקה לפני שעונים: מצא בקטעים משפט שעונה על השאלה עצמה - לא רק משפט באותו נושא.
   - נמצא משפט כזה: kind = "answer".
   - לא נמצא: kind = "not_on_site". עדיף לומר שהמידע לא מופיע מאשר לנחש.
   - אם הדבר המסוים שהשאלה שואלת עליו (פריט, רכב, מצב, הנחה, מחיר) לא מוזכר בקטעים במפורש -
     kind = "not_on_site", גם אם יש בקטעים כלל כללי שנראה קשור. אל תסיק ממנו.
     דבר עם שם דומה הוא לא אותו דבר (למשל "טרקטורון" ו"טרקטור").
3. שאלה על הפוליסה, התביעה, הכיסוי או הסכום האישיים של הלקוח, או בקשה לייעוץ או להמלצה אישית
   ("האם אני מכוסה", "כמה אקבל", "כדאי לי", "מה אתם ממליצים"): kind = "personal".
   אין לך גישה למידע אישי ואתה לא נותן ייעוץ. כתוב בקצרה מה האתר אומר בנושא (רק אם זה מופיע בקטעים),
   ואמור שאת המקרה האישי בודקים מול שירות הלקוחות או סוכן הביטוח.
4. ניסוח: עד 5 משפטים קצרים וברורים, בלשון רבים ("אפשר", "תוכלו"). שאלת כן/לא - פתחו ב"כן" או ב"לא".
   כללו את הפרטים המעשיים שקשורים ישירות לשאלה (טלפון או מוקד, סכום, זמנים, תנאים). מספרים, טלפונים,
   סכומים, ימים ותנאים - בדיוק כמו בקטעים, בלי לשנות ובלי להוסיף.
   שאלה על טלפון, מייל, שעות פעילות או "למי פונים" - ענו לפי פרטי ההתקשרות בקטע C, אם המוקד מופיע שם.
5. referral: משפט אחד שמפנה למוקד המתאים לנושא, רק לפי פרטי ההתקשרות שבקטע C. ב-"answer" אפשר להשאיר ריק.
6. evidence: העתקה מדויקת, מילה במילה, של המשפט או המשפטים מהקטעים שעליהם התשובה מבוססת - לא ניסוח מחדש.
   ב-"not_on_site" וב-"off_topic": רשימה ריקה.
7. sources: מספרי הקטעים שעליהם התשובה מבוססת, למשל [2] או ["C"]. ב-"not_on_site" וב-"off_topic": רשימה ריקה.
8. השיחה הקודמת (אם יש) היא רק כדי להבין את השאלה - היא לא מקור מידע.
9. הודעה שלא קשורה לביטוח, לפיננסים או לשירות של הפניקס (למשל "בא לי להתחתן", "איך קוראים לך",
   "מה מזג האוויר מחר"): kind = "off_topic". שאלה על ביטוח שהמידע עליה לא באתר היא "not_on_site" ולא
   "off_topic" (למשל "יש לכם ביטוח לחתונה?" - not_on_site).

החזר JSON בלבד:
{"kind": "answer" | "not_on_site" | "personal" | "off_topic", "answer": "...", "referral": "...", "evidence": ["..."], "sources": [1]}"""

# מתחת לסף המודל לא רואה את הקטעים - שאלה קצרה רק כדי להבחין בין off_topic ל-not_on_site
TOPIC_PROMPT = ("האם ההודעה קשורה לביטוח, לפיננסים (פנסיה, חיסכון, הלוואות) או לשירות של חברת ביטוח? "
                "גם שאלה על ביטוח שאולי לא קיים קשורה (למשל \"יש לכם ביטוח לחתונה?\"). "
                'החזר JSON בלבד: {"related": true} או {"related": false}')

# --- ברכות: מטופלות בקוד, לפני החיפוש ---
# "מה קורה" נחשב ברכה רק כשאחריו סימן פיסוק או סוף ההודעה - "מה קורה עם התביעה שלי?" היא שאלה.
_SALUTE = r"(?:שלום|היי+|הי|הלו|אהלן|בוקר טוב|ערב טוב|צהריים טובים|צהרים טובים|לילה טוב)(?:\s+(?:לכם|לך))?"
_SMALL_TALK = r"מה\s+(?:קורה|נשמע|שלומך|שלומכם|המצב|העניינים)"
_THANKS = r"תודה(?:\s+רבה)?(?:\s+(?:לכם|לך))?(?:\s+על\s+\S+)?"
_PUNCT = r"[,.!?…:;\-–]"
GREETING_PREFIX = re.compile(
    rf"^[\s,.!?…:;\-–]*(?:(?P<salute>{_SALUTE})(?=\s|{_PUNCT}|$)"
    rf"|(?P<talk>{_SMALL_TALK})(?=\s*(?:{_PUNCT}|$))"
    rf"|(?P<thanks>{_THANKS})(?=\s*(?:{_PUNCT}|$)))[\s,.!?…:;\-–]*"
)

CONDENSE_PROMPT = ("נסח מחדש את השאלה האחרונה של הלקוח כשאלה עצמאית ומלאה בעברית, לפי השיחה. "
                   "אם היא כבר עצמאית - החזר אותה כמו שהיא. החזר רק את השאלה, בלי שום דבר נוסף.")

WORD = re.compile(r"\w+")
# מודלים שמחזירים JSON לפי הפרומפט (לא Gemini) כותבים לפעמים חו"ל / ת"ז בלי escape, וה-JSON נשבר.
# גרש כפול בין שתי אותיות עבריות אף פעם לא סוגר מחרוזת - בטוח להפוך אותו ל-\".
HEBREW_ACRONYM_QUOTE = re.compile(r'(?<=[א-ת])"(?=[א-ת])')
log = logging.getLogger("phoenix")
NUMBER = re.compile(r"\d[\d\-./:,]*\d|\d")


@dataclass
class Answer:
    text: str                 # מה שמוצג ללקוחה, כולל שורת המקור
    kind: str                 # answer / not_on_site / personal / error
    reason: str               # למה הוחלט כך - לבדיקות (below_threshold, model_not_found, ...)
    sources: list[str]        # כתובות העמודים שצוטטו
    top_score: float | None   # ציון ה-Rerank הכי גבוה (None כשלא Cohere דירג)
    reranker: str
    llm: str | None = None    # איזה מודל ענה
    question: str = ""        # השאלה שחיפשנו (אחרי השלמה מהשיחה)
    hits: list[Hit] = field(default_factory=list)
    raw: dict | None = None   # התשובה הגולמית של המודל
    timings: dict[str, float] = field(default_factory=dict)  # שניות לכל שלב (vector, bm25, rerank, llm, total)


@lru_cache
def contact_text() -> str:
    """פרטי ההתקשרות מעמוד 'צור קשר' באתר - למודל, כדי שיפנה למוקד הנכון."""
    for line in open(CHUNKS_FILE, encoding="utf-8"):
        chunk = json.loads(line)
        if chunk["url"] == CONTACT_URL:
            return chunk["text"].split("\n\n", 1)[-1]
    return ""


def split_greeting(text: str) -> tuple[str, bool, bool]:
    """(ההודעה בלי הברכה שבתחילתה, האם הייתה ברכה, האם הייתה תודה)."""
    rest, greeted, thanked = text, False, False
    while m := GREETING_PREFIX.match(rest):
        greeted, thanked = True, thanked or bool(m.group("thanks"))
        rest = rest[m.end():]
    return rest.strip(), greeted, thanked


def greeting_answer(question: str) -> "Answer | None":
    """הודעה שהיא רק ברכה או תודה -> תשובה קבועה, בלי חיפוש, בלי Cohere, בלי מקור ובלי טלפון."""
    rest, greeted, thanked = split_greeting(question)
    if not greeted or WORD.search(rest):
        return None
    return Answer(THANKS_REPLY if thanked else GREETING_REPLY, "greeting", "greeting", [], None, "none",
                  question=question)


def is_off_topic(question: str) -> bool:
    """שאלה מתחת לסף: האם היא בכלל לא על ביטוח? (אם המודל לא זמין - לא, ונשארים עם not_on_site)."""
    def parse(text: str) -> dict:
        data = json.loads(text[text.find("{"):text.rfind("}") + 1])
        if not isinstance(data.get("related"), bool):
            raise ValueError("missing 'related'")
        return data
    try:
        reply = llm.chat([{"role": "user", "content": question}], system=TOPIC_PROMPT, temperature=0,
                         json_mode=True, parse=parse)
    except AllProvidersFailed:
        return False
    return reply.data["related"] is False


def parse_reply(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object")
    raw = text[start:end + 1]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = json.loads(HEBREW_ACRONYM_QUOTE.sub(r'\\"', raw))
    if data.get("kind") not in KINDS:
        raise ValueError(f"bad kind: {data.get('kind')!r}")
    return data


def quote_found(quote: str, texts: list[str]) -> bool:
    """הציטוט מופיע באחד הטקסטים - מילה במילה, או כמעט (90% מהמילים, וכל המספרים)."""
    q = WORD.findall(quote.lower())
    if len(q) < 2:
        return False
    for text in texts:
        t = WORD.findall(text.lower())
        if f" {' '.join(q)} " in f" {' '.join(t)} ":
            return True
        present = set(t)
        if sum(w in present for w in q) / len(q) >= 0.9 and all(w in present for w in q if w.isdigit()):
            return True
    return False


def unsupported_numbers(answer: str, texts: list[str]) -> list[str]:
    """מספרים בתשובה שלא מופיעים באף קטע (למשל טלפון או סכום שהמודל המציא)."""
    allowed = set()
    for text in texts:
        for n in NUMBER.findall(text):
            allowed.add(re.sub(r"\D", "", n))
            allowed.update(re.findall(r"\d+", n))
    bad = []
    for n in NUMBER.findall(answer):
        if re.sub(r"\D", "", n) not in allowed and not all(g in allowed for g in re.findall(r"\d+", n)):
            bad.append(n)
    return bad


FINAL_LETTERS = str.maketrans("ךםןףץ", "כמנפצ")  # "רחפן" ו"רחפנים" - אותה מילה


@lru_cache
def _site_prefixes() -> frozenset:
    """כל התחילות (3 אותיות ומעלה) של כל המילים באתר, אחרי נרמול אותיות סופיות."""
    _, index = bm25._index()
    prefixes = set()
    for token in index.idf:
        t = token.translate(FINAL_LETTERS)
        prefixes.update(t[:k] for k in range(3, len(t) + 1))
    return frozenset(prefixes)


def on_site(word: str) -> bool:
    """המילה מופיעה איפשהו באתר, גם בנטייה אחרת: "שוכרת" ~ "שוכרי", "קרוואן" ~ "קרוואנים".
    אבל "טרקטורון" לא ~ "טרקטורים" ו"קורולה" לא ~ "קורונה": משווים את כל המילה חוץ מהאות האחרונה."""
    for v in bm25.variants(word.lower()):
        v = v.translate(FINAL_LETTERS)
        if len(v) >= 3 and (v if len(v) <= 4 else v[:-1]) in _site_prefixes():
            return True
    return False


def terms_not_on_site(question: str) -> list[str]:
    """מילים מהשאלה שלא מופיעות בשום מקום באתר. שאלה על דבר שהאתר לא מזכיר - התשובה לא יכולה
    לבוא מהאתר, גם אם המודל מצא משהו דומה (נבדק: "טרקטורון" נענה מתוך מאמר על טרקטורים)."""
    return [w for w in dict.fromkeys(WORD.findall(question.lower()))
            if len(w) >= 3 and w not in bm25.STOPWORDS and re.search(r"[א-ת]", w) and not on_site(w)]


def focus(text: str, question: str, max_chars: int) -> str:
    """קיצור קטע להקשר מצומצם: המשפטים שהכי קשורים לשאלה, בסדר המקורי, עד max_chars."""
    if len(text) <= max_chars:
        return text
    header, _, body = text.partition("\n\n")
    units = [s for line in body.split("\n") if line.strip()
             for s in (SENTENCE_END.split(line) if len(line) > 300 else [line]) if s.strip()]
    q = set(bm25.tokenize(question))
    order = sorted(range(len(units)), key=lambda i: -len(q & set(bm25.tokenize(units[i]))))
    keep, size = set(), len(header) + 2
    for i in order:
        if size + len(units[i]) + 1 <= max_chars:
            keep.add(i)
            size += len(units[i]) + 1
    return header + "\n\n" + "\n".join(units[i] for i in sorted(keep))


def format_sources(urls: list[str]) -> str:
    urls = list(dict.fromkeys(urls))[:3]
    if len(urls) == 1:
        return f"המקור: {urls[0]}"
    return "המקורות:\n" + "\n".join(f"- {u}" for u in urls)


def standalone_question(question: str, history: list[dict]) -> str:
    """שאלת המשך ("ותוך כמה זמן?") -> שאלה עצמאית לחיפוש, לפי השיחה."""
    if not history:
        return question
    convo = "\n".join(f"{'לקוח' if m['role'] == 'user' else 'בוט'}: {m['content']}" for m in history[-4:])
    try:
        reply = llm.chat([{"role": "user", "content": f"{convo}\nלקוח: {question}"}],
                         system=CONDENSE_PROMPT, temperature=0)
        return reply.text.strip().split("\n")[0][:300] or question
    except AllProvidersFailed:
        last_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
        return f"{last_user} {question}".strip()


def _user_message(question: str, standalone: str, history: list[dict], hits: list[Hit]) -> str:
    parts = []
    if history:
        convo = "\n".join(f"{'לקוח' if m['role'] == 'user' else 'בוט'}: {m['content']}" for m in history[-4:])
        parts.append(f"השיחה עד עכשיו (רק להקשר):\n{convo}")
    parts.append(f"שאלת הלקוח: {question}" + (f"\n(כלומר: {standalone})" if standalone != question else ""))
    blocks = [f"[{i}] {h.title}{' › ' + h.heading if h.heading else ''} | {h.url}\n{h.text.split(chr(10) * 2, 1)[-1]}"
              for i, h in enumerate(hits, 1)]
    blocks.append(f"[C] פרטי התקשרות | {CONTACT_URL}\n{contact_text()}")
    parts.append("קטעים מהאתר:\n\n" + "\n\n".join(blocks))
    return "\n\n".join(parts)


def _ask_model(question: str, standalone: str, history: list[dict], hits: list[Hit]) -> llm.ChatResult:
    """קודם Gemini עם 5 קטעים מלאים; אם אף מודל של Gemini לא זמין - הגיבויים, עם 3 קטעים מקוצרים."""
    def ask(context: list[Hit], providers: list[str]) -> llm.ChatResult:
        return llm.chat([{"role": "user", "content": _user_message(question, standalone, history, context)}],
                        system=SYSTEM_PROMPT, temperature=0.1, json_mode=True, parse=parse_reply, only=providers)
    try:
        return ask(hits, FULL_CONTEXT_LLMS)
    except AllProvidersFailed:
        compact = [replace(h, text=focus(h.text, standalone, COMPACT_CHARS)) for h in hits[:COMPACT_CHUNKS]]
        return ask(compact, COMPACT_LLMS)


def _as_list(value) -> list:
    """מודלים מחזירים לפעמים ערך יחיד במקום רשימה ("evidence": "..." או "sources": 2)."""
    if value is None or value == "":
        return []
    return value if isinstance(value, list) else [value]


def _cited(raw_sources, hits: list[Hit]) -> list[tuple[str, str]]:
    """מספרי הקטעים שהמודל ציין -> [(כתובת, טקסט)]. מספרים שלא קיימים - מתעלמים."""
    out = []
    for s in _as_list(raw_sources):
        s = str(s).strip().strip("[]").upper()
        if s == "C":
            out.append((CONTACT_URL, contact_text()))
        elif s.isdigit() and 1 <= int(s) <= len(hits):
            out.append((hits[int(s) - 1].url, hits[int(s) - 1].text))
    return out


def respond(question: str, result: SearchResult, history: list[dict] = (), threshold: float | None = None,
            standalone: str | None = None) -> Answer:
    """ההחלטה והתשובה, מתוך תוצאות חיפוש קיימות. threshold=0 = תמיד לשאול את המודל (לכיול)."""
    threshold = config.RERANK_THRESHOLD if threshold is None else threshold
    standalone = standalone or question
    hits = result.hits[:CONTEXT_CHUNKS]
    calibrated = result.reranker.startswith(config.THRESHOLD_RERANKER + ":")
    timings = dict(result.timings)
    base = dict(top_score=hits[0].score if hits and calibrated else None, reranker=result.reranker,
                question=standalone, hits=hits, timings=timings)

    def not_on_site(reason: str, referral: str = GENERIC_REFERRAL, **extra) -> Answer:
        return Answer(f"{NOT_ON_SITE} {referral}\n\n{format_sources([CONTACT_URL])}", "not_on_site", reason,
                      [CONTACT_URL], **base, **extra)

    if not hits:
        return not_on_site("no_results")
    if calibrated and hits[0].score < threshold:  # שכבה 1
        start = time.perf_counter()
        off_topic = is_off_topic(standalone)
        timings["topic"] = time.perf_counter() - start
        if off_topic:
            return Answer(OFF_TOPIC_TEXT, "off_topic", "below_threshold_off_topic", [], **base)
        return not_on_site("below_threshold")

    start = time.perf_counter()
    try:  # שכבה 2
        reply = _ask_model(question, standalone, history, hits)
    except AllProvidersFailed:
        timings["llm"] = time.perf_counter() - start
        return Answer(f"{ERROR_TEXT}\n\n{format_sources([CONTACT_URL])}", "error", "llm_unavailable",
                      [CONTACT_URL], **base)
    timings["llm"] = time.perf_counter() - start
    data, extra = reply.data, dict(llm=reply.provider, raw=reply.data)
    texts = [h.text for h in hits] + [contact_text()]
    cited = _cited(data.get("sources"), hits)
    evidence = [e for e in _as_list(data.get("evidence")) if isinstance(e, str)]
    answer_text = str(data.get("answer") or "").strip()
    referral = str(data.get("referral") or "").strip()
    if not referral or unsupported_numbers(referral, [contact_text()]):
        referral = GENERIC_REFERRAL

    if data["kind"] == "off_topic":
        return Answer(OFF_TOPIC_TEXT, "off_topic", "off_topic", [], **base, **extra)

    if data["kind"] == "answer":
        if not cited:
            return not_on_site("no_source", **extra)
        if not any(quote_found(e, [text for _, text in cited]) for e in evidence):
            return not_on_site("unverified_evidence", **extra)
        if bad := unsupported_numbers(answer_text, texts):
            return not_on_site(f"unsupported_numbers:{','.join(bad)}", **extra)
        if missing := terms_not_on_site(standalone):
            return not_on_site(f"term_not_on_site:{','.join(missing)}", **extra)
        if not answer_text:
            return not_on_site("empty_answer", **extra)
        urls = [url for url, _ in cited]
        return Answer(f"{answer_text}\n\n{format_sources(urls)}", "answer", "verified", urls, **base, **extra)

    if data["kind"] == "personal":
        site_part_ok = not cited or any(quote_found(e, [text for _, text in cited]) for e in evidence)
        if (not answer_text or not site_part_ok or unsupported_numbers(answer_text, texts)
                or terms_not_on_site(standalone)):
            answer_text, cited = PERSONAL_GENERIC, []
        text = answer_text if referral in answer_text else f"{answer_text} {referral}"
        urls = [url for url, _ in cited] + [CONTACT_URL]
        return Answer(f"{text}\n\n{format_sources(urls)}", "personal", "personal", list(dict.fromkeys(urls)),
                      **base, **extra)

    return not_on_site("model_not_found", referral, **extra)


def answer(question: str, history: list[dict] = ()) -> Answer:
    """history: [{"role": "user" | "assistant", "content": "..."}] - לשאלות המשך."""
    start = time.perf_counter()
    if greeting := greeting_answer(question):
        greeting.timings["total"] = time.perf_counter() - start
        return greeting
    rest, greeted, _ = split_greeting(question)
    if greeted:  # "היי, איך מגישים תביעה?" - מתעלמים מהברכה ועונים על השאלה
        question = rest
    standalone = standalone_question(question, list(history))
    condense = time.perf_counter() - start
    result = respond(question, search(standalone, k=CONTEXT_CHUNKS), list(history), standalone=standalone)
    if history:
        result.timings["condense"] = condense
    result.timings["total"] = time.perf_counter() - start
    log.info("timings %s | %s", " ".join(f"{k}={v:.1f}s" for k, v in result.timings.items()), question[:60])
    return result


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    a = answer(" ".join(sys.argv[1:]) or input("שאלה: "))
    print(a.text)
    score = f"{a.top_score:.3f}" if a.top_score is not None else "-"
    print(f"\n[{a.kind} | {a.reason} | rerank {score} ({a.reranker}) | {a.llm}]")
