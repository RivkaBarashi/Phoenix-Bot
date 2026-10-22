"""בדיקת ב2 (זיכרון של כל השיחה) על סט השיחות data/conversations.json.

לכל הודעה של הלקוחה - כמו בצ'אט: הבנת השאלה (עם השיחה עד עכשיו), ירושת סוג הביטוח מזיכרון השיחה, ושאלת הבהרה. התוצאה
נבדקת מול הציפיות שבקובץ: סוג הביטוח, מילים שחייבות (או אסור) להופיע בשאלה שמחפשים, שאלת הבהרה, כוונה. רק שלב ההבנה -
בלי חיפוש ובלי מודל תשובה, ולכן זול: ~57 קריאות לגרסה.

הרצה (במפתח בתשלום - בחינמי זה כמעט כל המכסה היומית של 3.8):
  python eval_conversations.py              שתי הגרסאות, זו מול זו (A/B מזווג): בלי זיכרון / עם זיכרון
  python eval_conversations.py --memory on  גרסה אחת (on / off)
  --fresh                                   בלי המטמון של המודלים
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONVERSATIONS = ROOT / "data" / "conversations.json"
NON_TOPICAL = ("greeting", "thanks", "off_topic", "injection")


def run_conversation(conv: dict) -> list[dict]:
    """שיחה אחת, הודעה אחרי הודעה. התשובות של הבוט בשיחה - הטקסט הקבוע מהקובץ (רק להקשר), עם זיכרון הנושא כמו בצ'אט."""
    from phoenix import answer as A
    from phoenix import config, understand
    history, out = [], []
    for n, turn in enumerate(conv["turns"], 1):
        question = turn["user"]
        memory = understand.last_memory(history) if config.CONVERSATION_MEMORY else None
        u, kind, standalone, inherited = None, "answer", question, None
        if A.greeting_answer(question):
            kind, intent = "greeting", "greeting"
        else:
            u = understand.try_understand(question, history)
            intent = u.intent if u else "failed"
            if u is None:
                kind = "failed"  # תקלה בשלב ההבנה - נספרת בנפרד, לא כטעות בזיכרון
            elif u.intent in NON_TOPICAL:
                kind, standalone = "other", u.standalone_question
            else:
                inherited = A.inherited_type(u, memory, question)
                standalone = u.sub_questions[0] if len(u.sub_questions) > 1 else u.standalone_question
                if inherited and not A.has_insurance_type(standalone):
                    standalone = A.with_insurance_type(standalone, inherited)
                if u.needs_clarification and not u.insurance_type and not inherited \
                        and not (memory or {}).get("clarified"):
                    kind = "clarify"
        answer = A.Answer("", kind, kind, [], None, "none", question=standalone)
        new_memory = A.next_memory(memory, u, answer) if config.CONVERSATION_MEMORY and kind != "greeting" else None
        kind_type = (inherited or (u.insurance_type if u else None)) if kind not in ("other", "greeting") else None
        problems = check(turn, kind, intent, standalone, kind_type)
        out.append({"conversation": conv["id"], "turn": n, "question": question, "standalone": standalone,
                    "type": kind_type, "intent": intent, "kind": kind, "inherited": inherited,
                    "llm": u.llm if u else None, "problems": problems, "ok": not problems and kind != "failed"})
        history += [{"role": "user", "content": question},
                    {"role": "assistant", "content": turn.get("bot") or "..."} | ({"memory": new_memory} if new_memory else {})]
    return out


def check(turn: dict, kind: str, intent: str, standalone: str, kind_type: str | None) -> list[str]:
    """מה לא תאם את הציפיות. סוג הביטוח נבדק רק בהודעה עם נושא (לא בתודה, ברכה או ניסיון לשנות את ההוראות)."""
    problems = []
    if kind == "failed":
        return ["understanding failed"]
    if "intent" in turn and intent != turn["intent"]:
        problems.append(f"intent {intent} (expected {turn['intent']})")
    if "clarify" in turn and (kind == "clarify") != turn["clarify"]:
        problems.append(f"clarify {kind == 'clarify'} (expected {turn['clarify']})")
    if "type" in turn and kind not in ("other", "greeting") and kind != "clarify" and kind_type != turn["type"]:
        problems.append(f"type {kind_type} (expected {turn['type']})")
    for word in turn.get("has", []):
        if word not in standalone:
            problems.append(f"missing '{word}'")
    for word in turn.get("not", []):
        if word in standalone:
            problems.append(f"has '{word}'")
    return problems


def run_mode(fresh: bool) -> list[dict]:
    from phoenix import config, llm
    config.patient()  # כמו eval_answers: בזה אחר זה, בלי מגבלת הזמן של הצ'אט
    if fresh:
        llm.CACHE_MODE = "write"
    conversations = json.loads(CONVERSATIONS.read_text(encoding="utf-8"))["conversations"]
    results = []
    for conv in conversations:
        rows = run_conversation(conv)
        results += rows
        for r in rows:
            print(f"[{r['conversation']}.{r['turn']}] {'OK ' if r['ok'] else 'BAD'} {r['standalone']}"
                  + (f"  <- {'; '.join(r['problems'])}" if r["problems"] else ""))
    return results


def summary(label: str, results: list[dict]) -> str:
    ok = sum(r["ok"] for r in results)
    failed = sum(r["kind"] == "failed" for r in results)
    clarify = sum(r["kind"] == "clarify" for r in results)
    return f"{label}: {ok}/{len(results)} תקינות | תקלות בהבנה: {failed} | שאלות הבהרה: {clarify}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--memory", choices=["on", "off"])
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--out")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    if args.memory:  # גרסה אחת - המשתנה כבר נקבע לפני שהקוד של הבוט נטען (ההוראות נבנות בטעינה)
        results = run_mode(args.fresh)
        print(summary(f"זיכרון {args.memory}", results))
        if args.out:
            Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        return
    runs = {}
    for mode in ("off", "on"):  # כל גרסה בתהליך נפרד: CONVERSATION_MEMORY נקרא כשהקוד נטען
        out = ROOT / "data" / f"eval_conversations_{mode}.json"
        env = os.environ | {"CONVERSATION_MEMORY": "true" if mode == "on" else "false", "PYTHONIOENCODING": "utf-8"}
        print(f"\n=== זיכרון {mode} ===")
        subprocess.run([sys.executable, __file__, "--memory", mode, "--out", str(out)] + (["--fresh"] if args.fresh else []),
                       env=env, check=True)
        runs[mode] = json.loads(out.read_text(encoding="utf-8"))
    print("\n=== השוואה (אותן הודעות בשתי הגרסאות) ===")
    for mode in ("off", "on"):
        print(summary(f"זיכרון {mode}", runs[mode]))
    pairs = list(zip(runs["off"], runs["on"]))
    better = [b for a, b in pairs if b["ok"] and not a["ok"]]
    worse = [b for a, b in pairs if a["ok"] and not b["ok"]]
    print(f"השתפרו: {len(better)} | נפגעו: {len(worse)}")
    for label, rows in (("השתפרו", better), ("נפגעו", worse)):
        for r in rows:
            print(f"  {label}: [{r['conversation']}.{r['turn']}] {r['question']}")


if __name__ == "__main__":
    main()
