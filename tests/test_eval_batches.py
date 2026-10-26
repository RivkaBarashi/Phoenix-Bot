"""eval_batches.py: בדיקה בחבילות על מפתחות Gemini חינמיים - סבב המפתחות, ניסיון חוזר על תקלת תשתית, והפרדה בין
תשתית לאיכות (גם ב-ab_compare). בלי API: החבילות מזויפות."""
import pytest

import ab_compare
import eval_batches as B


def row(n, **fields) -> dict:
    return {"n": str(n), "type": "answerable", "question": f"q{n}", "decision": "answer", "reason": "verified",
            "verdict": "correct", "intent": "info", "grounding": "", "top": "0.9"} | fields


@pytest.mark.parametrize("fields, kind", [
    ({}, ""),
    ({"decision": "error", "reason": "llm_unavailable"}, "error"),
    ({"decision": "error", "reason": "search_failed: x"}, "error"),
    ({"decision": "unverified", "reason": "no_model:timeout", "verdict": ""}, "no_model"),
    ({"decision": "personal", "reason": "personal_no_model"}, "no_model"),
    ({"intent": "rules"}, "understand"),               # הבנת השאלה לא ענתה - כללי הקוד, לא מה שהצ'אט עושה
    ({"grounding": "unavailable:verifier: down"}, "verifier"),
    ({"verdict": "n/a"}, "judge"),
    ({"decision": "unverified", "reason": "unverified:unverified_evidence", "verdict": ""}, ""),  # איכות, לא תשתית
    ({"verdict": "wrong"}, ""),
])
def test_infrastructure_is_told_apart_from_quality(fields, kind):
    assert B.infra(row(1, **fields)) == kind


def test_parse_keys():
    assert B.parse_keys("1,3-5,9", list(range(1, 29))) == [1, 3, 4, 5, 9]
    assert B.parse_keys(None, [2, 4]) == [2, 4]
    for bad in ("3-x", "40", "a"):
        with pytest.raises(SystemExit):
            B.parse_keys(bad, list(range(1, 29)))


def fake_batches(behaviour):
    """behaviour(n, key) -> שורה, או None (השאלה חסרה בתוצאה - התהליך קרס). מחזיר run_batch ואת יומן הקריאות."""
    calls = []

    def run_batch(batch, key):
        calls.append((list(batch), key))
        return {n: r for n in batch if (r := behaviour(n, key)) is not None}
    return run_batch, calls


def test_a_question_that_failed_on_infrastructure_runs_again_on_the_next_key():
    run_batch, calls = fake_batches(lambda n, key: row(n, decision="error", reason="llm_unavailable")
                                    if (n, key) == (2, 1) else row(n))
    results = B.run_all([1, 2, 3, 4], [1, 2, 3], 2, 3, run_batch)
    assert calls == [([1, 2], 1), ([3, 4], 2), ([2], 3)]  # חבילה אחרת, מפתח אחר
    assert (results[2]["key"], results[2]["attempts"], B.infra(results[2])) == (3, 2, "")
    assert all(results[n]["attempts"] == 1 for n in (1, 3, 4))


def test_a_key_whose_whole_batch_failed_is_not_used_again():
    run_batch, calls = fake_batches(lambda n, key: None if key == 1 else row(n))  # המכסה של מפתח 1 נגמרה
    results = B.run_all([1, 2, 3, 4, 5, 6], [1, 2, 3], 2, 3, run_batch)
    assert calls[0] == ([1, 2], 1) and all(key != 1 for _, key in calls[1:])
    assert all(not B.infra(results[n]) for n in results)


def test_a_good_result_is_never_replaced_by_a_failure_and_attempts_are_limited():
    run_batch, calls = fake_batches(lambda n, key: row(n, decision="unverified", reason="no_model:timeout", verdict=""))
    results = B.run_all([1], [1, 2, 3, 4, 5], 1, 3, run_batch)
    assert len(calls) == 3 and results[1]["attempts"] == 3 and B.infra(results[1]) == "no_model"
    answers = iter([row(1), row(1, decision="error", reason="llm_unavailable")])
    run_batch, _ = fake_batches(lambda n, key: next(answers))
    assert B.infra(B.run_all([1], [1, 2], 1, 3, run_batch)[1]) == ""


def test_when_every_key_is_spent_the_rest_is_reported_as_infrastructure():
    run_batch, calls = fake_batches(lambda n, key: None)
    results = B.run_all([1, 2, 3], [1, 2], 1, 5, run_batch)
    assert len(calls) == 2 and all(B.infra(results[n]) == "error" for n in (1, 2, 3))


def test_the_summary_keeps_infrastructure_apart():
    results = {1: row(1, key=1), 2: row(2, verdict="wrong", key=1),
               3: row(3, decision="unverified", reason="no_model:timeout", verdict="", key=2, attempts=3)}
    text = "\n".join(B.summary(results))
    assert "נבדקו באיכות: 2" in text and "תקלות תשתית שנשארו: 1" in text and "[3] no_model" in text
    assert "correct: 1" in text and "wrong: 1" in text


def test_the_summary_uses_what_the_chat_shows():
    """שורה מקובץ: הדגלים כמחרוזות. אישית מתחת לסף - אישית (לא "לא מופיע"); חסימה לפני החיפוש - כמו שהיא."""
    personal = row(1, type="personal", decision="personal", verdict="", top="0.5", personal_rule="True")
    blocked = row(2, type="injection", decision="off_topic", verdict="blocked", top="0.0", pre_search="True")
    text = "\n".join(B.summary({1: personal, 2: blocked}))
    assert "personal     personal: 1" in text and "injection    off_topic: 1" in text


def test_main_runs_each_batch_in_its_own_process_with_another_key(monkeypatch, tmp_path):
    """הצנרת עד הסוף, בלי API: התהליך מזויף, ובודקים מה הוא קיבל - פקודה, מפתח, מתג - ומה נאסף לקובץ המאוחד."""
    import csv
    import subprocess
    questions = tmp_path / "set.csv"
    lines = "".join(f"q{n},e,,answerable\n" for n in range(1, 4))
    questions.write_text("question,expected_answer,source_url,type\n" + lines, encoding="utf-8-sig")
    seen = []

    def fake_run(command, env, stdout, stderr, cwd, timeout):
        only = [int(n) for n in command[command.index("--only") + 1].split(",")]
        seen.append((only, env["LLM_GEMINI_KEY"], env.get("TERMS_IN_CITED"), "--no-history" in command,
                     command[command.index("--llm") + 1]))
        assert env["COHERE_MIN_INTERVAL"] == "6.7"  # מפתח Cohere מסוג Trial - עד 9 קריאות Rerank לדקה
        with open(command[command.index("--out") + 1], "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row(1)))
            writer.writeheader()
            writer.writerows(row(n) for n in only)
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(B.subprocess, "run", fake_run)
    monkeypatch.setattr(B.config, "GEMINI_KEYS", [(4, "x"), (7, "y")])
    monkeypatch.setattr(B, "ROOT", tmp_path)
    out = tmp_path / "on.csv"
    monkeypatch.setattr("sys.argv", ["eval_batches.py", "--questions", str(questions), "--batch", "2",
                                     "--out", str(out), "--env", "TERMS_IN_CITED=true", "--", "--llm", "gemini-3.6-flash"])
    B.main()
    assert seen == [([1, 2], "4", "true", True, "gemini-3.6-flash"), ([3], "7", "true", True, "gemini-3.6-flash")]
    merged = list(csv.DictReader(open(out, encoding="utf-8-sig")))
    assert [(r["n"], r["key"], r["infra"]) for r in merged] == [("1", "4", ""), ("2", "4", ""), ("3", "7", "")]


def test_cohere_trial_pacing_waits_between_real_calls_only(monkeypatch):
    """מפתח Trial: 10 קריאות Rerank לדקה. COHERE_MIN_INTERVAL - מרווח בין קריאות אמיתיות; מהמטמון - בלי המתנה."""
    import time
    from phoenix import rerank
    monkeypatch.setattr(rerank, "COHERE_MIN_INTERVAL", 0.3)
    monkeypatch.setitem(rerank._cohere_pace, "last", 0.0)
    rerank._cohere_paced()
    start = time.monotonic()
    rerank._cohere_paced()
    assert time.monotonic() - start >= 0.25
    monkeypatch.setattr(rerank, "COHERE_MIN_INTERVAL", 0)
    start = time.monotonic()
    rerank._cohere_paced()
    assert time.monotonic() - start < 0.05  # בלי מרווח (הצ'אט, מפתח בתשלום)

    paced, calls = [], []
    monkeypatch.setattr(rerank, "_cohere_paced", lambda: paced.append(1))
    monkeypatch.setattr(rerank, "_cache", {})
    monkeypatch.setattr(rerank, "_save_cache", lambda: None)
    monkeypatch.setattr(rerank.store, "count_call", lambda service: None)
    cached = rerank._cached("cohere:test", lambda q, docs, n, t: calls.append(1) or [(i, 0.5) for i in range(len(docs))])
    cached("q", ["a", "b"], 2)
    cached("q", ["a", "b"], 2)  # הכול במטמון - בלי קריאה ובלי המתנה
    assert (len(calls), len(paced)) == (1, 1)


def test_the_holdout_needs_final(monkeypatch):
    monkeypatch.setattr("sys.argv", ["eval_batches.py", "--questions", "data/questions_holdout.csv", "--out", "x.csv"])
    with pytest.raises(SystemExit, match="--final"):
        B.main()


def test_ab_compare_skips_rows_with_an_infrastructure_failure():
    before = {"1": {"verdict": "correct", "question": "a"}, "2": {"verdict": "correct", "question": "b"}}
    after = {"1": {"verdict": "wrong", "question": "a", "infra": ""},
             "2": {"verdict": "wrong", "question": "b", "infra": "understand"}}
    r = ab_compare.compare(before, after)
    assert (r["questions"], r["worse"], r["infra"]) == (1, ["1"], ["2"])
