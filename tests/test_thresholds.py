"""ספי ה-Rerank המכוילים (docs/before-paid-tests.md, סעיף 1): הבחירה הסטטיסטית, הקובץ עם הגרסה, ו"סף שהתיישן"."""
import json

import pytest

import calibrate_thresholds as calibrate
from phoenix import answer_cache, config, stats, thresholds


# ---------------------------------------------------------------- הגבול העליון (Clopper-Pearson)
@pytest.mark.parametrize("k, n, expected", [(0, 52, 5.6), (1, 52, 8.8), (28, 52, 65.8), (0, 86, 3.4)])
def test_clopper_pearson_upper_bound(k, n, expected):
    assert round(stats.clopper_pearson_upper(k, n) * 100, 1) == expected


def test_upper_bound_edges():
    assert stats.clopper_pearson_upper(0, 0) == 1.0 and stats.clopper_pearson_upper(5, 5) == 1.0
    assert stats.clopper_pearson_upper(1, 100) < stats.clopper_pearson_upper(2, 100)  # יותר טעויות - גבול גבוה יותר


# ---------------------------------------------------------------- בחירת הסף
def test_choose_the_highest_threshold_whose_upper_bound_meets_alpha():
    """הריצה הסופית 21/10: רחפן צילום 0.785, ואחריו 0.88... - ב-alpha 0.10 נבחר 0.88 (1 מתוך 52, גבול 8.8%);
    ב-0.89 כבר 2 מתוך 52 (גבול 11.7%) - עוצרים."""
    scores = [0.785, 0.88, 0.896, 0.908] + [0.95] * 48
    assert calibrate.choose(scores, 0.10)[:2] == (0.88, 1)
    assert calibrate.choose(scores, 0.06)[:2] == (0.78, 0)  # 0 חסומות (5.6%) - רק מתחת לציון הנמוך


def test_choose_stops_at_the_first_threshold_that_fails():
    """fixed sequence: מהנמוך לגבוה, עוצרים בראשון שנכשל - גם אם סף גבוה יותר "היה עובר" במקרה."""
    grid = [0.5, 0.6, 0.7, 0.8]
    scores = [0.55] * 3 + [0.9] * 7  # ב-0.6 כבר 3 מתוך 10
    assert calibrate.choose(scores, 0.30, grid)[0] == 0.5


def test_choose_returns_the_lowest_when_nothing_meets_alpha():
    threshold, blocked, upper = calibrate.choose([0.1, 0.2, 0.9], 0.01, [0.3, 0.4])
    assert threshold == 0.3 and blocked == 2 and upper > 0.01  # הגבול מראה שהדרישה לא התקיימה


def test_the_holdout_is_refused(tmp_path):
    holdout = tmp_path / "questions_holdout.csv"
    holdout.write_text("question,expected_answer,source_url,type\nq,a,,answerable\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        calibrate.load_rows([holdout])


def test_only_answerable_and_not_on_site_are_calibrated(tmp_path):
    path = tmp_path / "set.csv"
    path.write_text("question,expected_answer,source_url,type\na,,,answerable\nb,,,personal\nc,,,not_on_site\n"
                    "d,,,injection\n", encoding="utf-8-sig")
    assert [r["question"] for r in calibrate.load_rows([path])] == ["a", "c"]  # אישית - לא נחסמת בסף


# ---------------------------------------------------------------- הקובץ, הגרסה וההתיישנות
@pytest.fixture
def thresholds_file(tmp_path, monkeypatch):
    path = tmp_path / "thresholds.json"
    monkeypatch.setattr(config, "THRESHOLDS_FILE", path)
    return path


def entry(**changes) -> dict:
    return {"threshold": 0.8, "model": dict(config.RERANK_CHAIN)["cohere"],
            "index_version": answer_cache.index_version()} | changes


def test_the_calibration_file_wins_over_the_environment(thresholds_file):
    thresholds_file.write_text(json.dumps({"cohere": entry(threshold=0.77), "broken": "x"}), encoding="utf-8")
    assert config._calibrated_thresholds() == {"cohere": 0.77}
    thresholds_file.write_text("לא JSON", encoding="utf-8")
    assert config._calibrated_thresholds() == {}  # קובץ פגום - ערכי ברירת המחדל, בלי לקרוס


@pytest.mark.parametrize("changes, expected", [({}, "calibrated"), ({"model": "rerank-v9"}, "stale:model"),
                                               ({"index_version": "old"}, "stale:index")])
def test_status_knows_when_the_threshold_is_stale(thresholds_file, changes, expected):
    thresholds_file.write_text(json.dumps({"cohere": entry(**changes)}), encoding="utf-8")
    assert thresholds.status("cohere") == expected
    assert thresholds.status("voyage") == "uncalibrated"


def test_warnings_for_a_stale_threshold_and_a_conflicting_environment(thresholds_file, monkeypatch):
    thresholds_file.write_text(json.dumps({"cohere": entry(index_version="old")}), encoding="utf-8")
    monkeypatch.setenv("RERANK_THRESHOLD", "0.82")
    lines = thresholds.warnings()
    assert any("stale:index" in line for line in lines) and any("RERANK_THRESHOLD=0.82" in line for line in lines)
    monkeypatch.setenv("RERANK_THRESHOLD", "0.8")
    thresholds_file.write_text(json.dumps({"cohere": entry()}), encoding="utf-8")
    assert thresholds.warnings() == []


def test_no_file_means_the_old_behaviour(thresholds_file):
    assert thresholds.load() == {} and thresholds.warnings() == []
    assert "לא מכויל" in thresholds.summary()


def test_report_lists_questions_blocked_just_below_the_threshold(monkeypatch):
    import report
    monkeypatch.setitem(config.RERANK_THRESHOLDS, "cohere", 0.82)

    def record(question, score, reason="below_threshold", reranker="cohere"):
        return {"question": question, "top_score": score, "reason": reason, "reranker": reranker}
    entries = [record("רחוקה", 0.70), record("קרובה", 0.80), record("הכי קרובה", 0.815),
               record("עוברת היום", 0.83),  # נחסמה כשהסף היה גבוה יותר
               record("נושא אחר", 0.81, reason="below_threshold_off_topic"), record("בלי ציון", None),
               record("בלי מדרג מכויל", 0.81, reranker="none"), record("עם שם מודל", 0.79, reranker="cohere:v3.5")]
    near = report.near_threshold(entries)
    assert [e["question"] for e, _ in near] == ["עוברת היום", "הכי קרובה", "קרובה", "עם שם מודל"]
    assert all(threshold == 0.82 for _, threshold in near)


def test_a_committed_threshold_is_never_stale():
    """השומר: אם יש כיול בקובץ שב-Git, הוא חייב להתאים לאינדקס ולמודל הדירוג הנוכחיים. אינדקס שנבנה מחדש בלי כיול -
    הבדיקה נכשלת, וסף ישן לא עולה לאוויר. (לפני הכיול הראשון אין קובץ - הבדיקה מדולגת.)"""
    if not config.THRESHOLDS_FILE.exists():
        pytest.skip("עוד אין data/thresholds.json - להריץ calibrate_thresholds.py --write")
    states = {ranker: thresholds.status(ranker) for ranker in thresholds.load()}
    assert all(s == "calibrated" for s in states.values()), f"{states} - להריץ calibrate_thresholds.py --write"
