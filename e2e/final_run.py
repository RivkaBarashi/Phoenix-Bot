"""The final run, step by step (started by Windows Task Scheduler, task PhoenixFinalRun, 21/10 10:00; runs by hand too):

  1. waits for the owner's approval - the file logs/APPROVE_FINAL_RUN (created when she says "approved"); no file, no run
  2. opens the chat in a real browser (Streamlit first, Render if it does not come up)
  3. the smoke gate: 5 real questions, 5/5 and p90 of the bot <= 20 s - up to --smoke-attempts attempts, alternating between
     the two sites, --smoke-gap-min minutes after a failed smoke (a site that does not come up within 4 minutes: the next site
     right away, no wait); if none passes: BLOCKED (nothing else runs, the holdout is not touched)
  4. the main file, then the holdout, each question in a new chat on the public link (e2e_cloud.py)
  5. the scoring of both, like eval_answers (judge_cloud.py)
Everything goes to --out-dir: status.txt (one line per step), run.log, *_answers.csv, *_judged.csv, *_summary.txt.
The holdout file is read only by e2e_cloud.py during step 4 - this script never opens it.
"""
import argparse
import datetime as dt
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
SITES = {"streamlit": "https://phoenixbot.streamlit.app", "render": "https://phoenixbot-tik7.onrender.com"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "logs" / "final"))
    ap.add_argument("--main", default=str(ROOT / "data" / "questions.csv"))
    ap.add_argument("--holdout", default=str(ROOT / "data" / "questions_holdout.csv"))
    ap.add_argument("--approval-file", default=str(ROOT / "logs" / "APPROVE_FINAL_RUN"))
    ap.add_argument("--wait-hours", type=float, default=12.0)
    ap.add_argument("--smoke-attempts", type=int, default=6)
    ap.add_argument("--smoke-gap-min", type=float, default=10.0)
    ap.add_argument("--smoke-csv", help="לבדיקת התהליך: שאלות smoke אחרות (question, expected), בלי קריאות למודלים")
    ap.add_argument("--sites", default="streamlit,render", help="לפי הסדר; או כתובת מלאה (לבדיקה מקומית)")
    ap.add_argument("--browser", default="msedge")
    ap.add_argument("--min-gap", default="8", help="שניות לפחות בין שאלה לשאלה בקובץ הראשי ובסט הנסתר (קצב של אדם)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    status_file, log_file = out / "status.txt", out / "run.log"

    def status(text: str):
        line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {text}"
        print(line, flush=True)
        with open(status_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def run(*cmd: str) -> int:
        with open(log_file, "a", encoding="utf-8") as log:
            log.write(f"\n$ {' '.join(cmd)}\n")
            log.flush()
            return subprocess.run([PY, *cmd], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True,
                                  env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}).returncode

    def last_line(path: Path) -> str:
        lines = [line for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
        return lines[-1] if lines else ""

    status(f"start | main={args.main} | holdout={args.holdout} | approval={args.approval_file}")

    # 1. the owner's approval
    approval, waited = Path(args.approval_file), 0.0
    while not approval.exists():
        if waited >= args.wait_hours * 3600:
            status(f"NOT APPROVED after {args.wait_hours:g} hours - nothing ran")
            return 4
        if waited % 600 == 0:
            status("waiting for the owner's approval (create the file above, or tell Claude 'approved')")
        time.sleep(30)
        waited += 30
    status("approved")

    # 2 + 3. the site, and the smoke gate
    sites = [SITES.get(s, s) for s in args.sites.split(",")]
    chosen = None
    for attempt in range(args.smoke_attempts):
        site = sites[attempt % len(sites)]
        if run("e2e/e2e_cloud.py", "--url", site, "--wake", "--browser", args.browser) != 0:
            status(f"smoke attempt {attempt + 1}: {site} did not come up - the next site right away")
            continue  # a site that did not come up is not a smoke failure: no 10-minute wait
        else:
            smoke_csv = ["--smoke-csv", args.smoke_csv] if args.smoke_csv else []
            code = run("e2e/e2e_cloud.py", "--url", site, "--smoke", "--browser", args.browser, *smoke_csv,
                       "--out", str(out / f"smoke_{attempt + 1}.csv"))
            status(f"smoke attempt {attempt + 1} on {site}: {last_line(log_file)}")
            if code == 0:
                chosen = site
                break
        if attempt + 1 < args.smoke_attempts:
            time.sleep(args.smoke_gap_min * 60)
    if not chosen:
        status("BLOCKED: the smoke gate did not pass - the final run did not start, the holdout was not touched")
        return 3
    status(f"smoke passed on {chosen} - the final run starts")

    # 4. main file, then the holdout
    for name, path in (("main", args.main), ("holdout", args.holdout)):
        answers = out / f"{name}_answers.csv"
        code = run("e2e/e2e_cloud.py", "--url", chosen, "--questions", path, "--out", str(answers), "--browser", args.browser,
                   "--min-gap", args.min_gap)
        status(f"{name} done (exit {code}): {last_line(log_file)}")

    # 5. the scoring
    for name, path in (("main", args.main), ("holdout", args.holdout)):
        summary = out / f"{name}_summary.txt"
        with open(summary, "w", encoding="utf-8") as f:
            code = subprocess.run([PY, "e2e/judge_cloud.py", "--questions", path, "--answers", str(out / f"{name}_answers.csv"),
                                   "--out", str(out / f"{name}_judged.csv")], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                                  text=True, env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}).returncode
        status(f"{name} scored (exit {code}) -> {summary.name}")
    status("end")
    return 0


if __name__ == "__main__":
    sys.exit(main())
