"""מסמכי ה-PDF שהאתר מקשר אליהם (תנאי פוליסות, מדריכים, טפסים): הורדה מנומסת - בקשה אחת בשנייה לכל היותר, רק
מה ש-robots.txt מתיר - ל-data/pdf_raw (לא ב-Git), עם רשימה ב-data/pdf_manifest.json: כתובת, קובץ, גודל, תאריך הורדה.
אפשר להפסיק ולהמשיך: קובץ שכבר ירד - לא יורד שוב.

  .venv\\Scripts\\python -m phoenix.pdf_fetch data/pdf_links.txt
"""
import hashlib
import json
import sys
import time
import urllib.robotparser
from datetime import datetime, timezone
from pathlib import Path

import truststore

truststore.inject_into_ssl()  # Netspark: מאגר התעודות של המערכת, האימות נשאר דלוק
import httpx  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "pdf_raw"
MANIFEST = ROOT / "data" / "pdf_manifest.json"
AGENT = "PhoenixSiteBot/1.0 (hackathon RAG; polite: 1 request/second)"
GAP = 1.0


def file_for(url: str) -> Path:
    return RAW / f"{hashlib.md5(url.encode('utf-8')).hexdigest()[:16]}.pdf"


def main(links_path: str):
    RAW.mkdir(parents=True, exist_ok=True)
    urls = [u.strip() for u in Path(links_path).read_text(encoding="utf-8").splitlines() if u.strip()]
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    robots = urllib.robotparser.RobotFileParser()
    with httpx.Client(headers={"User-Agent": AGENT}, timeout=60, follow_redirects=True) as client:
        robots.parse(client.get("https://www.fnx.co.il/robots.txt").text.splitlines())
        done = skipped = failed = 0
        for i, url in enumerate(urls, 1):
            path = file_for(url)
            if path.exists() and path.stat().st_size > 0 and url in manifest:
                continue
            if not robots.can_fetch(AGENT, url):
                manifest[url] = {"status": "robots_disallow"}
                skipped += 1
                continue
            start = time.perf_counter()
            try:
                r = client.get(url)
                ok = r.status_code == 200 and r.content[:5] == b"%PDF-"
                if ok:
                    path.write_bytes(r.content)
                manifest[url] = {"status": r.status_code if not ok else "ok", "file": path.name if ok else "",
                                 "bytes": len(r.content), "final_url": str(r.url),
                                 "last_modified": r.headers.get("last-modified", ""),
                                 "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
                done += ok
                failed += not ok
            except Exception as exc:  # noqa: BLE001 - נרשם ברשימה, ממשיכים
                manifest[url] = {"status": f"error {type(exc).__name__}"}
                failed += 1
            if i % 25 == 0:
                MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
                print(f"{i}/{len(urls)} | ok {done} | failed {failed} | robots {skipped}", flush=True)
            time.sleep(max(0.0, GAP - (time.perf_counter() - start)))
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"done: ok {done} | failed {failed} | robots {skipped} | total in manifest {len(manifest)}")


if __name__ == "__main__":
    main(sys.argv[1])
