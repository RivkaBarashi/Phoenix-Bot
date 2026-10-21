"""מסמכי PDF מהאתר שעונים ללקוחה -> עמודים רגילים ב-data/pages (עמוד PDF = עמוד, הכתובת עם #page=N), כדי שהחיתוך
והאינדקס יטפלו בהם כמו בכל עמוד. רק המסמכים ב-TITLES נכנסים - נבחרו 21/10 לפי התוכן (data/pdf_catalog.json):
  - טבלאות המועדים לטיפול בתביעות (עמוד אחד לכל מוצר) - "תוך כמה זמן", "עד מתי מגישים";
  - חוברות "כללים לבירור ויישוב תביעות" ו"אמנת שירות" - איך מגישים, אילו מסמכים, מה הזכויות;
  - תקנון קרן הפנסיה המקיפה בתוקף (מהדורת דצמבר 2025).
בחוץ: דוחות כספיים, הצהרות הנהלה, תעודות, תשואות, מדיניות השקעה, ארכיון, מבצעים לסוכנים, מהדורות קודמות של התקנונים
(תשובה מתקנון שכבר לא בתוקף - שגויה), תקנון המשלימה בתוקף (החילוץ שלו לא תקין) וקרנות ותיקות סגורות.

חילוץ: PyMuPDF בלי מיון (סדר המילים נכון ברוב השורות; עם מיון - הפוך). שורה שרוב צמדי המילים שלה מופיעים באתר
דווקא בסדר ההפוך (בעיקר שורות של טבלאות) - סדר המילים בה מתהפך. שורות של תוכן עניינים ("......") ומספרי עמודים - בחוץ.

  .venv\\Scripts\\python -m phoenix.pdf_pages --raw-dir C:/Phoenix/data/pdf_raw
"""
import argparse
import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote

import pymupdf

from .config import ROOT

TITLES = {  # תחילת שם הקובץ -> השם שהבוט מציג כמקור
    # מועדים לטיפול בתביעות
    "tvihut_nituchim": "טבלת מועדים לטיפול בתביעות - ביטוח ניתוחים",
    "tviot-moadim-ovdan-kosher-avoda": "טבלת מועדים לטיפול בתביעות - אובדן כושר עבודה",
    "tviut dira": "טבלת מועדים לטיפול בתביעות - ביטוח דירה",
    "tviut_bituah_hayim": "טבלת מועדים לטיפול בתביעות - ביטוח חיים",
    "tviut_kisuhim_abulatarim": "טבלת מועדים לטיפול בתביעות - כיסויים אמבולטוריים",
    "tviut_mahalut_kashut": "טבלת מועדים לטיפול בתביעות - מחלות קשות",
    "tviut_nesihut_lachul": "טבלת מועדים לטיפול בתביעות - ביטוח נסיעות לחו\"ל",
    "tviut_nituch_bechul": "טבלת מועדים לטיפול בתביעות - ניתוח בחו\"ל",
    "tviut_pensya_vegmel": "טבלת מועדים לטיפול בתביעות - פנסיה וגמל",
    "tviut_siud": "טבלת מועדים לטיפול בתביעות - ביטוח סיעודי",
    # חוברות כללים לבירור ויישוב תביעות
    "300104345_sheerim-tviot-roles": "חוברת כללים לתביעות שאירים - פנסיה",
    "300305153": "חוברת כללים לבירור ויישוב תביעות",
    "ambolatoriim_hoveret": "חוברת כללים לתביעות - כיסויים אמבולטוריים",
    "dira_hoveret": "חוברת כללים לתביעות - ביטוח דירה",
    "hayim_hoveret": "חוברת כללים לתביעות - ביטוח חיים",
    "hool_hoveret": "חוברת כללים לתביעות - ביטוח נסיעות לחו\"ל",
    "hoveret-disabiliy-pension": "חוברת כללים לתביעות נכות - פנסיה",
    "illnes-klalim": "חוברת כללים לתביעות - מחלות קשות",
    "makif_dira": "חוברת כללים לתביעות - ביטוח דירה מקיף",
    "nituhim_hoveret": "חוברת כללים לתביעות - ניתוחים",
    "rechev_chova_hovert": "חוברת כללים לתביעות - ביטוח רכב חובה",
    "rehevrehush_hoveret": "חוברת כללים לתביעות - ביטוח רכב רכוש",
    "siud_hoveret": "חוברת כללים לתביעות - ביטוח סיעודי",
    "teunut_hoveret": "חוברת כללים לתביעות - תאונות אישיות",
    "tipulim_hoveret": "חוברת כללים לתביעות - טיפולים",
    "trufut_hoveret": "חוברת כללים לתביעות - תרופות",
    "tviot_new_makifa_site": "חוברת כללים לתביעות - פנסיה מקיפה",
    "yeshuv-tviaa-ovdan-kosher-avoda-nagish": "חוברת כללים לתביעות - אובדן כושר עבודה",
    # אמנות שירות ואיגרות מידע לתביעות
    "amana-disabiliy-pension": "אמנת שירות לתביעות - נכות בפנסיה",
    "amana-kosher-avoda-nagish": "אמנת שירות לתביעות - אובדן כושר עבודה",
    "amana-machlifey-nituach": "אמנת שירות לתביעות - מחליפי ניתוח",
    "amana_ambulator": "אמנת שירות לתביעות - כיסויים אמבולטוריים",
    "bemikre_ptira": "אמנת שירות - תביעות במקרה פטירה",
    "bituach_dira": "אמנת שירות לתביעות - ביטוח דירה",
    "bituach_letrufot_shelo_klulot_basal": "אמנת שירות לתביעות - תרופות שלא בסל",
    "bituach_nesiyot_lechol": "אמנת שירות לתביעות - ביטוח נסיעות לחו\"ל",
    "bituach_rechev_rechush": "אמנת שירות לתביעות - ביטוח רכב רכוש",
    "bituach_siudi_amana_from010919": "אמנת שירות לתביעות - ביטוח סיעודי",
    "claims_igeret_mc_siudi_brochure": "איגרת מידע לתביעות - מכבי סיעודי",
    "igeret-hova": "איגרת מידע לתביעות - ביטוח רכב חובה",
    "igeret_nituhim_behul_hashtalut_vetipolim_meyooha": "איגרת מידע לתביעות - ניתוחים בחו\"ל, השתלות וטיפולים מיוחדים",
    "igeret_rehev_rhosh_zad_gimel": "איגרת מידע לתביעות - ביטוח רכב רכוש וצד ג'",
    "makif_dira_tzad": "אמנת שירות לתביעות - ביטוח דירה וצד ג'",
    "pensia_cllalit_sheerim": "אמנת שירות - תביעות שאירים בפנסיה",
    "pensya_nechut_makifa": "אמנת שירות - תביעות נכות בפנסיה מקיפה",
    "sheerim-mashlima-makifa": "אמנת שירות - תביעות שאירים בפנסיה מקיפה ומשלימה",
    "tehunot_hishiyot": "אמנת שירות לתביעות - תאונות אישיות",
    "tviot_nechut_zmita": "אמנת שירות - תביעות נכות",
    "ביטוח מחלות קשות": "אמנת שירות לתביעות - מחלות קשות",
    # תקנון בתוקף
    "1749579393_תקנון_הפניקס_פנסיה_מקיפה_דצמבר_2025": "תקנון קרן הפנסיה הפניקס פנסיה מקיפה (מהדורת דצמבר 2025)",
}
HEB = re.compile(r"[\u05d0-\u05ea]{2,}")
TOC = re.compile(r"\.{5,}|…{3,}")
PAGE_NUMBER = re.compile(r"^\s*(?:עמוד\s*)?\d{1,3}\s*(?:מתוך\s*\d{1,3})?\s*$")


def title_for(url: str) -> str | None:
    name = unquote(url.rsplit("/", 1)[-1])
    starts = sorted(TITLES, key=len, reverse=True)  # השם הארוך קודם: makif_dira_tzad לפני makif_dira
    return next((TITLES[start] for start in starts if name.startswith(start)), None)


def site_bigrams() -> set[tuple[str, str]]:
    pairs = set()
    for line in open(ROOT / "data" / "chunks.jsonl", encoding="utf-8"):
        w = HEB.findall(json.loads(line)["text"])
        pairs.update(zip(w, w[1:]))
    return pairs


GLUED = re.compile(r"(?<=[א-ת])(?=\d)|(?<=\d)(?=[א-ת])")  # "גיל60", "ימים30" - בלי רווח


def fix_line(line: str, pairs: set) -> str:
    """רווח בין אותיות למספר שהודבקו בחילוץ ("גיל60" -> "גיל 60"; בלי להחליף סדר - הכיוון תלוי בהקשר), ושורה שרוב
    צמדי המילים העבריות שלה מופיעים באתר בסדר ההפוך - סדר המילים בה מתהפך (שורות טבלה ב-PDF)."""
    line = GLUED.sub(" ", line)
    words = HEB.findall(line)
    forward = sum(p in pairs for p in zip(words, words[1:]))
    backward = sum(p in pairs for p in zip(words[::-1], words[::-1][1:]))
    if backward >= 2 and backward > 2 * forward:
        return " ".join(line.split()[::-1])
    return line


def pages_of(path: Path, pairs: set):
    doc = pymupdf.open(path)
    for number, page in enumerate(doc, 1):
        lines = []
        for raw in page.get_text("text").splitlines():
            line = " ".join(raw.split())
            if not line or TOC.search(line) or PAGE_NUMBER.match(line):
                continue
            lines.append(fix_line(line, pairs))
        yield number, "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=str(ROOT / "data" / "pdf_raw"))
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    manifest = json.loads((ROOT / "data" / "pdf_manifest.json").read_text(encoding="utf-8"))
    pairs = site_bigrams()
    out = ROOT / "data" / "pages"
    for old in out.glob("pdf_*.json"):  # בונים מחדש בכל הרצה
        old.unlink()
    documents = pages = 0
    for url, info in manifest.items():
        title = title_for(url)
        if not title or info.get("status") != "ok":
            continue
        documents += 1
        stem = re.sub(r"[^0-9A-Za-z\u05d0-\u05ea_-]+", "_", unquote(url.rsplit("/", 1)[-1]).rsplit(".", 1)[0])[:60]
        for number, text in pages_of(Path(args.raw_dir) / info["file"], pairs):
            if len(HEB.findall(text)) < 15:  # עמוד כמעט ריק (שער, תמונה)
                continue
            record = {"url": f"{url}#page={number}", "requested_url": url, "title": f"{title} (PDF, עמוד {number})",
                      "text": text, "fetched_at": info.get("fetched_at", ""), "via": "pdf"}
            (out / f"pdf_{stem}_p{number:03d}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1),
                                                                encoding="utf-8")
            pages += 1
    missing = [start for start in TITLES if not any(title_for(u) == TITLES[start] for u in manifest)]
    print(f"מסמכים: {documents} מתוך {len(TITLES)} ברשימה | עמודים שנכתבו: {pages}" +
          (f" | לא נמצאו: {missing}" if missing else ""))


if __name__ == "__main__":
    main()
