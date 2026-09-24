"""
Download the input books into data/books/.

  python scripts/get_books.py

Sabha Parva (Ganguli translation, public domain):
  1. ready-made PDF (sacred-texts text, typeset):  my.eng.utah.edu mirror
  2. fallback: Project Gutenberg #7965 plain text -> rendered to a plain PDF here
     (same font for everything; ingestion then finds "SECTION ..." headings by its
     generic upper-case-line rule).
The Old Man and the Sea is still under copyright: copy your own PDF into data/books/.
"""
import sys
import urllib.request
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "books"
PDF_URL = "https://my.eng.utah.edu/~banerjee/Ebooks/Maha02_SabhaParva.pdf"
TXT_URLS = ["https://www.gutenberg.org/cache/epub/7965/pg7965.txt",
            "https://www.gutenberg.org/files/7965/7965.txt"]
UA = {"User-Agent": "Mozilla/5.0 (Kaalkram research; book download)"}


def _get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return r.read()


def _has_text(pdf: Path) -> bool:
    try:
        with fitz.open(pdf) as d:
            return sum(len(p.get_text()) for p in d) > 50_000
    except Exception:
        return False


def _txt_to_pdf(text: str, dest: Path) -> None:
    start, end = text.find("*** START"), text.find("*** END")
    if start != -1:
        text = text[text.find("\n", start) + 1:]
    if end != -1:
        text = text[:text.find("*** END")]
    paras = [p.replace("\n", " ").strip() for p in text.replace("\r", "").split("\n\n")]
    doc = fitz.open()
    rect = fitz.Rect(50, 50, 545, 792)
    buf = ""
    for p in paras:
        if not p:
            continue
        cand = buf + p + "\n\n"
        page_text = cand
        if len(page_text) > 3200:
            page = doc.new_page()
            page.insert_textbox(rect, buf, fontsize=10, fontname="helv")
            buf = p + "\n\n"
        else:
            buf = cand
    if buf:
        page = doc.new_page()
        page.insert_textbox(rect, buf, fontsize=10, fontname="helv")
    doc.save(dest)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / "sabha_parva_ganguli.pdf"
    if dest.exists() and _has_text(dest):
        print(f"already present: {dest}")
        return 0
    try:
        dest.write_bytes(_get(PDF_URL))
        if _has_text(dest):
            print(f"downloaded PDF -> {dest}")
            return 0
        print("PDF downloaded but has too little text; trying Gutenberg")
    except Exception as exc:
        print(f"PDF download failed ({exc}); trying Gutenberg")
    for url in TXT_URLS:
        try:
            text = _get(url).decode("utf-8", errors="replace")
            (OUT / "sabha_parva_ganguli.txt").write_text(text, encoding="utf-8")
            _txt_to_pdf(text, dest)
            if _has_text(dest):
                print(f"rendered Gutenberg text -> {dest}")
                return 0
        except Exception as exc:
            print(f"{url} failed: {exc}")
    print("Could not fetch the book. Download manually (see README) into data/books/.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
