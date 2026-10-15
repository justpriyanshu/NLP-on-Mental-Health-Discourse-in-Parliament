"""03 — Hybrid text extraction (replaces 03_ocr.py).

Why: each record on the new portal usually ships TWO files describing the SAME
debate — a native-text PDF (PD_/TQ_) and its scanned counterpart (ID_/IQ_).
v1 OCR'd every part and stitched them, which duplicates every debate's text and
doubles OCR time. The native text layer is exact for English but its Hindi
stream is often font-corrupted (dissertation §2.1.2), so:

  1. pdftotext on the text PDF, page by page, through a quality gate
     (function-word share, junk-char share, gibberish share).
  2. Pages that pass are kept verbatim (source='pdftotext').
  3. Pages that fail are OCR'd from the image PDF at 300 DPI (source='ocr'),
     or the whole image PDF is OCR'd if the text layer is mostly bad.
  4. Duplicate parts (same content twice) are dropped.

Output: raw_text.csv  one row per PAGE with newlines preserved
        extraction_log.csv  per-record decision (report this in the methods chapter)
Resumable via extract_checkpoint.json."""
import os, re, csv, json, difflib, subprocess, sys
from urllib.parse import unquote
import pandas as pd
from tqdm import tqdm
import config as C

CKPT, OUT, LOG = "extract_checkpoint.json", "raw_text.csv", "extraction_log.csv"
FUNC = set("""the of and to in is that for it as was with be by on not he this are or his from at which
but have an had they you were their one all we can her has there been if more when will would who so no
said what up its about into than them only other new some could time these two may then do first any my
now such like our over man me even most made after also did many before must through back years where
much your way well down should because each just those people how too little state good very make
world still own see men work long get here between both life being under never day same another know
while last might us great old year off come since against go came right used take three""".split())
PUA = re.compile(r"[-�\x00-\x08\x0b-\x1f]")
WORD = re.compile(r"[A-Za-z]+")
NOVOWEL = re.compile(r"^[^aeiouAEIOUyY]{5,}$|^[^aeiouAEIOU]+$")

def page_quality(text):
    toks = WORD.findall(text)
    n = len(toks)
    if n < 20:
        return dict(ok=False, reason="blank", n_tokens=n)
    func = sum(t.lower() in FUNC for t in toks) / n
    nonspace = max(1, sum(not c.isspace() for c in text))
    junk = len(PUA.findall(text)) / nonspace
    long_toks = [t for t in toks if len(t) >= 3]
    gib = sum(bool(NOVOWEL.match(t)) for t in long_toks) / max(1, len(long_toks))
    ok = func >= C.TEXT_MIN_FUNCTION_WORD_SHARE and junk <= C.TEXT_MAX_JUNK_SHARE and gib <= 0.10
    return dict(ok=ok, reason="" if ok else "lowq", n_tokens=n, func=round(func, 3),
                junk=round(junk, 3), gib=round(gib, 3))

def pdftotext_pages(pdf):
    out = subprocess.run(["pdftotext", "-enc", "UTF-8", pdf, "-"], capture_output=True, timeout=600)
    pages = out.stdout.decode("utf-8", "replace").split("\f")
    return [p for p in pages[:-1]] if pages and pages[-1].strip() == "" else pages

def n_pages(pdf):
    out = subprocess.run(["pdfinfo", pdf], capture_output=True, text=True, timeout=120).stdout
    m = re.search(r"Pages:\s+(\d+)", out)
    return int(m.group(1)) if m else 0

def ocr_page(pdf, pno, lang):
    from pdf2image import convert_from_path
    import pytesseract
    img = convert_from_path(pdf, dpi=C.OCR_DPI, first_page=pno + 1, last_page=pno + 1)[0]
    return pytesseract.image_to_string(img, lang=lang, config=f"--psm {C.OCR_PSM}")

def norm(t):
    return re.sub(r"\s+", " ", t).strip().lower()

def similar(a, b):
    return difflib.SequenceMatcher(None, norm(a)[:3000], norm(b)[:3000]).quick_ratio()

def kinds_for(row):
    files = [f for f in str(row.files).split("|") if f.startswith("http")]
    if "file_kinds" in row and isinstance(row.file_kinds, str) and row.file_kinds:
        ks = row.file_kinds.split("|")
    else:
        def k(url):
            fn = unquote(url).rsplit("filename=", 1)[-1].strip('"')
            return "image" if fn.startswith(("ID_", "IQ_")) else "text" if fn.startswith(("PD_", "TQ_")) else "other"
        ks = [k(f) for f in files]
    return list(enumerate(ks))

def main():
    ledger = pd.read_csv("metadata_ledger.csv")
    done = set(json.load(open(CKPT))) if os.path.exists(CKPT) else set()
    new = not os.path.exists(OUT)
    fout = open(OUT, "a", newline="", encoding="utf-8"); w = csv.writer(fout, quoting=csv.QUOTE_ALL)
    flog = open(LOG, "a", newline="", encoding="utf-8"); wl = csv.writer(flog)
    if new:
        w.writerow(["record_id", "part", "page", "source", "text"])
        wl.writerow(["record_id", "decision", "text_parts", "image_parts", "pd_pages", "pd_pass_rate",
                     "pages_written", "pages_ocr", "duplicates_dropped", "note"])

    for _, row in tqdm(ledger.iterrows(), total=len(ledger)):
        rid = int(row.record_id)
        if str(rid) in done: continue
        langs = str(row.get("language", "") or "")
        mixed = (langs.strip() == "") or any(not l.strip().lower().startswith("eng") for l in langs.split("|") if l.strip())
        ocr_lang = "eng+hin" if mixed else "eng"
        parts = [(k, kind, f"pdfs/{rid}_part{k:02d}.pdf") for k, kind in kinds_for(row)]
        parts = [(k, kind, fn) for k, kind, fn in parts if os.path.exists(fn) and os.path.getsize(fn) > 0]
        if not parts:
            wl.writerow([rid, "missing", 0, 0, 0, "", 0, 0, 0, "no downloaded parts"]); done.add(str(rid)); continue

        # 'other' parts: decide kind by whether pdftotext yields a usable layer
        texts, images = [], []
        for k, kind, fn in parts:
            if kind == "text": texts.append((k, fn))
            elif kind == "image": images.append((k, fn))
            else:
                pg = pdftotext_pages(fn)
                (texts if pg and sum(page_quality(p)["ok"] for p in pg) / len(pg) >= 0.5 else images).append((k, fn))

        # drop duplicate parts of the same kind (same debate delivered twice)
        dups = 0
        def dedupe(lst, get):
            nonlocal dups
            keep, seen = [], []
            for k, fn in lst:
                t = get(fn)
                if any(similar(t, s) >= C.DUPLICATE_PART_SIMILARITY for s in seen): dups += 1; continue
                seen.append(t); keep.append((k, fn))
            return keep
        texts = dedupe(texts, lambda fn: " ".join(pdftotext_pages(fn)[:3]))
        if len(images) > 1:   # cheap OCR of page 0 only to detect copies
            images = dedupe(images, lambda fn: ocr_page(fn, 0, "eng"))

        written = n_ocr = pd_pages = 0; pass_rate = ""; decision = ""; note = ""
        if texts:
            pd_text = {k: pdftotext_pages(fn) for k, fn in texts}
            quals = {k: [page_quality(p) for p in pgs] for k, pgs in pd_text.items()}
            flat_ok = [q["ok"] for qs in quals.values() for q in qs]
            pd_pages = len(flat_ok); pass_rate = round(sum(flat_ok) / max(1, pd_pages), 3)
            if pass_rate >= C.TEXT_MIN_PAGE_PASS_RATE or not images:
                decision = "pdftotext" + ("+ocr_fallback" if images else "")
                # page-aligned image counterpart: only trusted when page counts match
                fallback = {}
                if len(texts) == len(images):
                    for (k, fn), (ki, fi) in zip(texts, images):
                        if n_pages(fi) == len(pd_text[k]): fallback[k] = fi
                for k, fn in texts:
                    for pno, (page, q) in enumerate(zip(pd_text[k], quals[k])):
                        if q["ok"]:
                            w.writerow([rid, k, pno, "pdftotext", page]); written += 1
                        elif k in fallback:
                            try:
                                w.writerow([rid, k, pno, "ocr", ocr_page(fallback[k], pno, ocr_lang)]); written += 1; n_ocr += 1
                            except Exception as e: note += f"ocr p{pno}:{e};"
                        elif q["reason"] == "blank":
                            pass
                        else:
                            note += f"p{pno} dropped({q['reason']});"
            else:
                decision = "ocr(text layer failed)"
        if decision == "" or decision.startswith("ocr"):
            decision = decision or "ocr"
            for k, fn in images:
                for pno in range(n_pages(fn)):
                    try:
                        w.writerow([rid, k, pno, "ocr", ocr_page(fn, pno, ocr_lang)]); written += 1; n_ocr += 1
                    except Exception as e: note += f"ocr p{pno}:{e};"
        wl.writerow([rid, decision, len(texts), len(images), pd_pages, pass_rate, written, n_ocr, dups, note[:500]])
        fout.flush(); flog.flush(); done.add(str(rid)); json.dump(sorted(done), open(CKPT, "w"))
    fout.close(); flog.close()
    log = pd.read_csv(LOG)
    print(log.decision.value_counts().to_string())
    print(f"pages written: {log.pages_written.sum()}, of which OCR: {log.pages_ocr.sum()}; duplicate parts dropped: {log.duplicates_dropped.sum()}")

if __name__ == "__main__":
    main()
