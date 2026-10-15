"""OCR quality audit: character/word error rate on a period-stratified page sample.
The dissertation states OCR noise is worse for early scans but never measures it.

  python eval_ocr.py --sample 40      -> ocr_eval/<id>_p<page>.png + .hyp.txt + empty .ref.txt
  (transcribe each .ref.txt by hand — the PNG is the ground truth)
  python eval_ocr.py --score          -> ocr_eval_report.csv (CER/WER by period, and overall)

Only OCR'd pages are sampled (source == 'ocr'); pdftotext pages are exact by construction."""
import argparse, os, re, pandas as pd, numpy as np
import config as C

ap = argparse.ArgumentParser()
ap.add_argument("--sample", type=int); ap.add_argument("--score", action="store_true")
a = ap.parse_args()
os.makedirs("ocr_eval", exist_ok=True)

def period_of(y):
    for name, (lo, hi) in C.PERIODS.items():
        if lo <= y <= hi: return name
    return "out"

if a.sample:
    from pdf2image import convert_from_path
    raw = pd.read_csv("raw_text.csv"); led = pd.read_csv("metadata_ledger.csv")[["record_id", "year"]]
    ocr = raw[raw.source == "ocr"].merge(led, on="record_id")
    ocr["period"] = ocr.year.astype(int).map(period_of)
    per = max(1, a.sample // ocr.period.nunique())
    samp = pd.concat([g.sample(min(per, len(g)), random_state=0) for _, g in ocr.groupby("period")])
    for _, r in samp.iterrows():
        stem = f"ocr_eval/{r.record_id}_part{int(r.part):02d}_p{int(r.page):03d}"
        pdf = f"pdfs/{r.record_id}_part{int(r.part):02d}.pdf"
        convert_from_path(pdf, dpi=C.OCR_DPI, first_page=int(r.page) + 1, last_page=int(r.page) + 1)[0].save(stem + ".png")
        open(stem + ".hyp.txt", "w", encoding="utf-8").write(str(r.text))
        if not os.path.exists(stem + ".ref.txt"): open(stem + ".ref.txt", "w").write("")
    samp[["record_id", "part", "page", "year", "period"]].to_csv("ocr_eval/sample_index.csv", index=False)
    print(f"{len(samp)} pages written to ocr_eval/. Transcribe the .ref.txt files, then --score.")

if a.score:
    import jiwer
    idx = pd.read_csv("ocr_eval/sample_index.csv")
    norm = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemoveMultipleSpaces(), jiwer.Strip(),
                          jiwer.RemovePunctuation(), jiwer.ReduceToListOfListOfWords()])
    rows = []
    for _, r in idx.iterrows():
        stem = f"ocr_eval/{r.record_id}_part{int(r.part):02d}_p{int(r.page):03d}"
        ref = open(stem + ".ref.txt", encoding="utf-8").read().strip()
        hyp = open(stem + ".hyp.txt", encoding="utf-8").read().strip()
        if not ref: continue
        rows.append({"record_id": r.record_id, "page": r.page, "year": r.year, "period": r.period,
                     "cer": jiwer.cer(ref, hyp), "wer": jiwer.wer(ref, hyp, reference_transform=norm, hypothesis_transform=norm)})
    res = pd.DataFrame(rows)
    if res.empty: raise SystemExit("no transcribed .ref.txt files yet")
    rep = res.groupby("period")[["cer", "wer"]].agg(["mean", "std", "count"]).round(3)
    rep.loc["ALL"] = [res.cer.mean(), res.cer.std(), len(res), res.wer.mean(), res.wer.std(), len(res)]
    rep.to_csv("ocr_eval_report.csv"); print(rep.to_string())
    # does OCR quality trend with year?  (Spearman)
    from scipy.stats import spearmanr
    rho, p = spearmanr(res.year, res.cer)
    print(f"\nSpearman(year, CER) = {rho:.3f} (p={p:.3f}) -> negative = older scans noisier")
