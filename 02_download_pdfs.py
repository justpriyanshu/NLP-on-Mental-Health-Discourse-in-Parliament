"""02 — Download the files of each record. Resumable: re-running skips what is on disk.
File names are pdfs/{record_id}_part{k:02d}.pdf where k is the index in the ledger's
'files' column (03 maps k back to the file kind).

DOWNLOAD_KINDS: which variants to fetch. A record usually has a native-text PDF
(PD_/TQ_) and its scan (ID_/IQ_); 03 wants both (text first, scan as OCR fallback).
Start with ("text",) if disk is tight — the download is incremental."""
import os, time, csv, shutil, requests, pandas as pd
from urllib.parse import unquote
from collections import Counter
from tqdm import tqdm
import config as C
try:
    import truststore; truststore.inject_into_ssl()
except ImportError:
    pass

DOWNLOAD_KINDS = ("text", "image")
MAX_RETRIES, PERMANENT_404_AFTER = 4, 2
PDF_DIR, LOG, TMP = "pdfs", "download_log.csv", "pdfs_tmp"
os.makedirs(PDF_DIR, exist_ok=True); os.makedirs(TMP, exist_ok=True)

def kind(url):
    fn = unquote(url).rsplit("filename=", 1)[-1].strip('"')
    return "image" if fn.startswith(("ID_", "IQ_")) else "text" if fn.startswith(("PD_", "TQ_")) else "other"

df = pd.read_csv("metadata_ledger.csv")
have = set(os.listdir(PDF_DIR))
fails_404 = Counter()
if os.path.exists(LOG):
    old = pd.read_csv(LOG)
    for _, r in old[old.status.astype(str).str.startswith("fail:404")].iterrows():
        fails_404[(int(r.record_id), int(r.part))] += 1
print(f"{len(df)} records; {len(have)} files already present")

sess = requests.Session(); sess.headers.update(C.HEADERS)
new_log = not os.path.exists(LOG)
flog = open(LOG, "a", newline=""); w = csv.writer(flog)
if new_log: w.writerow(["record_id", "part", "kind", "status", "bytes"])
ok = skipped = failed = 0; total = 0
try:
    for _, row in tqdm(df.iterrows(), total=len(df)):
        urls = [u for u in str(row.files).split("|") if u.startswith("http")]
        for k, url in enumerate(urls):
            kd = kind(url)
            if kd not in DOWNLOAD_KINDS: continue
            fn = f"{row.record_id}_part{k:02d}.pdf"
            if fn in have: skipped += 1; continue
            if fails_404[(int(row.record_id), k)] >= PERMANENT_404_AFTER: continue
            tmp = f"{TMP}/{fn}"; status = None
            for attempt in range(MAX_RETRIES):
                try:
                    with sess.get(url, stream=True, timeout=180) as r:
                        if r.status_code == 404: status = "fail:404"; break
                        if r.status_code == 429: time.sleep(15 * (attempt + 1)); continue
                        r.raise_for_status()
                        with open(tmp, "wb") as f:
                            for chunk in r.iter_content(1 << 16): f.write(chunk)
                    size = os.path.getsize(tmp)
                    if size < 1000 or open(tmp, "rb").read(5) != b"%PDF-":
                        raise ValueError(f"not a pdf ({size} bytes)")
                    shutil.move(tmp, f"{PDF_DIR}/{fn}"); have.add(fn); status = "ok"; total += size; break
                except Exception as e:
                    status = f"fail:{str(e)[:80]}"; time.sleep(2 ** attempt)
            if os.path.exists(tmp): os.remove(tmp)
            w.writerow([row.record_id, k, kd, status, os.path.getsize(f"{PDF_DIR}/{fn}") if status == "ok" else 0])
            ok += status == "ok"; failed += status != "ok"
            if (ok + failed) % 25 == 0: flog.flush()
            time.sleep(C.REQUEST_DELAY_S)
finally:
    flog.close()

log = pd.read_csv(LOG); done = log[log.status == "ok"]
need = sum(1 for _, r in df.iterrows() for u in str(r.files).split("|") if u.startswith("http") and kind(u) in DOWNLOAD_KINDS)
print(f"\nok {ok}  already-present {skipped}  failed {failed}  downloaded {total/1e9:.2f} GB this run")
print(f"files wanted {need}  on disk {len(have)}  remaining {need - len(have)}")
if len(done): print(f"mean file {done['bytes'].mean()/1e6:.2f} MB -> projected total {need * done['bytes'].mean()/1e9:.1f} GB")
print(log.status.str[:8].value_counts().to_string())
