"""Metadata ledger from integration.rajyasabha.digital /fetch/all.
Resumable: progress saved in scrape_state.json; safe to Ctrl-C and rerun."""
import json, os, sys, time
import pandas as pd, requests
from tqdm import tqdm
import config as C

STATE, OUT = "scrape_state.json", "metadata_ledger.csv"
s = requests.Session(); s.headers.update(C.HEADERS)
base = {"rows": C.ROWS, "order": "all_desc", "collectionId": C.COLLECTION,
        "bucketFields": "year,type,sessionNo,title,date,debateTitleSubject,members"}

def page(start, extra):
    for attempt in range(6):
        try:
            r = s.get(C.LIST_URL, params={**base, "start": start, **extra}, timeout=120)
            if r.status_code == 429: time.sleep(15 * (attempt + 1)); continue
            r.raise_for_status(); return r.json()
        except (requests.RequestException, ValueError) as e:
            print(f"  retry {attempt+1}: {e}", file=sys.stderr); time.sleep(2 ** attempt)
    raise SystemExit("API unreachable after retries")

def kind(url):
    """ID_/IQ_ = scanned image PDF, PD_/TQ_ = native text PDF (see 03_extract_text.py)."""
    from urllib.parse import unquote
    fn = unquote(url).rsplit("filename=", 1)[-1].strip('"')
    return "image" if fn.startswith(("ID_", "IQ_")) else "text" if fn.startswith(("PD_", "TQ_")) else "other"

def flat(rec):
    files = rec.get("files") or []
    return {"record_id": rec.get("resourceId"), "handle": rec.get("handle"),
            "file_kinds": "|".join(kind(f) for f in files),
            "title": rec.get("title"), "date": rec.get("date"), "year": rec.get("year"),
            "type": rec.get("type"), "session": rec.get("sessionNo"),
            "subject": rec.get("debateTitleSubject"), "pages": rec.get("pageNoFromTo"),
            "members": "|".join(rec.get("members") or []),
            "language": "|".join(rec.get("language") or []),
            "pdf_text": next((f for f in files if "PD_" in f or "TQ_" in f), None),
            "pdf_image": next((f for f in files if "ID_" in f or "IQ_" in f), None),
            "files": "|".join(files)}

state = json.load(open(STATE)) if os.path.exists(STATE) else {"done": [], "rows": []}
rows = state["rows"]

def save():
    json.dump(state, open(STATE, "w"))

if C.FILTER_PARAMS:                                   # ---- filtered mode
    for y in tqdm(range(C.FROM_YEAR, C.TO_YEAR + 1), desc="years"):
        if y in state["done"]: continue
        extra = {k: v.replace("{year}", str(y)) for k, v in C.FILTER_PARAMS.items()}
        start = 0
        while True:
            j = page(start, extra); recs = j.get("records", [])
            rows += [flat(r) for r in recs]
            if len(recs) < C.ROWS: break
            start += C.ROWS; time.sleep(C.REQUEST_DELAY_S)
        state["done"].append(y); save()
else:                                                  # ---- walk-all fallback
    start = state.get("start", 0); total = int(page(0, {}).get("rowsCount", 0))
    pbar = tqdm(total=total, initial=start, desc="records (newest first)")
    while True:
        j = page(start, {}); recs = j.get("records", [])
        if not recs: break
        keep = [flat(r) for r in recs if r.get("type") == C.PART2
                and str(r.get("year", "")).isdigit() and C.FROM_YEAR <= int(r["year"]) <= C.TO_YEAR]
        rows += keep
        years = [int(r["year"]) for r in recs if str(r.get("year", "")).isdigit()]
        start += len(recs); state["start"] = start; pbar.update(len(recs))
        if start % (C.ROWS * 10) == 0: save()
        if years and max(years) < C.FROM_YEAR: break   # sorted desc: nothing older is in range
        time.sleep(C.REQUEST_DELAY_S)
    save()

df = pd.DataFrame(rows)
df = df[df.type == C.PART2].drop_duplicates("record_id")
df["year"] = df.year.astype(int); df = df[(df.year >= C.FROM_YEAR) & (df.year <= C.TO_YEAR)]
df["pdf_link"] = df.pdf_image.fillna(df.pdf_text).fillna(df.files.str.split("|").str[0])
df.to_csv(OUT, index=False)
print(f"\n{len(df)} Part 2 debates {C.FROM_YEAR}-{C.TO_YEAR} -> {OUT}")
print(df.groupby("year").size().to_string())