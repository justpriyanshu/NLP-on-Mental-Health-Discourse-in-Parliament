"""Spot-check: are the debates from the OLD ledger still retrievable on sansad.in?

Usage: python validate_sample.py old_ledger.csv --n 30
old_ledger.csv needs columns: title, date (any parseable format)
"""
import argparse, difflib, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("old"); ap.add_argument("--n", type=int, default=30)
ap.add_argument("--new", default="metadata_ledger.csv")
a = ap.parse_args()

old = pd.read_csv(a.old); new = pd.read_csv(a.new)
old["date"] = pd.to_datetime(old["date"], errors="coerce", dayfirst=True).dt.date
new["date"] = pd.to_datetime(new["date"], errors="coerce").dt.date
sample = old.dropna(subset=["date"]).sample(min(a.n, len(old)), random_state=1)

hits = []
for _, r in sample.iterrows():
    same_day = new[new.date == r.date]
    best, score = None, 0
    for _, s in same_day.iterrows():
        sc = difflib.SequenceMatcher(None, str(r.title).lower(), str(s.title).lower()).ratio()
        if sc > score: best, score = s.title, sc
    hits.append({"old_title": r.title, "date": r.date, "best_match": best,
                 "similarity": round(score, 2), "found": score >= 0.8})

res = pd.DataFrame(hits)
res.to_csv("validation_report.csv", index=False)
print(res[["date", "similarity", "found"]])
print(f"\nRecovered {res.found.sum()}/{len(res)} sampled debates on the new portal.")
