"""Find the filter syntax for /fetch/all and dump per-year Part 2 counts.
Run once:  python probe_api.py"""
import json, time, requests
import config as C

s = requests.Session(); s.headers.update(C.HEADERS)
base = {"start": 0, "rows": 5, "bucketFields": "year,type", "order": "all_desc", "collectionId": C.COLLECTION}

def count(extra):
    r = s.get(C.LIST_URL, params={**base, **extra}, timeout=60)
    try:
        j = r.json(); return int(j.get("rowsCount", -1)), j.get("records", [{}])[:1]
    except Exception:
        return -1, []

total, _ = count({})
print(f"unfiltered rowsCount = {total}")

# candidate filter syntaxes (Solr-ish backends vary); a hit = rowsCount drops AND first record is Part 2 / 2005
cands = [
    {"type": C.PART2, "year": "2005"},
    {"fq": f'type:"{C.PART2}" AND year:2005'},
    {"filter": f'type:"{C.PART2}",year:2005'},
    {"filters": json.dumps({"type": [C.PART2], "year": ["2005"]})},
    {"bucketRanges": "year:[2005 TO 2005]", "type": C.PART2},
    {"q": f'type:"{C.PART2}" year:2005'},
    {"searchText": "", "type": C.PART2, "fromYear": "2005", "toYear": "2005"},
    {"typeFilter": C.PART2, "yearFilter": "2005"},
]
hit = None
for extra in cands:
    n, rec = count(extra); time.sleep(C.REQUEST_DELAY_S)
    r0 = rec[0] if rec else {}
    ok = 0 < n < total and r0.get("type") == C.PART2 and str(r0.get("year")) == "2005"
    print(f"{'HIT ' if ok else '    '} rowsCount={n:>8}  first={r0.get('type')!s:.12}/{r0.get('year')}  params={extra}")
    if ok and hit is None: hit = extra
print("\nFILTER_PARAMS =", json.dumps(hit).replace("2005", "{year}") if hit else "None  (use walk-all fallback)")

# page-size ceiling
for rows in (500, 1000, 2000):
    r = s.get(C.LIST_URL, params={**base, "rows": rows}, timeout=120)
    got = len(r.json().get("records", [])); print(f"rows={rows} -> returned {got}")
    if got < rows: break

# per-year facet counts (denominator for 10_temporal.py)
r = s.get(C.FACET_URL, params={"start": 0, "rows": 200, "field": "year", "collectionId": C.COLLECTION}, timeout=60)
open("year_facet.json", "w").write(r.text); print("\nyear facet ->", r.text[:400])
r = s.get(C.FACET_URL, params={"start": 0, "rows": 100, "field": "type", "order": "count", "collectionId": C.COLLECTION}, timeout=60)
open("type_facet.json", "w").write(r.text); print("type facet ->", r.text[:400])