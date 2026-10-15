"""04 — Cleaning, speaker-turn segmentation and chunking (replaces 04_normalize.py).

Why: v1 cut each debate into blind 500-word windows, so one 'document' could
straddle three speakers and the model learned speaker names as topic words
(T16 'jagat prakash nadda deputyy'). Rajya Sabha transcripts mark every turn
("SHRI X (State): ...", "MR. DEPUTY CHAIRMAN: ...", "श्री ...:"), so the
natural unit is the speech. Chunks (<=500 words, model context limit) are cut
WITHIN a turn and inherit the speaker, which also unlocks the speaker/party
disaggregation listed as future work in the dissertation.

Language flag is by Devanagari share (en / mixed / hi / hi-roman) instead of
'any Devanagari character => hi'. The tabular-leakage filter now excludes
Devanagari marks from the special-character count, so it can be applied to all
languages instead of exempting Hindi.

Input : raw_text.csv (from 03)   Output: segments.csv, segmentation_stats.json"""
import re, json, numpy as np, pandas as pd
import config as C

HON = r"(?:SHRI|SHRIMATI|SMT\.?|DR\.?|PROF\.?|KUMARI|SARDAR|MR\.?|MRS\.?|MS\.?|MAULANA|PANDIT|SWAMI|CH\.?|SH\.?|THE|HON'?BLE)"
SPK_EN = re.compile(r"^[ \t]*((?:%s)[ \t][A-Z][A-Z0-9 .,'\-&]{1,120}?(?:\([^)\n]{1,90}\))?)[ \t]*[:;][ \t]" % HON, re.M)
SPK_HI = re.compile(r"^[ \t]*((?:श्री|श्रीमती|डॉ\.?|प्रो\.?|कुमारी|सरदार|उपसभापति|सभापति|उपसभाध्यक्ष|माननीय)[^:\n]{0,90}?)[ \t]*:[ \t]", re.M)
CHAIR = re.compile(r"CHAIRMAN|CHAIRPERSON|SPEAKER|सभापति|उपसभाध्यक्ष|PANEL", re.I)
DEV = re.compile(r"[ऀ-ॿ]")
LATIN = re.compile(r"[A-Za-z]")
ROMAN_HI = re.compile(r"\b(hai|hain|nahi|nahin|aur|ke|ki|ka|mein|kya|yeh|woh|hum|aap|liye|karna|sarkar|kisan|mantri|hona|tha|thi|ko|kuch|bahut|desh|log)\b", re.I)
PAGE_NO = re.compile(r"^\s*(\[?\d{1,4}\]?|\[?RAJYA SABHA\]?|\d{1,2}\s*(?:st|nd|rd|th)?\s+[A-Za-z]+,?\s+\d{4})\s*$", re.I)
FIXES = {r"\bMINIST\b": "MINISTER", r"\bHon'?ble\b": "Honourable", r"\bGovt\.?\b": "Government"}

def strip_running_heads(pages):
    """Drop page numbers / running heads: short lines that recur across pages."""
    lines = [p.split("\n") for p in pages]
    counts = {}
    for ls in lines:
        for l in set(x.strip() for x in ls[:3] + ls[-3:] if 0 < len(x.strip()) < 60):
            counts[l] = counts.get(l, 0) + 1
    recurring = {l for l, c in counts.items() if c >= max(3, 0.3 * len(pages))}
    out = []
    for ls in lines:
        out.append("\n".join(l for l in ls if l.strip() not in recurring and not PAGE_NO.match(l)))
    return out

def clean(t):
    t = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", t)
    t = re.sub(r"(\w)-\n\s*(\w)", r"\1\2", t)                  # rejoin hyphenated line breaks
    t = re.sub(r"[^\w\sऀ-ॿ.,;:!?()'\"%\-]", " ", t)
    for pat, rep in FIXES.items(): t = re.sub(pat, rep, t)
    return re.sub(r"\s+", " ", t).strip()

def parse_speaker(raw):
    s = re.sub(r"\s+", " ", raw).strip()
    role, name, state = "", s, ""
    m = re.search(r"\(([^)]*)\)", s)
    if m:
        inside, outside = m.group(1).strip(), re.sub(r"\([^)]*\)", "", s).strip()
        if re.match(HON, inside) and re.match(r"THE\b", outside):      # THE MINISTER OF X (SHRI Y)
            role, name = outside, inside
        else:                                                            # SHRI Y (State)
            name, state = outside, inside
    if CHAIR.search(name) or CHAIR.search(role):
        role = role or name; name = "CHAIR"
    name = re.sub(r"^(?:%s)\s+" % HON, "", name)
    name = re.sub(r"^(?:श्री|श्रीमती|डॉ\.?|प्रो\.?|कुमारी|सरदार|माननीय)\s+", "", name).strip(" .,:;-")
    return name.upper(), role.upper(), state.title()

def turns_of(text):
    marks = [(m.start(), m.end(), m.group(1)) for m in SPK_EN.finditer(text)]
    marks += [(m.start(), m.end(), m.group(1)) for m in SPK_HI.finditer(text)]
    marks.sort()
    if len(marks) < 2:
        return [("", text)]
    out = []
    if text[:marks[0][0]].strip(): out.append(("", text[:marks[0][0]]))
    for i, (s, e, spk) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        out.append((spk, text[e:end]))
    return out

def lang_of(t):
    dev = len(DEV.findall(t)); lat = len(LATIN.findall(t)); tot = max(1, dev + lat)
    share = dev / tot
    if share >= C.DEVANAGARI_HI_SHARE: return "hi", share
    if share >= C.DEVANAGARI_MIXED_SHARE: return "mixed", share
    words = max(1, len(t.split()))
    if len(ROMAN_HI.findall(t)) / words * 100 >= C.ROMAN_HINDI_HITS_PER_100: return "hi-roman", share
    return "en", share

def tabular(t):
    body = [c for c in t if not DEV.match(c)]
    if not body: return False
    special = sum(not c.isalnum() and not c.isspace() for c in body) / len(body)
    digits = sum(c.isdigit() for c in body) / len(body)
    return special > 0.40 or digits > 0.30

def main():
    raw = pd.read_csv("raw_text.csv").sort_values(["record_id", "part", "page"])
    meta = pd.read_csv("metadata_ledger.csv")
    rows, stats = [], {"records": 0, "records_with_turns": 0, "turns": 0, "chunks_dropped_tabular": 0}
    for rid, g in raw.groupby("record_id", sort=False):
        stats["records"] += 1
        pages = strip_running_heads(g.text.fillna("").astype(str).tolist())
        text = "\n".join(pages)
        turns = turns_of(text)
        if len(turns) > 1: stats["records_with_turns"] += 1
        for tid, (spk_raw, body) in enumerate(turns):
            t = clean(body)
            words = t.split()
            if len(words) < C.MIN_WORDS: continue
            stats["turns"] += 1
            name, role, state = parse_speaker(spk_raw) if spk_raw else ("", "", "")
            bounds = list(range(0, len(words), C.CHUNK_WORDS))
            if len(bounds) > 1 and len(words) - bounds[-1] < C.MIN_WORDS: bounds.pop()   # merge short tail
            for ci, s in enumerate(bounds):
                e = bounds[ci + 1] if ci + 1 < len(bounds) else len(words)
                ct = " ".join(words[s:e])
                if tabular(ct): stats["chunks_dropped_tabular"] += 1; continue
                lang, share = lang_of(ct)
                rows.append({"record_id": rid, "turn_id": tid, "chunk": ci, "speaker_raw": spk_raw.strip(),
                             "speaker": name, "speaker_role": role, "speaker_state": state,
                             "lang": lang, "dev_share": round(share, 3), "n_words": e - s, "text": ct})
    out = pd.DataFrame(rows).merge(meta[["record_id", "title", "date", "year", "session", "subject", "language"]],
                                   on="record_id", how="left")
    out.to_csv("segments.csv", index=False)
    stats["chunks"] = len(out)
    stats["share_words_with_named_speaker"] = round(float(out.loc[out.speaker.ne(""), "n_words"].sum() / out.n_words.sum()), 3)
    stats["lang_counts"] = out.lang.value_counts().to_dict()
    stats["turns_per_record_median"] = float(out.groupby("record_id").turn_id.nunique().median())
    json.dump(stats, open("segmentation_stats.json", "w"), indent=2)
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
