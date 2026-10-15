"""Translation fidelity audit (translate mode only).

Automatic: cross-lingual semantic similarity between each Hindi source sentence
and its English MT output using LaBSE (language-agnostic sentence embeddings).
A faithful translation sits near its source in LaBSE space; a hallucinated or
truncated one does not. Reported as a distribution + share below 0.6.

Human: mt_fidelity_sheet.csv with 60 random sentence pairs for a bilingual rater to
score adequacy (1-5) and flag lost domain terms. --score reports the mean and the
correlation with LaBSE similarity (validates the automatic metric on this corpus)."""
import argparse, json, re, numpy as np, pandas as pd
import config as C

ap = argparse.ArgumentParser(); ap.add_argument("--score", action="store_true"); a = ap.parse_args()
DEV = re.compile(r"[ऀ-ॿ]"); LATIN = re.compile(r"[A-Za-z]")
SENT = re.compile(r"(?<=[.!?।])\s+")

if not a.score:
    from sentence_transformers import SentenceTransformer
    df = pd.read_csv("bertopic_corpus.csv")
    cache = {}
    import hashlib
    for line in open("mt_cache.jsonl", encoding="utf-8"):
        r = json.loads(line); cache[r["k"]] = r["t"]
    pairs = []
    for t in df.loc[df.lang.isin(["hi", "mixed"]), "text"].astype(str):
        for s in SENT.split(t):
            k = hashlib.sha1(s.encode()).hexdigest()
            if k in cache and len(s.split()) >= 4: pairs.append((s, cache[k]))
    pairs = list(dict.fromkeys(pairs))
    rng = np.random.default_rng(0); idx = rng.choice(len(pairs), min(3000, len(pairs)), replace=False)
    sample = [pairs[i] for i in idx]
    m = SentenceTransformer("sentence-transformers/LaBSE")
    a_ = m.encode([p[0] for p in sample], normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    b_ = m.encode([p[1] for p in sample], normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    sim = (a_ * b_).sum(1)
    res = pd.DataFrame({"hindi": [p[0] for p in sample], "english": [p[1] for p in sample], "labse_cos": sim.round(3)})
    res.to_csv("mt_similarity.csv", index=False)
    print(res.labse_cos.describe().round(3).to_string())
    print(f"share below 0.6 (probable mistranslation): {(res.labse_cos < 0.6).mean():.3f}")
    sheet = res.sample(min(60, len(res)), random_state=1).copy()
    sheet["adequacy_1to5"] = ""; sheet["lost_terms"] = ""
    sheet.to_csv("mt_fidelity_sheet.csv", index=False)
    print("mt_fidelity_sheet.csv written -> bilingual rater fills adequacy_1to5, then --score")
else:
    from scipy.stats import spearmanr
    s = pd.read_csv("mt_fidelity_sheet.csv").dropna(subset=["adequacy_1to5"])
    s = s[s.adequacy_1to5.astype(str).str.strip() != ""]
    s["adequacy_1to5"] = s.adequacy_1to5.astype(float)
    rho, p = spearmanr(s.adequacy_1to5, s.labse_cos)
    print(f"n={len(s)}  mean adequacy={s.adequacy_1to5.mean():.2f}  share<=2: {(s.adequacy_1to5 <= 2).mean():.3f}")
    print(f"Spearman(adequacy, LaBSE cos) = {rho:.3f} (p={p:.3g})")
    print("Most frequently lost terms:\n", s.lost_terms.dropna().str.lower().str.split("[,;]").explode().str.strip().value_counts().head(15).to_string())
