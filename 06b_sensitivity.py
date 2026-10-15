"""06b — Hyper-parameter sensitivity grid (reuses embeddings.npy; only UMAP+HDBSCAN re-run).

Why: the dissertation reports min_cluster_size=20 / min_samples=10 with one-line
rationales; on a corpus ~20x larger the same values would yield hundreds of
micro-topics. Table 2.1 should instead cite this grid: for every combination,
number of topics, noise share, NPMI, C_v and diversity. A transparent rule then
picks the configuration; override with --choose <row_index> if you prefer another
row and say why in the text.

Selection rule (edit here, report in the thesis): among rows with noise share
<= 0.35 and 25 <= n_topics <= 150, take the highest NPMI; ties -> higher diversity."""
import argparse, itertools, json, numpy as np, pandas as pd
from tqdm import tqdm
import config as C, tm_common as T, tm_eval as E

ap = argparse.ArgumentParser(); ap.add_argument("--choose", type=int); a = ap.parse_args()
df, docs = T.load_corpus(); emb = np.load("embeddings.npy"); assert len(emb) == len(docs)

if a.choose is None:
    ref = E.Reference(docs)
    rows = []
    G = C.SENSITIVITY_GRID
    for mcs, ms, nn in tqdm(list(itertools.product(G["min_cluster_size"], G["min_samples"], G["n_neighbors"]))):
        if ms > mcs: continue
        hdb = dict(T.hdbscan_params(), min_cluster_size=mcs, min_samples=ms)
        um = dict(C.UMAP_KW, n_neighbors=nn)
        m = T.build(C.SEEDS[0], embedding_model=None, hdb=hdb, umap_kw=um, prediction_data=False)
        t, _ = m.fit_transform(docs, emb)
        t = np.array(t)
        sc = ref.score(E.bertopic_words(m))
        rows.append({"min_cluster_size": mcs, "min_samples": ms, "n_neighbors": nn,
                     "n_topics": int(len(set(t)) - (1 if -1 in t else 0)), "noise_share": float((t == -1).mean()),
                     "largest_topic_share": float(pd.Series(t[t != -1]).value_counts(normalize=True).max()) if (t != -1).any() else np.nan,
                     **{k: round(v, 4) for k, v in sc.items()}})
        pd.DataFrame(rows).to_csv("sensitivity_grid.csv", index=False)
    grid = pd.DataFrame(rows)
else:
    grid = pd.read_csv("sensitivity_grid.csv")

print(grid.to_string())
ok = grid[(grid.noise_share <= 0.35) & (grid.n_topics.between(25, 150))]
if a.choose is not None: best = grid.loc[a.choose]
elif len(ok): best = ok.sort_values(["c_npmi", "diversity"], ascending=False).iloc[0]
else: best = grid.sort_values("c_npmi", ascending=False).iloc[0]; print("WARNING: no row met the constraints")
chosen = {"hdbscan": {"min_cluster_size": int(best.min_cluster_size), "min_samples": int(best.min_samples)},
          "umap": {"n_neighbors": int(best.n_neighbors)},
          "rule": "max NPMI s.t. noise<=0.35 and 25<=k<=150" if a.choose is None else f"manual row {a.choose}",
          "metrics": {k: (float(best[k]) if k in best else None) for k in ["n_topics", "noise_share", "c_npmi", "c_v", "diversity"]}}
json.dump(chosen, open("chosen_params.json", "w"), indent=2)
print("\nchosen_params.json:", json.dumps(chosen, indent=2))
