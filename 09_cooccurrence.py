"""09 — Base-rate-normalised thematic co-occurrence, at the DEBATE and the SPEECH level.

Why: Table 4.1 in the dissertation reports raw counts (43 vs 14) and reads the
gap as 'structural separation', but Suicide&Crisis is ~7x larger than DirectMH,
so raw counts are dominated by cluster size. For each pair this script reports
observed, expected under independence, O/E, PMI, Jaccard, Fisher's exact p,
a permutation p, and Benjamini-Hochberg q-values across all pairs.

The dissertation's claim is about the same 'deliberative breath' — that is a
SPEECH-level claim, which v1 could not test because units were 500-word windows.
With speaker turns from 04 both levels are reported:
  python 09_cooccurrence.py            -> cooccurrence_debate.csv, cooccurrence_speech.csv"""
import itertools, numpy as np, pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
import config as C

dt_all = pd.read_csv("doc_topics.csv"); tg = pd.read_csv("topic_groups.csv")
dt_all = dt_all.merge(tg[["topic", "group_final"]], on="topic", how="left")
dt = dt_all[dt_all.group_final.notna() & (dt_all.group_final != "Other")]
groups = sorted(dt.group_final.unique())
rng = np.random.default_rng(0)

def run(unit_cols, name, min_chunks=2):
    # a unit with a single chunk can carry only one grouping (hard assignment), so it can
    # never co-occur; such units would deflate the independence baseline -> excluded
    size = dt_all.groupby(unit_cols).size()
    eligible = size[size >= min_chunks].index
    d = dt.set_index(unit_cols).loc[dt.set_index(unit_cols).index.isin(eligible)].reset_index()
    P = (d.groupby(unit_cols + ["group_final"]).size().unstack(fill_value=0) > 0).astype(int)
    P = P.reindex(columns=groups, fill_value=0); N = len(P); freq = P.sum() / N
    rows = []
    for a, b in itertools.combinations(groups, 2):
        A, B = P[a].values, P[b].values
        obs = int((A & B).sum()); exp = N * freq[a] * freq[b]; union = int((A | B).sum())
        table = [[obs, int(A.sum()) - obs], [int(B.sum()) - obs, N - union]]
        _, p_fisher_less = fisher_exact(table, alternative="less")
        _, p_fisher_greater = fisher_exact(table, alternative="greater")
        null = np.array([(A & rng.permutation(B)).sum() for _ in range(C.N_PERMUTATIONS)])
        rows.append({"unit": name, "A": a, "B": b, "n_units": N, "n_A": int(A.sum()), "n_B": int(B.sum()),
                     "observed": obs, "expected": round(exp, 1), "obs_over_exp": round(obs / exp, 2) if exp else np.nan,
                     "pmi": round(np.log2(max(obs, 0.5) / exp), 2) if exp else np.nan,
                     "jaccard": round(obs / union, 3) if union else 0.0,
                     "p_perm_less": (null <= obs).mean(), "p_perm_greater": (null >= obs).mean(),
                     "p_fisher_less": p_fisher_less, "p_fisher_greater": p_fisher_greater})
    res = pd.DataFrame(rows)
    for side in ["less", "greater"]:
        res[f"q_fisher_{side}"] = multipletests(res[f"p_fisher_{side}"], method="fdr_bh")[1]
    res = res.sort_values("obs_over_exp")
    res.to_csv(f"cooccurrence_{name}.csv", index=False)
    print(f"\n=== unit = {name}: {N} units with >=1 MH grouping and >={min_chunks} chunks ===")
    print(res[["A", "B", "observed", "expected", "obs_over_exp", "jaccard", "p_fisher_less", "q_fisher_less", "q_fisher_greater"]]
          .round(4).to_string(index=False))
    return res

run(["record_id"], "debate")
if "turn_id" in dt and dt.turn_id.nunique() > 1:
    run(["record_id", "turn_id"], "speech")
print("\nRead: obs_over_exp < 1 with small q_fisher_less => discussed together LESS than base rates predict"
      " (structural separation); obs_over_exp > 1 with small q_fisher_greater => discursive alignment.")
