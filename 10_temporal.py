"""10 — Attention share over time, with proper denominators, uncertainty and
data-driven change points.

Why: Figure 5.1 in the dissertation plots raw counts of a keyword-filtered
corpus, so 'volume' conflates how much Parliament sat with how much it talked
about MH. With the full corpus the denominator is known. Three are reported:
  words     — all chunk words in the year (primary; volume of speech)
  debates   — Part 2 records in the year (ledger)
  pages     — parsed from pageNoFromTo when present (v1 summed the STRING field)
Share = MH debates / all debates carries a Wilson 95% CI (binomial, debate level).

Change points: PELT with a penalty sweep (breakpoints that survive many
penalties are robust), and Binseg with n_bkps=4 = the best 5-period split, to be
compared with the a-priori scheme in config.PERIODS. A chi-square test of the
group x period contingency table asks whether composition differs across periods."""
import re, json, numpy as np, pandas as pd
import ruptures as rpt
from scipy.stats import chi2_contingency
from statsmodels.stats.proportion import proportion_confint
import config as C

years = list(range(C.FROM_YEAR, C.TO_YEAR + 1))
ledger = pd.read_csv("metadata_ledger.csv"); seg = pd.read_csv("segments.csv", usecols=["record_id", "year", "n_words"])
dt = pd.read_csv("doc_topics.csv"); tg = pd.read_csv("topic_groups.csv")
dt = dt.merge(tg[["topic", "group_final"]], on="topic", how="left")
mh = dt[dt.group_final.notna() & (dt.group_final != "Other")]

def pages_of(s):
    m = re.search(r"p?(\d+)\s*-\s*p?(\d+)", str(s)); return int(m.group(2)) - int(m.group(1)) + 1 if m and int(m.group(2)) >= int(m.group(1)) else np.nan

den = pd.DataFrame(index=years)
den["words"] = seg.groupby("year").n_words.sum().reindex(years)
den["debates"] = ledger.groupby("year").size().reindex(years)
ledger["n_pages"] = ledger.pages.map(pages_of) if "pages" in ledger else np.nan
den["pages"] = ledger.groupby("year").n_pages.sum(min_count=1).reindex(years)

share = pd.DataFrame(index=years)
mh_deb = mh.groupby("year").record_id.nunique().reindex(years).fillna(0)
share["mh_debates"] = mh_deb; share["all_debates"] = den.debates
share["share_debates"] = mh_deb / den.debates
lo, hi = proportion_confint(mh_deb.fillna(0).astype(int), den.debates.fillna(0).astype(int).clip(lower=1), method="wilson")
share["share_debates_lo95"], share["share_debates_hi95"] = lo, hi
share["share_words"] = mh.groupby("year").n_words.sum().reindex(years).fillna(0) / den.words
share["share_pages"] = mh_deb / den.pages if den.pages.notna().any() else np.nan
for g, sub in mh.groupby("group_final"):
    share[f"{g}_share_debates"] = sub.groupby("year").record_id.nunique().reindex(years).fillna(0) / den.debates
    share[f"{g}_share_words"] = sub.groupby("year").n_words.sum().reindex(years).fillna(0) / den.words
share = share.join(den.add_prefix("denominator_"))
share.to_csv("attention_share_by_year.csv")
print(share[["mh_debates", "all_debates", "share_debates", "share_debates_lo95", "share_debates_hi95", "share_words"]].round(4).to_string())

# ---- change points -----------------------------------------------------------
sig = share["share_words"].fillna(0).values.reshape(-1, 1)
sweep = {}
for pen in np.round(np.logspace(-2, 1, 13), 3):
    bk = rpt.Pelt(model="rbf", min_size=3, jump=1).fit(sig).predict(pen=pen)
    sweep[str(pen)] = [years[b] for b in bk[:-1]]
votes = pd.Series([y for v in sweep.values() for y in v]).value_counts()
bin4 = rpt.Binseg(model="rbf", min_size=3, jump=1).fit(sig).predict(n_bkps=4)
best5 = [years[b] for b in bin4[:-1]]
apriori = [lo for _, (lo, _) in list(C.PERIODS.items())[1:]]
cp = {"signal": "share_words", "pelt_penalty_sweep": sweep,
      "breakpoint_votes_across_penalties": votes.to_dict(),
      "binseg_best_5_period_split_starts": best5, "apriori_period_starts": apriori}
print("\nBreakpoint votes across PELT penalties:", votes.to_dict())
print("Best data-driven 5-period split starts at:", best5, " | a-priori scheme:", apriori)

# ---- does thematic composition differ across the a-priori periods? ----------
def period_of(y):
    for name, (lo, hi) in C.PERIODS.items():
        if lo <= y <= hi: return name
mh = mh.assign(period=mh.year.astype(int).map(period_of))
ct = mh.drop_duplicates(["record_id", "group_final"]).groupby(["period", "group_final"]).size().unstack(fill_value=0)
ct = ct.reindex(list(C.PERIODS.keys())).fillna(0).astype(int)
chi2, p, dof, _ = chi2_contingency(ct)
n = ct.values.sum(); cramers_v = float(np.sqrt(chi2 / (n * (min(ct.shape) - 1))))
comp = ct.div(ct.sum(axis=1), axis=0).round(3)
comp.to_csv("period_composition.csv"); ct.to_csv("period_counts.csv")
cp["composition_chi2"] = {"chi2": float(chi2), "p": float(p), "dof": int(dof), "cramers_v": cramers_v}
json.dump(cp, open("changepoints.json", "w"), indent=2)
print(f"\nGroup x period (debate counts):\n{ct.to_string()}\nchi2={chi2:.1f} dof={dof} p={p:.2g} Cramer's V={cramers_v:.3f}")
print("\nRow-normalised composition:\n", comp.to_string())
