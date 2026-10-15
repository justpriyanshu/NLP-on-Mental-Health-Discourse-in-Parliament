"""12 — Dissertation figures from the analysis outputs (PNG, 300 dpi, figures/).
Colour: fixed categorical order per grouping (colour-blind-validated, never cycled);
diverging blue/grey/orange for O/E; one y-axis per panel; series direct-labelled."""
import os, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C

os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.edgecolor": "#8a8984",
                     "figure.facecolor": "white", "axes.facecolor": "#fcfcfb", "axes.axisbelow": True})
GROUPS = [g for g in C.MH_GROUPS if g != "Other"]
COL = dict(zip(GROUPS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
TXT = "#0b0b0b"; MUTED = "#52514e"

share = pd.read_csv("attention_share_by_year.csv", index_col=0)
years = share.index.values
cp = json.load(open("changepoints.json")) if os.path.exists("changepoints.json") else {}
apriori = [lo for _, (lo, _) in list(C.PERIODS.items())[1:]]

# ---- Fig 1: overall attention share with CI, a-priori vs data-driven boundaries
fig, ax = plt.subplots(figsize=(7.2, 3.4))
ax.fill_between(years, share.share_debates_lo95, share.share_debates_hi95, color="#2a78d6", alpha=0.15, lw=0)
ax.plot(years, share.share_debates, color="#2a78d6", lw=2, label="share of debates (95% Wilson CI)")
ax.plot(years, share.share_words, color="#eb6834", lw=2, label="share of words")
for x in apriori: ax.axvline(x, color=MUTED, lw=0.8, ls="--")
for x in cp.get("binseg_best_5_period_split_starts", []): ax.axvline(x, color="#e34948", lw=1.2, ls=":")
ax.set_ylabel("MH share of plenary volume"); ax.set_xlabel("")
ax.text(years[-1], share.share_debates.iloc[-1], "  debates", color="#2a78d6", va="center")
ax.text(years[-1], share.share_words.iloc[-1], "  words", color="#eb6834", va="center")
ax.set_title("Parliamentary attention to mental health, normalised by total plenary volume", loc="left", color=TXT)
ax.set_xlabel("dashed = a-priori period boundaries; dotted red = data-driven 5-period split (Binseg)", fontsize=7.5, color=MUTED)
fig.tight_layout(); fig.savefig("figures/fig_attention_share.png", dpi=300); plt.close(fig)

# ---- Fig 2: small multiples per grouping (share of words) --------------------
cols = [f"{g}_share_words" for g in GROUPS if f"{g}_share_words" in share]
fig, axes = plt.subplots(2, 3, figsize=(8.5, 4.6), sharex=True, sharey=True)
for ax, c in zip(axes.flat, cols):
    g = c.replace("_share_words", "")
    ax.plot(years, share[c], color=COL[g], lw=2); ax.set_title(g, loc="left", fontsize=9, color=TXT)
    for x in apriori: ax.axvline(x, color=MUTED, lw=0.6, ls="--")
for ax in axes.flat[len(cols):]: ax.axis("off")
fig.supylabel("share of all plenary words", fontsize=9); fig.tight_layout()
fig.savefig("figures/fig_group_shares.png", dpi=300); plt.close(fig)

# ---- Fig 3: co-occurrence O/E heatmaps (debate & speech) ---------------------
for unit in ["debate", "speech"]:
    f = f"cooccurrence_{unit}.csv"
    if not os.path.exists(f): continue
    co = pd.read_csv(f); gs = sorted(set(co.A) | set(co.B))
    M = pd.DataFrame(np.nan, index=gs, columns=gs); Q = pd.DataFrame("", index=gs, columns=gs)
    for _, r in co.iterrows():
        M.loc[r.A, r.B] = M.loc[r.B, r.A] = np.log2(r.obs_over_exp) if r.obs_over_exp > 0 else np.nan
        star = "*" if min(r.q_fisher_less, r.q_fisher_greater) < 0.05 else ""
        Q.loc[r.A, r.B] = Q.loc[r.B, r.A] = f"{r.obs_over_exp:.2f}{star}\n(n={int(r.observed)})"
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    lim = float(np.nanmax(np.abs(M.values))) if np.isfinite(M.values).any() and np.nanmax(np.abs(M.values)) > 0 else 1.0
    im = ax.imshow(M.values, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("d", ["#2a78d6", "#f0efec", "#eb6834"]),
                   vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(gs))); ax.set_yticks(range(len(gs))); ax.set_xticklabels(gs, rotation=35, ha="right"); ax.set_yticklabels(gs)
    ax.grid(False)
    for i in range(len(gs)):
        for j in range(len(gs)):
            if i != j: ax.text(j, i, Q.iloc[i, j], ha="center", va="center", fontsize=7, color=TXT)
    cb = fig.colorbar(im, ax=ax, shrink=0.8); cb.set_label("log2(observed / expected)")
    ax.set_title(f"Thematic co-occurrence within a {unit}, normalised by base rates\n(* = BH q < 0.05)", loc="left", fontsize=9)
    fig.tight_layout(); fig.savefig(f"figures/fig_cooccurrence_{unit}.png", dpi=300); plt.close(fig)

# ---- Fig 4: composition by period (100% stacked, fixed order, 2px gaps) ------
if os.path.exists("period_composition.csv"):
    comp = pd.read_csv("period_composition.csv", index_col=0).reindex(columns=[g for g in GROUPS if g in pd.read_csv("period_composition.csv", index_col=0)])
    fig, ax = plt.subplots(figsize=(7.2, 3.4)); left = np.zeros(len(comp))
    for g in comp.columns:
        ax.barh(comp.index, comp[g], left=left, color=COL[g], label=g, height=0.62, edgecolor="#fcfcfb", linewidth=1.5)
        for i, (v, l) in enumerate(zip(comp[g], left)):
            if v >= 0.08: ax.text(l + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=7, color="white")
        left += comp[g].values
    ax.set_xlim(0, 1); ax.invert_yaxis(); ax.set_xlabel("share of MH debates in period (a debate may carry several groupings)")
    ax.legend(ncol=3, fontsize=7.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(); fig.savefig("figures/fig_period_composition.png", dpi=300); plt.close(fig)

# ---- Fig 5: sensitivity grid ---------------------------------------------------
if os.path.exists("sensitivity_grid.csv"):
    g = pd.read_csv("sensitivity_grid.csv")
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    for ax, m in zip(axes, ["n_topics", "noise_share", "c_npmi"]):
        for ((ms, nn), sub), col in zip(g.groupby(["min_samples", "n_neighbors"]), list(COL.values()) + ["#4a3aa7", "#e34948"]):
            ax.plot(sub.min_cluster_size, sub[m], marker="o", ms=3, lw=1.2, color=col, label=f"min_samples={ms}, nn={nn}")
        ax.set_xlabel("min_cluster_size"); ax.set_title(m, loc="left", fontsize=9)
    axes[-1].legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig("figures/fig_sensitivity.png", dpi=300); plt.close(fig)
print("figures/ written:", sorted(os.listdir("figures")))
