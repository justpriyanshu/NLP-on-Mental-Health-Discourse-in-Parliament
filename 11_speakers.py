"""11 — Who raises mental health? Speaker-level disaggregation (was 'future work').

Needs the speaker columns from 04. Without any extra data it answers:
  * which members carry each MH grouping (top speakers, concentration: share of MH
    chunks from the top-10 speakers, Gini);
  * ministers vs. non-ministers (proxy for treasury bench vs. floor) per grouping,
    with a chi-square — is MH *raised* by members or *answered* by government?
  * state of the speaker (from the '(State)' suffix) x grouping.
With an optional members.csv (columns: name, party, state; e.g. exported from the
sansad.in members directory) it adds party x grouping.

Outputs: speakers_mh.csv, role_by_group.csv, state_by_group.csv, [party_by_group.csv]"""
import os, re, difflib, numpy as np, pandas as pd
from scipy.stats import chi2_contingency
import config as C

dt = pd.read_csv("doc_topics.csv"); tg = pd.read_csv("topic_groups.csv")
dt = dt.merge(tg[["topic", "group_final"]], on="topic", how="left")
dt["group_final"] = dt.group_final.fillna("Other")
dt["speaker"] = dt.speaker.fillna("").astype(str).str.upper().str.strip()
dt = dt[dt.speaker.ne("") & dt.speaker.ne("CHAIR")]
dt["is_minister"] = dt.speaker_role.fillna("").str.contains("MINISTER", case=False)
mh = dt[dt.group_final != "Other"]

def chi(ct, label):
    if ct.shape[0] < 2 or ct.shape[1] < 2: return
    chi2, p, dof, _ = chi2_contingency(ct); v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
    print(f"{label}: chi2={chi2:.1f} dof={dof} p={p:.2g} Cramer's V={v:.3f}")

# ---- speakers ----------------------------------------------------------------
tot = dt.groupby("speaker").size().rename("chunks_total")
sp = mh.groupby(["speaker", "group_final"]).size().unstack(fill_value=0)
sp["mh_chunks"] = sp.sum(1); sp = sp.join(tot); sp["mh_share_of_own_speech"] = (sp.mh_chunks / sp.chunks_total).round(3)
sp["state"] = mh.groupby("speaker").speaker_state.agg(lambda s: s.dropna().mode().iloc[0] if s.dropna().size else "")
sp = sp.sort_values("mh_chunks", ascending=False); sp.to_csv("speakers_mh.csv")
top10 = sp.mh_chunks.head(10).sum() / sp.mh_chunks.sum()
x = np.sort(sp.mh_chunks.values); gini = 1 - 2 * np.sum(np.cumsum(x) / x.sum()) / len(x) + 1 / len(x)
print(f"{len(sp)} speakers with MH chunks; top-10 carry {top10:.1%}; Gini of MH chunks across speakers = {gini:.3f}")
print(sp.head(15)[["mh_chunks", "chunks_total", "mh_share_of_own_speech", "state"]].to_string())

# ---- ministers vs members --------------------------------------------------------
role = pd.crosstab(mh.is_minister.map({True: "minister", False: "member"}), mh.group_final)
role.to_csv("role_by_group.csv"); print("\nMinister vs member (chunks):\n", role.to_string()); chi(role, "role x group")
base = dt.is_minister.mean()
print(f"minister share of ALL chunks = {base:.3f}; of MH chunks = {mh.is_minister.mean():.3f}")

# ---- state -------------------------------------------------------------------
st = mh[mh.speaker_state.fillna("") != ""]
state = pd.crosstab(st.speaker_state, st.group_final)
state = state[state.sum(axis=1) >= 20]; state.to_csv("state_by_group.csv")
print("\nState x group (states with >=20 MH chunks):\n", state.to_string()); chi(state, "state x group")

# ---- party (optional) ---------------------------------------------------------
if os.path.exists("members.csv"):
    mem = pd.read_csv("members.csv"); mem["key"] = mem.name.astype(str).str.upper().str.replace(r"[^A-Z ]", "", regex=True).str.strip()
    keys = mem.key.tolist()
    def match(n):
        n = re.sub(r"[^A-Z ]", "", n).strip()
        m = difflib.get_close_matches(n, keys, n=1, cutoff=0.85); return m[0] if m else None
    lut = {s: match(s) for s in mh.speaker.unique()}
    mh = mh.assign(key=mh.speaker.map(lut)).merge(mem[["key", "party"]], on="key", how="left")
    print(f"\nmatched {mh.key.notna().mean():.1%} of MH chunks to members.csv")
    party = pd.crosstab(mh.party.fillna("UNMATCHED"), mh.group_final); party.to_csv("party_by_group.csv")
    print(party.to_string()); chi(party[party.index != "UNMATCHED"], "party x group")
else:
    print("\n(no members.csv -> party breakdown skipped; columns needed: name, party, state)")
