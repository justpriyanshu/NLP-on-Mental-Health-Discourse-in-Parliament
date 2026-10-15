"""07 — Post-hoc MH-topic identification with two annotators, at TWO levels.

Topic level (as v1, fixed alignment bug):
  --build      annotation_sheet.csv: lexicon density, 20 MMR terms, 5 samples, dendrogram parent
  --kappa      Cohen's kappa on group_A/group_B; writes topic_groups.csv once group_final is filled

Document level (new): topic->group labels are only as good as the clusters. A
stratified sample of chunks is labelled independently of the model so the
precision of the automatic MH-relevance labelling can be reported.
  --doc-sample   doc_validation_sheet.csv (config.DOC_VALIDATION_N chunks; strata = final group,
                 'Other', noise, and lexicon-hit-but-Other so the recall side is also visible)
  --doc-kappa    kappa between annotators + precision/recall/F1 of the model's MH label
                 against the human majority, per group

Labels: DirectMH, SuicideCrisis, SubstanceAbuse, Disability, SocialDet, GenderViolence, Other"""
import argparse, numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support
import config as C

ap = argparse.ArgumentParser()
for f in ["build", "kappa", "doc-sample", "doc-kappa"]: ap.add_argument("--" + f, action="store_true")
a = ap.parse_args()
KEYS = ["record_id", "turn_id", "chunk"]

def corpus_with_topics():
    dt = pd.read_csv("doc_topics.csv")
    corpus = pd.read_csv("bertopic_corpus.csv")
    col = "text" if C.EMBEDDING_MODE == "multilingual" else "english"
    return corpus.merge(dt[KEYS + ["topic"]], on=KEYS, how="inner"), col

if a.build:
    from bertopic import BERTopic
    model = BERTopic.load("model")
    corpus, col = corpus_with_topics()
    hier = pd.read_csv("hierarchical_topics.csv") if __import__("os").path.exists("hierarchical_topics.csv") else None
    def parent_of(t):
        if hier is None: return ""
        rows = hier[hier.Topics.apply(lambda s: t in eval(s) if isinstance(s, str) else False)]
        if rows.empty: return ""
        r = rows.sort_values("Distance").iloc[0]      # smallest merge that contains t
        return f"{int(r.Parent_ID)}: {r.Parent_Name[:60]}"
    rows = []
    for t in sorted(corpus.topic.unique()):
        if t == -1: continue
        sub = corpus[corpus.topic == t]
        rows.append({"topic": t, "n_docs": len(sub), "n_debates": sub.record_id.nunique(),
                     "lexicon_density": round(sub.lexicon_hit.mean(), 3),
                     "top_terms": ", ".join(w for w, _ in model.get_topic(t)[:20]),
                     "dendrogram_parent": parent_of(t),
                     "samples": " ||| ".join(sub[col].astype(str).str[:400].sample(min(5, len(sub)), random_state=0)),
                     "group_A": "", "group_B": "", "group_final": ""})
    sheet = pd.DataFrame(rows).sort_values("lexicon_density", ascending=False)
    sheet.to_csv("annotation_sheet.csv", index=False)
    print(sheet[["topic", "n_docs", "n_debates", "lexicon_density"]].to_string())
    print("\nAnnotators label EVERY topic independently (blind to lexicon_density if possible).")

if a.kappa:
    s = pd.read_csv("annotation_sheet.csv").fillna("")
    lab = s[(s.group_A != "") & (s.group_B != "")]
    k = cohen_kappa_score(lab.group_A, lab.group_B)
    k_bin = cohen_kappa_score(lab.group_A != "Other", lab.group_B != "Other")
    print(f"Topic-level Cohen's kappa: {k:.3f} (7-way), {k_bin:.3f} (MH vs Other), n={len(lab)} topics")
    dis = lab[lab.group_A != lab.group_B]
    print(f"{len(dis)} disagreements:\n", dis[["topic", "group_A", "group_B", "top_terms"]].to_string())
    if (s.group_final != "").all():
        s[["topic", "group_final", "n_docs", "n_debates", "lexicon_density"]].to_csv("topic_groups.csv", index=False)
        print("topic_groups.csv written")

if a.doc_sample:
    corpus, col = corpus_with_topics()
    tg = pd.read_csv("topic_groups.csv")
    corpus = corpus.merge(tg[["topic", "group_final"]], on="topic", how="left")
    corpus["group_model"] = corpus.group_final.fillna("Noise")
    corpus["stratum"] = corpus.group_model
    corpus.loc[(corpus.group_model == "Other") & corpus.lexicon_hit, "stratum"] = "Other+lexicon"
    per = max(5, C.DOC_VALIDATION_N // corpus.stratum.nunique())
    samp = pd.concat([g.sample(min(per, len(g)), random_state=0) for _, g in corpus.groupby("stratum")])
    samp = samp.sample(frac=1, random_state=0)                       # shuffle so annotators are blind to stratum
    out = samp[KEYS + ["year", "title", col]].rename(columns={col: "text"})
    out["mh_relevant_A"] = ""; out["group_A"] = ""; out["mh_relevant_B"] = ""; out["group_B"] = ""
    out.to_csv("doc_validation_sheet.csv", index=False)
    samp[KEYS + ["stratum", "group_model", "topic"]].to_csv("doc_validation_key.csv", index=False)   # keep hidden
    print(f"{len(out)} chunks -> doc_validation_sheet.csv (annotators) ; doc_validation_key.csv (do not show)")

if a.doc_kappa:
    s = pd.read_csv("doc_validation_sheet.csv").merge(pd.read_csv("doc_validation_key.csv"), on=KEYS)
    s = s[(s.mh_relevant_A.astype(str) != "") & (s.mh_relevant_B.astype(str) != "")]
    s["mh_A"] = s.mh_relevant_A.astype(int); s["mh_B"] = s.mh_relevant_B.astype(int)
    print(f"Document-level kappa (MH relevant y/n): {cohen_kappa_score(s.mh_A, s.mh_B):.3f}  n={len(s)}")
    both = s[(s.group_A.fillna('') != '') & (s.group_B.fillna('') != '')]
    if len(both): print(f"Document-level kappa (7-way group): {cohen_kappa_score(both.group_A, both.group_B):.3f}  n={len(both)}")
    s["human_mh"] = ((s.mh_A + s.mh_B) >= 2).astype(int)              # both must agree -> conservative truth
    s["model_mh"] = (~s.group_model.isin(["Other", "Noise"])).astype(int)
    p, r, f, _ = precision_recall_fscore_support(s.human_mh, s.model_mh, average="binary")
    print(f"Model MH label vs human (both agree): precision={p:.3f} recall={r:.3f} F1={f:.3f}")
    print("\nBy model group (share judged MH-relevant by both annotators):")
    print(s.groupby("group_model").human_mh.agg(["mean", "count"]).round(3).to_string())
    print("\nRecall diagnostics — 'Other+lexicon' stratum judged MH-relevant:",
          round(s[s.stratum == "Other+lexicon"].human_mh.mean(), 3) if (s.stratum == "Other+lexicon").any() else "n/a")
