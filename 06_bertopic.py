"""06 — Embed once, fit BERTopic on the FULL corpus with the chosen hyper-parameters.

  python 06_bertopic.py --embed-only     # embeddings.npy (then run 06b_sensitivity.py)
  python 06_bertopic.py                  # fit with chosen_params.json (or config fallback)

Fixes vs v1: doc_topics referenced a non-existent 'file' column (crash); topic
words now use MMR + min_df so OCR one-offs and speaker names stop appearing as
topic terms; hierarchical topic tree saved so the six thematic groupings can be
justified against the dendrogram rather than asserted; top_n_words=20 so the
annotation sheet and coherence use the same word lists.

Dynamic topics: topics_over_time(evolution_tuning=True) averages each year-bin's
c-TF-IDF with the previous bin's -> a FIXED alpha of 0.5 (Eq. 2.4 in the
dissertation describes alpha as tuned; it is not tunable in BERTopic)."""
import argparse, json, numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
import config as C, tm_common as T

ap = argparse.ArgumentParser(); ap.add_argument("--embed-only", action="store_true"); a = ap.parse_args()
df, docs = T.load_corpus()
model_name = C.EMBED_MODELS[C.EMBEDDING_MODE]
emb_model = SentenceTransformer(model_name)

try:
    emb = np.load("embeddings.npy"); assert len(emb) == len(docs)
    print("reusing embeddings.npy")
except Exception:
    emb = emb_model.encode(docs, show_progress_bar=True, batch_size=64)
    np.save("embeddings.npy", emb)
if a.embed_only: raise SystemExit("embeddings saved; now run 06b_sensitivity.py")

model = T.build(C.SEEDS[0], embedding_model=emb_model)
topics, _ = model.fit_transform(docs, emb)
df["topic"] = topics
model.get_topic_info().to_csv("topics.csv", index=False)
df[T.KEEP + ["topic"]].to_csv("doc_topics.csv", index=False)

# hierarchy -> justification for the thematic groupings (07) and Appendix figure
hier = model.hierarchical_topics(docs)
hier.to_csv("hierarchical_topics.csv", index=False)
open("topic_tree.txt", "w", encoding="utf-8").write(model.get_topic_tree(hier))

years = df.year.astype(int).tolist()
tot = model.topics_over_time(docs, years, nr_bins=len(set(years)), evolution_tuning=True, global_tuning=True)
tot.to_csv("topics_over_time.csv", index=False)
model.save("model", serialization="safetensors", save_ctfidf=True, save_embedding_model=model_name)

n_topics = len([t for t in set(topics) if t != -1])
stats = {"n_docs": len(docs), "n_debates": int(df.record_id.nunique()), "n_topics": n_topics,
         "noise_share": float(np.mean(np.array(topics) == -1)), "embedding": model_name,
         "umap": T.umap_params(), "hdbscan": T.hdbscan_params(), "top_n_words": C.TOP_N_WORDS,
         "mmr_diversity": C.MMR_DIVERSITY, "min_df": C.VECTORIZER_MIN_DF,
         "dtm_alpha_fixed": 0.5, "seed": C.SEEDS[0]}
json.dump(stats, open("model_stats.json", "w"), indent=2)
print(json.dumps(stats, indent=2))
