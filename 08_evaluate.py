"""08 — Model evaluation (the dissertation reported none).
 - NPMI + C_v coherence and topic diversity for the fitted BERTopic model
 - LDA and NMF baselines at the same k, same vocabulary, same metrics
 - seed stability: ARI between runs across config.SEEDS (all docs, and non-noise
   docs only, since HDBSCAN's noise label inflates/deflates ARI), plus per-seed
   coherence so the primary seed is shown not to be cherry-picked
Reuses embeddings.npy: seeds re-run only UMAP+HDBSCAN.  Output: evaluation.json"""
import json, numpy as np, pandas as pd
from bertopic import BERTopic
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import config as C, tm_common as T, tm_eval as E

df, docs = T.load_corpus(); emb = np.load("embeddings.npy"); assert len(emb) == len(docs)
ref = E.Reference(docs)

primary = BERTopic.load("model")
dt = pd.read_csv("doc_topics.csv"); assert len(dt) == len(docs), "doc_topics.csv / corpus mismatch: rerun 06"
base = dt.topic.values                     # safetensors save does not persist per-document topics_
k = len(primary.get_topics()) - (1 if -1 in primary.get_topics() else 0)
res = {"bertopic": {**ref.score(E.bertopic_words(primary)), "n_topics": k, "noise_share": float((base == -1).mean())}}

# ---- classical baselines at the same k, same vocabulary ---------------------
cv = T.vectorizer(); X = cv.fit_transform(docs); vocab = np.array(cv.get_feature_names_out())
lda = LatentDirichletAllocation(n_components=k, random_state=C.SEEDS[0], learning_method="online", max_iter=20).fit(X)
res["lda"] = {**ref.score([list(vocab[np.argsort(r)[::-1][:20]]) for r in lda.components_]), "n_topics": k}
tf = TfidfVectorizer(ngram_range=(1, 2), stop_words=T.STOP, min_df=C.VECTORIZER_MIN_DF); Xt = tf.fit_transform(docs)
vt = np.array(tf.get_feature_names_out())
nmf = NMF(n_components=k, random_state=C.SEEDS[0], init="nndsvd", max_iter=400).fit(Xt)
res["nmf"] = {**ref.score([list(vt[np.argsort(r)[::-1][:20]]) for r in nmf.components_]), "n_topics": k}

# ---- seed stability ---------------------------------------------------------
runs = {C.SEEDS[0]: base}
for s in C.SEEDS[1:]:
    m = T.build(s, embedding_model=None, prediction_data=False)
    t, _ = m.fit_transform(docs, emb); runs[s] = np.array(t)
    res.setdefault("per_seed", {})[str(s)] = {**ref.score(E.bertopic_words(m)),
                                             "n_topics": int(len(set(t)) - (1 if -1 in t else 0)),
                                             "noise_share": float((runs[s] == -1).mean())}
res["per_seed"][str(C.SEEDS[0])] = res["bertopic"]
pairs = [(a, b) for i, a in enumerate(C.SEEDS) for b in C.SEEDS[i + 1:]]
ari_all, ari_nn, nmi_nn = [], [], []
for a, b in pairs:
    ta, tb = runs[a], runs[b]; keep = (ta != -1) & (tb != -1)
    ari_all.append(adjusted_rand_score(ta, tb)); ari_nn.append(adjusted_rand_score(ta[keep], tb[keep]))
    nmi_nn.append(normalized_mutual_info_score(ta[keep], tb[keep]))
res["stability"] = {"seeds": C.SEEDS, "pairwise_ari_all_docs_mean": float(np.mean(ari_all)),
                    "pairwise_ari_nonnoise_mean": float(np.mean(ari_nn)), "pairwise_nmi_nonnoise_mean": float(np.mean(nmi_nn)),
                    "ari_vs_primary": {str(s): float(adjusted_rand_score(base, runs[s])) for s in C.SEEDS[1:]},
                    "n_topics_per_seed": {str(s): int(len(set(r)) - (1 if -1 in r else 0)) for s, r in runs.items()}}
json.dump(res, open("evaluation.json", "w"), indent=2)
print(json.dumps(res, indent=2))
