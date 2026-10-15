"""Shared model construction so 06, 06b and 08 fit IDENTICAL pipelines."""
import json, os, pandas as pd
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
import config as C

STOP = list(ENGLISH_STOP_WORDS) + C.STOP_WORDS
TEXT_COL = "text" if C.EMBEDDING_MODE == "multilingual" else "english"
KEEP = ["record_id", "turn_id", "chunk", "speaker", "speaker_role", "speaker_state",
        "year", "date", "title", "lang", "n_words", "lexicon_hit"]

def load_corpus():
    df = pd.read_csv("bertopic_corpus.csv")
    df = df[df[TEXT_COL].notna() & (df[TEXT_COL].astype(str).str.strip() != "")].reset_index(drop=True)
    return df, df[TEXT_COL].astype(str).tolist()

def hdbscan_params():
    p = dict(C.HDBSCAN_FALLBACK)
    if os.path.exists("chosen_params.json"):
        p.update(json.load(open("chosen_params.json"))["hdbscan"])
    for k, v in C.HDBSCAN_KW.items():
        if v is not None: p[k] = v
    return p

def umap_params():
    p = dict(C.UMAP_KW)
    if os.path.exists("chosen_params.json"):
        p.update(json.load(open("chosen_params.json")).get("umap", {}))
    return p

def vectorizer():
    return CountVectorizer(ngram_range=(1, 2), stop_words=STOP, min_df=C.VECTORIZER_MIN_DF)

def build(seed, embedding_model=None, hdb=None, umap_kw=None, prediction_data=True):
    return BERTopic(embedding_model=embedding_model,
                    umap_model=UMAP(**(umap_kw or umap_params()), random_state=seed),
                    hdbscan_model=HDBSCAN(**(hdb or hdbscan_params()), prediction_data=prediction_data),
                    vectorizer_model=vectorizer(),
                    representation_model=MaximalMarginalRelevance(diversity=C.MMR_DIVERSITY),
                    top_n_words=C.TOP_N_WORDS, calculate_probabilities=False, verbose=False)
