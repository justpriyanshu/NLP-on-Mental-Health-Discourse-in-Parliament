"""Coherence / diversity helpers shared by 06b and 08.
Coherence is computed on UNIGRAMS only: BERTopic topic words include bigrams
('mental health') that a whitespace-tokenised reference corpus never contains,
so v1 silently dropped them and scored topics on whatever unigrams survived."""
import numpy as np
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
import tm_common as T

class Reference:
    def __init__(self, docs):
        self.tokenized = [[w for w in d.lower().split() if w.isalpha() and w not in T.STOP] for d in docs]
        self.dictionary = Dictionary(self.tokenized)

    def topic_unigrams(self, words, n=10):
        out = []
        for w in words:
            if " " in w: continue
            if w in self.dictionary.token2id and w not in out: out.append(w)
            if len(out) == n: break
        return out

    def score(self, topic_words, metrics=("c_npmi", "c_v")):
        tw = [self.topic_unigrams(ws) for ws in topic_words]
        tw = [t for t in tw if len(t) >= 3]
        res = {"n_scored_topics": len(tw)}
        for m in metrics:
            res[m] = float(CoherenceModel(topics=tw, texts=self.tokenized, dictionary=self.dictionary,
                                          coherence=m, processes=1).get_coherence()) if tw else float("nan")
        flat = [w for t in tw for w in t]
        res["diversity"] = len(set(flat)) / max(1, len(flat))
        return res

def bertopic_words(model):
    return [[w for w, _ in model.get_topic(t)] for t in model.get_topics() if t != -1]
