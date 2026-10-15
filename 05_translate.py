"""05 — Hindi->English MT with an open, citable model (replaces Google web endpoint).

Why: deep_translator scrapes an undocumented Google endpoint — results are not
reproducible and the model is unnamed, which an examiner will query. Default is
AI4Bharat IndicTrans2 (Gala et al., TMLR 2023), run locally; set
config.MT_BACKEND='google' to reproduce v1 behaviour.

Only sentences that are actually Hindi (Devanagari share >= MT_SENT_DEVANAGARI_SHARE)
are translated; English sentences inside a mixed chunk are kept verbatim, so MT
noise cannot leak into text that was already English. Translations are cached
in mt_cache.jsonl (resumable; identical sentences translated once).

Skipped entirely when config.EMBEDDING_MODE == 'multilingual' (english == text).
Output: bertopic_corpus.csv"""
import re, json, os, time, hashlib
import pandas as pd
from tqdm import tqdm
import config as C

DEV = re.compile(r"[ऀ-ॿ]"); LATIN = re.compile(r"[A-Za-z]")
SENT = re.compile(r"(?<=[.!?।])\s+")
CACHE = "mt_cache.jsonl"

def dev_share(s):
    d = len(DEV.findall(s)); l = len(LATIN.findall(s)); return d / max(1, d + l)

def load_cache():
    c = {}
    if os.path.exists(CACHE):
        for line in open(CACHE, encoding="utf-8"):
            try: r = json.loads(line); c[r["k"]] = r["t"]
            except Exception: pass
    return c

def key(s): return hashlib.sha1(s.encode("utf-8")).hexdigest()

class IndicTrans2:
    def __init__(self):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from IndicTransToolkit.processor import IndicProcessor
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(C.MT_MODEL, trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(C.MT_MODEL, trust_remote_code=True).to(self.dev).eval()
        self.ip = IndicProcessor(inference=True)
    def translate(self, sents):
        out = []
        for i in range(0, len(sents), C.MT_BATCH):
            b = self.ip.preprocess_batch(sents[i:i + C.MT_BATCH], src_lang="hin_Deva", tgt_lang="eng_Latn")
            enc = self.tok(b, truncation=True, padding="longest", return_tensors="pt", max_length=256).to(self.dev)
            with self.torch.no_grad():
                gen = self.model.generate(**enc, num_beams=4, max_length=256, num_return_sequences=1)
            dec = self.tok.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            out += self.ip.postprocess_batch(dec, lang="eng_Latn")
        return out

class Google:
    def __init__(self):
        from deep_translator import GoogleTranslator
        self.tr = GoogleTranslator(source="auto", target="en")
    def translate(self, sents):
        out = []
        for s in sents:
            for attempt in range(5):
                try: out.append(self.tr.translate(s) or s); break
                except Exception: time.sleep(2 ** attempt)
            else: out.append(s)
        return out

def main():
    df = pd.read_csv("segments.csv")
    df = df[df.n_words >= C.MIN_WORDS].copy()
    df["english"] = df["text"].astype(str)
    df["mt_sentences"] = 0

    if C.EMBEDDING_MODE == "translate":
        todo = df.index[df.lang.isin(["hi", "mixed", "hi-roman"])]
        cache = load_cache()
        # collect every Hindi sentence once
        sent_lists = {}
        for i in todo:
            sents = SENT.split(df.at[i, "text"])
            sent_lists[i] = sents
        pending = sorted({s for sents in sent_lists.values() for s in sents
                          if dev_share(s) >= C.MT_SENT_DEVANAGARI_SHARE and key(s) not in cache})
        print(f"{len(todo)} chunks flagged; {len(pending)} unique Hindi sentences to translate "
              f"({len(cache)} cached) with backend={C.MT_BACKEND}")
        if pending:
            mt = IndicTrans2() if C.MT_BACKEND == "indictrans2" else Google()
            with open(CACHE, "a", encoding="utf-8") as fc:
                B = 256
                for j in tqdm(range(0, len(pending), B)):
                    batch = pending[j:j + B]
                    for s, t in zip(batch, mt.translate(batch)):
                        cache[key(s)] = t; fc.write(json.dumps({"k": key(s), "t": t}, ensure_ascii=False) + "\n")
        for i, sents in sent_lists.items():
            out, n = [], 0
            for s in sents:
                if dev_share(s) >= C.MT_SENT_DEVANAGARI_SHARE:
                    out.append(cache.get(key(s), s)); n += 1
                else: out.append(s)
            df.at[i, "english"] = " ".join(out); df.at[i, "mt_sentences"] = n
        # Romanised Hindi has no Devanagari; IndicTrans2 needs Devanagari input -> left as is, flagged
        df["mt_note"] = ""
        df.loc[df.lang == "hi-roman", "mt_note"] = "romanised_hindi_untranslated"

    pat = re.compile(r"\b(" + "|".join(re.escape(k) for k in C.LEXICON) + r")\b", re.I)
    df["lexicon_hit"] = [bool(pat.search(e) or pat.search(t)) for e, t in zip(df.english.astype(str), df.text.astype(str))]
    df.to_csv("bertopic_corpus.csv", index=False)
    print(len(df), "chunks;", round(df.lexicon_hit.mean(), 3), "share with lexicon hit;",
          int(df.mt_sentences.sum()), "sentences machine-translated")

if __name__ == "__main__":
    main()
