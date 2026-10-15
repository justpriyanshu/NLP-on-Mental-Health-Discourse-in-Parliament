"""Rajya Sabha debates API (discovered 2026-09): integration.rajyasabha.digital
Single source of truth for every script. Anything reported in the methods
chapter (thresholds, models, seeds, periods) should be read from here."""
API = "https://integration.rajyasabha.digital/rsdebate/api/restv4"
LIST_URL = f"{API}/fetch/all"
FACET_URL = f"{API}/field/browse"
HEADERS = {"User-Agent": "Mozilla/5.0 (research scraper)", "Referer": "https://sansad.in/",
           "Origin": "https://sansad.in", "Accept": "application/json"}
COLLECTION = "(1,2)"
PART2 = "Part 2 (Other Than Question And Answer)"
FROM_YEAR, TO_YEAR = 2000, 2025
ROWS = 1000
REQUEST_DELAY_S = 1.0
FILTER_PARAMS = {"type": PART2, "year": "{year}"}

# ---- Text extraction (03) ---------------------------------------------------
# Hybrid: native text layer (PD_/TQ_) via pdftotext when it passes the quality
# gate, otherwise OCR of the scanned counterpart (ID_/IQ_). Both files describe
# the SAME debate, so only one representation per record is kept.
OCR_DPI = 300                    # Tesseract's recommended resolution (150 in v1)
OCR_PSM = 6
TEXT_MIN_FUNCTION_WORD_SHARE = 0.03   # real English prose has ~0.25; corrupted streams ~0
TEXT_MAX_JUNK_SHARE = 0.15            # private-use / replacement chars => broken font encoding
TEXT_MIN_PAGE_PASS_RATE = 0.90        # share of PD pages that must pass before PD is trusted
DUPLICATE_PART_SIMILARITY = 0.90      # parts of one record this similar are treated as copies

# ---- Segmentation (04) ------------------------------------------------------
MIN_WORDS, CHUNK_WORDS = 30, 500      # chunk = <=500 words WITHIN a speaker turn
DEVANAGARI_HI_SHARE = 0.50            # >= -> lang 'hi'
DEVANAGARI_MIXED_SHARE = 0.05         # >= -> lang 'mixed' (still sent to MT)
ROMAN_HINDI_HITS_PER_100 = 3.0        # function-word density that flags Romanised Hindi

# ---- Machine translation (05) -----------------------------------------------
# 'indictrans2' = ai4bharat/indictrans2 (open, citable, reproducible; default)
# 'google'      = deep_translator web endpoint (v1 behaviour; not reproducible)
MT_BACKEND = "indictrans2"
MT_MODEL = "ai4bharat/indictrans2-indic-en-dist-200M"   # or ...-indic-en-1B if GPU allows
MT_BATCH = 16
MT_SENT_DEVANAGARI_SHARE = 0.30       # only sentences this Hindi are translated; English kept verbatim

LEXICON = ["mental health", "mental illness", "mentally ill", "psychiatric", "psychiatry",
           "psychological", "depression", "anxiety", "suicide", "suicides", "self-harm",
           "trauma", "counselling", "counseling", "drug addiction", "substance abuse",
           "narcotic", "psychotropic", "alcoholism", "autism", "cerebral palsy",
           "intellectual disability", "mental retardation", "lunacy", "NIMHANS", "ragging"]

# ---- Topic model (06) -------------------------------------------------------
EMBEDDING_MODE = "translate"          # or "multilingual"; run both, compare evaluation.json
EMBED_MODELS = {"translate": "all-mpnet-base-v2",
                "multilingual": "paraphrase-multilingual-mpnet-base-v2"}
UMAP_KW = dict(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine")
# min_cluster_size is corpus-size dependent. Leave None to use chosen_params.json
# written by 06b_sensitivity.py; the literal below is only the fallback.
HDBSCAN_KW = dict(min_cluster_size=None, min_samples=None, metric="euclidean",
                  cluster_selection_method="eom")
HDBSCAN_FALLBACK = dict(min_cluster_size=50, min_samples=10)
SENSITIVITY_GRID = {"min_cluster_size": [20, 35, 50, 75, 100, 150],
                    "min_samples": [5, 10, 20],
                    "n_neighbors": [15, 30]}
TOP_N_WORDS = 20
VECTORIZER_MIN_DF = 5                 # drops OCR one-offs from c-TF-IDF vocabulary
MMR_DIVERSITY = 0.3                   # representation model; reduces near-duplicate top terms
SEEDS = [42, 1, 7, 13, 99]
# NOTE: BERTopic's topics_over_time(evolution_tuning=True) averages each bin's
# c-TF-IDF with the previous bin's -> a FIXED alpha of 0.5. It is not tunable.
STOP_WORDS = ["sir", "hon", "shri", "shrimati", "madam", "honourable", "minister",
              "government", "house", "member", "members", "chairman", "deputy",
              "speaker", "bill", "said", "today", "time", "country", "india", "want",
              "like", "know", "say", "also", "mr", "dr", "kumari", "ji", "rajya", "sabha",
              "clause", "page", "line", "amendment", "moved", "question", "put", "motion",
              "adopted", "vice", "kindly", "please", "thank", "sitting", "yes", "no"]

# ---- Analysis (07-11) -------------------------------------------------------
MH_GROUPS = ["DirectMH", "SuicideCrisis", "SubstanceAbuse", "Disability",
             "SocialDet", "GenderViolence", "Other"]
PERIODS = {"Pre-MH Legislation": (2000, 2004), "Agrarian Crisis": (2005, 2009),
           "Institutional Reform": (2010, 2014), "Mental Healthcare Act": (2015, 2019),
           "Pandemic & after": (2020, 2025)}
N_PERMUTATIONS = 5000
DOC_VALIDATION_N = 200                # chunks for the document-level annotation sample
