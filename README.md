# Rajya Sabha MH corpus — pipeline v2 (sansad.in API, full-corpus design)

Read `REVISION_NOTES.md` for what changed and why; this file is the run order.

## Install
    pip install -r requirements.txt
    sudo apt install tesseract-ocr tesseract-ocr-hin poppler-utils

## Run order
| step | command | output |
|---|---|---|
| 00 | `python probe_api.py` | confirms FILTER_PARAMS, year/type facets |
| 01 | `python 01_scrape_metadata.py` | `metadata_ledger.csv` (now has `file_kinds`) |
| 02 | `python 02_download_pdfs.py` | `pdfs/`, `download_log.csv` (resumable; `DOWNLOAD_KINDS` picks text/image variants) |
| 03 | `python 03_extract_text.py` | `raw_text.csv`, `extraction_log.csv` |
|    | `python eval_ocr.py --sample 40` → transcribe → `python eval_ocr.py --score` | `ocr_eval_report.csv` |
| 04 | `python 04_segment.py` | `segments.csv`, `segmentation_stats.json` |
| 05 | `python 05_translate.py` | `bertopic_corpus.csv`, `mt_cache.jsonl` |
|    | `python eval_translation.py` → rater fills sheet → `--score` | `mt_similarity.csv`, fidelity |
| 06 | `python 06_bertopic.py --embed-only` | `embeddings.npy` |
| 06b| `python 06b_sensitivity.py` (optionally `--choose N`) | `sensitivity_grid.csv`, `chosen_params.json` |
| 06 | `python 06_bertopic.py` | `model/`, `topics.csv`, `doc_topics.csv`, `hierarchical_topics.csv`, `topic_tree.txt`, `topics_over_time.csv`, `model_stats.json` |
| 07 | `--build` → two annotators fill `group_A`/`group_B` → `--kappa` (after `group_final`) | `annotation_sheet.csv`, `topic_groups.csv` |
| 07 | `--doc-sample` → annotators fill → `--doc-kappa` | document-level validation |
| 08 | `python 08_evaluate.py` | `evaluation.json` |
| 09 | `python 09_cooccurrence.py` | `cooccurrence_debate.csv`, `cooccurrence_speech.csv` |
| 10 | `python 10_temporal.py` | `attention_share_by_year.csv`, `changepoints.json`, `period_composition.csv` |
| 11 | `python 11_speakers.py` (optional `members.csv`: name, party, state) | `speakers_mh.csv`, `role_by_group.csv`, `state_by_group.csv` |
| 12 | `python 12_figures.py` | `figures/*.png` |

Run 05→08 twice (`EMBEDDING_MODE = "translate"` then `"multilingual"`) and put both
`evaluation.json` results in the thesis to justify the choice. Every threshold or
model named in the methods chapter lives in `config.py`.

## Scale
Full mode is ~20x the v1 corpus. 03 now OCRs only the pages whose native text
layer fails, so most English pages cost a `pdftotext` call instead of 1–3 s of
Tesseract; budget OCR time from `extraction_log.csv` (`pages_ocr`) after a first
batch. MT with IndicTrans2-200M runs at ~10–30 sentences/s on CPU; the cache makes
it resumable.
