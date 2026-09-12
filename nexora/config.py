"""All tunable numbers. calibrate.py rewrites the weights here, so don't inline them
elsewhere."""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
# The resumes we actually rank.
RESUME_DIR = DATA_DIR / "resumes"
# Practice corpus for calibration only -- job family is in each filename, which
# gives us relevance labels. Never ranked.
CORPUS_DIR = DATA_DIR / "corpus"
JD_DIR = DATA_DIR / "jd"
EVAL_JD_DIR = JD_DIR / "eval"
OUT_DIR = ROOT / "out"
RANKING_JSON = OUT_DIR / "ranking.json"
RESOURCES = Path(__file__).resolve().parent / "resources"

# --------------------------------------------------------------------------- models
# Override to try other models: NEXORA_BI=... python tools/compare_models.py
# bge-large: +0.048 NDCG@10 over bge-small, ~7s slower on 18 resumes. Worth it.
BI_ENCODER = os.environ.get("NEXORA_BI", "BAAI/bge-large-en-v1.5")
CROSS_ENCODER = os.environ.get("NEXORA_CE", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# bge needs this on the query side only. Leaving it off quietly hurts scores.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def device() -> str:
    """Picked at runtime so a CPU-only install needs no code change."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# --------------------------------------------------------------------------- Stage 1: ingestion
# Same resume shows up in several formats; pick the cleanest.
FORMAT_PREFERENCE = ["xml", "docx", "txt", "pdf"]
OCR_MIN_CHARS = 100          # below this, a PDF is probably scanned -> OCR fallback
MIN_UNIT_CHARS = 25          # evidence units shorter than this are noise
MAX_UNIT_CHARS = 400         # longer ones get split; keeps them inside the encoder window

# --------------------------------------------------------------------------- Stage 2: parsing
HEADER_FUZZ_THRESHOLD = 85   # rapidfuzz ratio for matching a line to a section header
SKILL_FUZZ_THRESHOLD = 88    # rapidfuzz ratio for typo-tolerant skill matching
SKILL_FUZZ_MIN_LEN = 5       # don't fuzzy-match short tokens ("go", "r", "c") -- too noisy

# --------------------------------------------------------------------------- Stage 4: keyword
KW_SKILL_HIT_WEIGHT = 0.70   # per-requirement exact/alias skill overlap
KW_BM25_WEIGHT = 0.30        # cohort-normalised BM25 of the JD against each resume

# --------------------------------------------------------------------------- Stage 5/6: semantic + fusion
REQUIRED_WEIGHT = 1.0        # weight of a "required" JD requirement
PREFERRED_WEIGHT = 0.5       # weight of a "preferred"/nice-to-have requirement

# z-normalise each requirement across the cohort. Without this every score lands
# in a 10-point band.
Z_LOGISTIC_SCALE = 1.0       # logistic steepness applied to the z-score
SEM_TOP_K_UNITS = 2          # average the top-k matching units, not just the single max

W_KEYWORD = 0.50
W_SEMANTIC = 0.50

# Required requirement with no keyword hit and weak semantic evidence = hard miss.
# Also gives us the "missing skills" list for free.
HARD_MISS_SEM_THRESHOLD = 0.45
HARD_MISS_PENALTY = 0.12     # multiplicative, per hard miss
HARD_MISS_FLOOR = 0.50       # never scale a candidate below half

# Either channel clearing this counts as a match.
MATCH_SEM_THRESHOLD = 0.55
MATCH_KW_THRESHOLD = 0.50

# --------------------------------------------------------------------------- Stage 7: rerank
RERANK_TOP_N = 10            # candidates re-scored by the cross-encoder
RERANK_BLEND = 0.50          # how much the cross-encoder moves the fused score

# --------------------------------------------------------------------------- Stage 8: explanations
TOP_K_EXPLAIN = 3
GROQ_MODEL = "openai/gpt-oss-120b"   # verified against this account's model list
GROQ_TEMPERATURE = 0.2
# gpt-oss burns tokens on reasoning first, so a low max_tokens returns "".
GROQ_REASONING_EFFORT = "low"
GROQ_TIMEOUT = 20.0
GROQ_MAX_RETRIES = 2

# --------------------------------------------------------------------------- bonus: bias
# As a fraction of the pool -- a fixed count never fires on a small cohort.
BIAS_NARROW_MIN_FRACTION = 0.15
BIAS_NARROW_MIN_ABSOLUTE = 2
# Raw cosine, not calibrated -- a fixed threshold on the calibrated score flags
# everything.
BIAS_NARROW_SEM_THRESHOLD = 0.62
BIAS_NARROW_TOP_FRACTION = 0.35  # only count otherwise-competitive candidates

# --------------------------------------------------------------------------- demo
DEMO_COHORT_SIZE = 18
