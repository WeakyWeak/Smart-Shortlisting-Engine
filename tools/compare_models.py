#!/usr/bin/env python3
"""A/B embedding and reranking models against the calibration corpus.

We have labels, so "is the bigger model better?" is measurable rather than arguable.
Runs each model over the same six JDs and reports NDCG@10 and wall-clock cost.

    python tools/compare_models.py                 # bi-encoders
    python tools/compare_models.py --rerankers     # cross-encoders
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexora import config  # noqa: E402
from nexora.ingest.dedup import RELEVANCE  # noqa: E402
from nexora.ingest.jd import parse_jd  # noqa: E402
from nexora.matching.fuse import fuse  # noqa: E402
from nexora.matching.rerank import rerank  # noqa: E402
from nexora.pipeline import load_resumes  # noqa: E402
from tools.calibrate import jd_family, ndcg_at_k  # noqa: E402

BI_ENCODERS = [
    "BAAI/bge-small-en-v1.5",           # 33M  - current
    "BAAI/bge-large-en-v1.5",           # 335M - same family, drop-in
]
RERANKERS = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",  # 22M  - current
    "BAAI/bge-reranker-base",                # 278M
]


def evaluate(resumes, jds, do_rerank: bool) -> tuple[float, float, float]:
    """Mean strict and graded NDCG@10 across the eval JDs, plus elapsed seconds."""
    strict, graded = [], []
    started = time.time()
    for path in jds:
        jd = parse_jd(path)
        fam = jd_family(path)
        if fam not in RELEVANCE:
            continue
        scored = fuse(jd, resumes, verbose=False)
        if do_rerank:
            scored = rerank(scored, verbose=False)
        rel = RELEVANCE[fam]
        strict.append(ndcg_at_k([1 if c.resume.family == fam else 0 for c in scored], 10))
        graded.append(ndcg_at_k([rel.get(c.resume.family, 0) for c in scored], 10))
    return float(np.mean(strict)), float(np.mean(graded)), time.time() - started


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerankers", action="store_true",
                    help="vary the cross-encoder instead of the bi-encoder")
    args = ap.parse_args()

    jds = sorted(config.EVAL_JD_DIR.glob("*.md"))
    print(f"loading calibration corpus ({len(jds)} JDs)...")
    resumes = load_resumes(config.CORPUS_DIR, verbose=True)

    models = RERANKERS if args.rerankers else BI_ENCODERS
    label = "cross-encoder" if args.rerankers else "bi-encoder"
    print(f"\n{'model':<42}{'strict':<10}{'graded':<10}{'seconds':<10}")
    print("-" * 72)

    results = []
    for name in models:
        if args.rerankers:
            config.CROSS_ENCODER = name
        else:
            config.BI_ENCODER = name
        try:
            strict, graded, secs = evaluate(resumes, jds, do_rerank=args.rerankers)
        except Exception as exc:  # noqa: BLE001
            print(f"{name:<42}FAILED: {type(exc).__name__}: {str(exc)[:80]}")
            continue
        results.append((name, strict, graded, secs))
        print(f"{name:<42}{strict:<10.4f}{graded:<10.4f}{secs:<10.1f}")

    if len(results) < 2:
        return
    base, best = results[0], max(results, key=lambda r: r[1])
    gain = best[1] - base[1]
    slower = best[3] / base[3] if base[3] else 1.0
    print(f"\nbest {label}: {best[0]}")
    print(f"  strict NDCG@10 {base[1]:.4f} -> {best[1]:.4f}  ({gain:+.4f})")
    print(f"  cost: {slower:.1f}x the runtime of {base[0]}")
    if gain < 0.005:
        print("\n  VERDICT: no meaningful gain. Keep the smaller model - it is faster,")
        print("  and an unjustified dependency is a liability during a live demo.")
    else:
        print(f"\n  VERDICT: real gain. Set NEXORA_{'CE' if args.rerankers else 'BI'}="
              f"{best[0]} or edit nexora/config.py.")


if __name__ == "__main__":
    main()
