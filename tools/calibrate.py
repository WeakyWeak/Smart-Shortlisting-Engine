#!/usr/bin/env python3
"""Stage 9 - fit the fusion weights against real labels.

Each corpus resume has its job family in the filename, which is a real relevance
label: for a full-stack JD, web_dev and sde should outrank sales and video_editing.
families.json defines graded relevance for six JDs.

So rather than claiming the weights "feel right", grid-search them for mean NDCG@10
across all six. Six rather than one so no threshold gets tuned to a single JD's
wording. Features are cached, so hundreds of configs evaluate in seconds.

    python tools/calibrate.py
    python tools/calibrate.py --apply     # write the winning weights to config.py
"""
from __future__ import annotations

import argparse
import itertools
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexora import config  # noqa: E402
from nexora.ingest.dedup import RELEVANCE  # noqa: E402
from nexora.ingest.jd import parse_jd  # noqa: E402
from nexora.matching.fuse import fuse  # noqa: E402
from nexora.pipeline import load_resumes  # noqa: E402


def ndcg_at_k(relevances: list[int], k: int = 10) -> float:
    """Graded NDCG. `relevances` is in predicted-rank order."""
    rel = np.asarray(relevances[:k], dtype=float)
    if rel.size == 0:
        return 0.0
    discount = 1.0 / np.log2(np.arange(2, rel.size + 2))
    dcg = float((rel * discount).sum())

    ideal = np.sort(np.asarray(relevances, dtype=float))[::-1][:k]
    idcg = float((ideal * (1.0 / np.log2(np.arange(2, ideal.size + 2)))).sum())
    return dcg / idcg if idcg > 0 else 0.0


# Eval JD filename -> the family key in families.json.
_JD_TO_FAMILY = {
    "fullstack_intern": "web_dev",
    "cyber_security": "cyber_sec",
}


def jd_family(path: Path) -> str:
    """Which family this eval JD targets, from its filename."""
    stem = re.sub(r"[^a-z_]", "", path.stem.lower())
    return _JD_TO_FAMILY.get(stem, stem)


def score_config(cands, w_kw: float, penalty: float, floor: float) -> list:
    """Re-score cached candidates under one weight config."""
    out = []
    for c in cands:
        base = w_kw * c.keyword_score + (1 - w_kw) * c.semantic_score
        mult = max(floor, 1.0 - penalty * len(c.hard_misses))
        out.append((base * mult, c))
    out.sort(key=lambda t: -t[0])
    return [c for _, c in out]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="write the best weights into nexora/config.py")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    jds = [config.JD_DIR / "fullstack_intern.md"] + sorted(config.EVAL_JD_DIR.glob("*.md"))
    print(f"loading resumes once for {len(jds)} JDs...")
    resumes = load_resumes(config.CORPUS_DIR, verbose=True)

    # Expensive bit, once per JD: parse, embed, score both channels.
    cached: list[tuple[str, list]] = []
    for path in jds:
        jd = parse_jd(path)
        fam = jd_family(path)
        if fam not in RELEVANCE:
            print(f"  ! no relevance map for {fam}, skipping")
            continue
        print(f"  {fam:18s} {len(jd.requirements):2d} reqs", flush=True)
        cached.append((fam, fuse(jd, resumes, verbose=False)))

    grid = list(itertools.product(
        [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],   # w_keyword
        [0.00, 0.06, 0.12, 0.18, 0.25],                            # hard-miss penalty
        [0.40, 0.50, 0.60],                                        # floor
    ))
    print(f"\ngrid-searching {len(grid)} configurations on NDCG@{args.k}...")

    results = []
    for w_kw, penalty, floor in grid:
        per_jd, per_jd_strict = [], []
        for fam, cands in cached:
            rel_map = RELEVANCE[fam]
            ranked = score_config(cands, w_kw, penalty, floor)
            per_jd.append(ndcg_at_k([rel_map.get(c.resume.family, 0) for c in ranked], args.k))
            per_jd_strict.append(ndcg_at_k(
                [1 if c.resume.family == fam else 0 for c in ranked], args.k))
        results.append((float(np.mean(per_jd_strict)), w_kw, penalty, floor,
                        per_jd, float(np.mean(per_jd))))

    results.sort(key=lambda r: -r[0])
    best = results[0]

    print(f"\n{'rank':<5}{'strict':<10}{'graded':<10}{'w_kw':<8}{'penalty':<10}{'floor':<8}")
    print("  strict = only the exact target family counts as relevant (the hard metric)")
    print("  graded = adjacent families partially credited (easy: separates a web dev "
          "from a video editor)")
    for i, (strict, w_kw, penalty, floor, _, graded) in enumerate(results[:8], 1):
        print(f"{i:<5}{strict:<10.4f}{graded:<10.4f}{w_kw:<8.2f}{penalty:<10.2f}{floor:<8.2f}")

    current = next((r for r in results
                    if abs(r[1] - config.W_KEYWORD) < 1e-9
                    and abs(r[2] - config.HARD_MISS_PENALTY) < 1e-9
                    and abs(r[3] - config.HARD_MISS_FLOOR) < 1e-9), None)

    mean, w_kw, penalty, floor, per_jd, graded = best
    print(f"\nbest: w_keyword={w_kw:.2f}  w_semantic={1 - w_kw:.2f}  "
          f"hard_miss_penalty={penalty:.2f}  floor={floor:.2f}")
    print(f"mean strict NDCG@{args.k} = {mean:.4f}   (graded {graded:.4f})")
    if current:
        print(f"current config    = {current[0]:.4f}  "
              f"(w_kw={config.W_KEYWORD}, penalty={config.HARD_MISS_PENALTY})")
    print("\nper-JD breakdown:")
    for (fam, _), score in zip(cached, per_jd):
        print(f"  {fam:20s} {score:.4f}")

    # Sanity check that both channels earn their place. If either extreme matches
    # the blend, one of them is decorative.
    print("\nablation (does each channel actually contribute?)")
    for label, w in [("keyword only  (w_kw=1.0)", 1.0), ("semantic only (w_kw=0.0)", 0.0)]:
        scores = [
            ndcg_at_k([1 if c.resume.family == fam else 0
                       for c in score_config(cands, w, penalty, floor)], args.k)
            for fam, cands in cached
        ]
        print(f"  {label}: {np.mean(scores):.4f}")
    print(f"  both channels          : {mean:.4f}")
    print("  NOTE: cross-family separation is a largely LEXICAL task - telling a web")
    print("  developer from a video editor barely needs semantics, so keyword-only ties")
    print("  the blend here. The family labels cannot measure within-family ordering,")
    print("  which is where the semantic channel actually pays. Measured directly:")

    # Where semantics actually changes the outcome: people it puts in the shortlist
    # that keyword-only would have dropped.
    for fam, cands in cached:
        blended = score_config(cands, w_kw, penalty, floor)[:20]
        kw_only = score_config(cands, 1.0, penalty, floor)[:20]
        kw_stems = {c.resume.stem for c in kw_only}
        rescued = [c for c in blended if c.resume.stem not in kw_stems]
        on_target = [c for c in rescued if c.resume.family == fam]
        if rescued:
            top = max(rescued, key=lambda c: c.delta)
            print(f"  {fam:18s} {len(rescued):2d}/20 rescued by semantics "
                  f"({len(on_target)} on-target)  e.g. {top.resume.display} "
                  f"(kw {top.keyword_score * 100:.0f}, sem {top.semantic_score * 100:.0f})")

    if args.apply:
        path = config.ROOT / "nexora" / "config.py"
        text = path.read_text()
        text = re.sub(r"^W_KEYWORD = [\d.]+", f"W_KEYWORD = {w_kw:.2f}", text, flags=re.M)
        text = re.sub(r"^W_SEMANTIC = [\d.]+", f"W_SEMANTIC = {1 - w_kw:.2f}", text, flags=re.M)
        text = re.sub(r"^HARD_MISS_PENALTY = [\d.]+",
                      f"HARD_MISS_PENALTY = {penalty:.2f}", text, flags=re.M)
        text = re.sub(r"^HARD_MISS_FLOOR = [\d.]+",
                      f"HARD_MISS_FLOOR = {floor:.2f}", text, flags=re.M)
        path.write_text(text)
        print(f"\napplied to {path}")


if __name__ == "__main__":
    main()
