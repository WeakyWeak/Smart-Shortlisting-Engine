"""Stage 7 - cross-encoder re-rank of the top candidates.

Stages 5-6 use a bi-encoder: requirement and resume are embedded separately, so the
model never sees them together. A cross-encoder takes both as one input and attends
across them -- more accurate, but a forward pass per pair instead of a lookup. Hence
top-N only: retrieve cheap, re-rank precise.

We feed it the requirement against the best evidence sentence, not the whole resume.
That's its MS MARCO training shape, and a full resume would truncate anyway.
"""
from __future__ import annotations

import numpy as np

from .. import config
from ..schemas import CandidateScore
from .fuse import apply_final

_MODEL = None
_LOADED: str | None = None


def get_model():
    global _MODEL, _LOADED
    if _MODEL is None or _LOADED != config.CROSS_ENCODER:
        from sentence_transformers import CrossEncoder
        _MODEL = CrossEncoder(config.CROSS_ENCODER, device=config.device())
        _LOADED = config.CROSS_ENCODER
    return _MODEL


def rerank(scored: list[CandidateScore], verbose: bool = True) -> list[CandidateScore]:
    """Re-score the top N and re-sort. Mutates `scored`."""
    top = scored[:config.RERANK_TOP_N]
    if not top:
        return scored

    # One flat batch of (requirement, best-evidence) pairs.
    pairs: list[tuple[str, str]] = []
    index: list[tuple[int, int]] = []
    for ci, cand in enumerate(top):
        for ei, ev in enumerate(cand.evidence):
            if ev.required and ev.unit_text:
                pairs.append((ev.requirement_text, ev.unit_text))
                index.append((ci, ei))

    if not pairs:
        return scored

    if verbose:
        print(f"  cross-encoder re-ranking top {len(top)} on {len(pairs)} pairs...")
    raw = np.asarray(get_model().predict(pairs, show_progress_bar=False), dtype=np.float32)

    # These emit unbounded logits; squash to [0,1] before blending.
    probs = 1.0 / (1.0 + np.exp(-raw))
    for (ci, ei), p in zip(index, probs):
        top[ci].evidence[ei].ce_score = float(p)

    for cand in top:
        ce = [e.ce_score for e in cand.evidence if e.ce_score is not None]
        if not ce:
            continue
        # Blend, don't replace -- this refines the ordering, it doesn't get to
        # overrule the keyword channel on its own.
        cand.semantic_score = float(
            (1 - config.RERANK_BLEND) * cand.semantic_score
            + config.RERANK_BLEND * float(np.mean(ce))
        )
        cand.reranked = True

    apply_final(scored)
    return scored
