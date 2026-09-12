"""Stage 6 - calibration and fusion. Where the ranking spread comes from.

Raw cosines between two same-domain documents sit in ~0.45-0.80, so without
calibration every candidate lands within a few points of every other.

Two fixes. First, z-normalise each requirement across the cohort: we're ranking THIS
pool against THIS job, so a requirement everyone satisfies carries no information and
should stop separating people, while one that splits the pool should dominate.

Second, a gate -- a required requirement with no keyword hit and weak semantic
evidence is a hard miss, and hard misses scale the score down. The hard-miss list is
also what the explanations use for "missing skills".
"""
from __future__ import annotations

import numpy as np

from .. import config
from ..schemas import CandidateScore, Evidence, JobDescription, ParsedResume
from .keyword import KeywordMatcher
from .semantic import SemanticMatcher


def calibrate(raw: np.ndarray) -> np.ndarray:
    """Z-normalise each row (requirement) across the cohort, squash with a logistic.

    A requirement where everyone scores the same maps to a flat 0.5 rather than being
    amplified into noise by a tiny std.
    """
    mean = raw.mean(axis=1, keepdims=True)
    std = raw.std(axis=1, keepdims=True)
    flat = std < 1e-6
    z = np.where(flat, 0.0, (raw - mean) / np.where(flat, 1.0, std))
    return 1.0 / (1.0 + np.exp(-config.Z_LOGISTIC_SCALE * z))


def fuse(jd: JobDescription, resumes: list[ParsedResume],
         verbose: bool = True) -> list[CandidateScore]:
    """Run both channels and produce the ranked list."""
    kw = KeywordMatcher(jd, resumes)
    sem = SemanticMatcher(jd, resumes, verbose=verbose)

    sem_raw = sem.raw_matrix()
    sem_cal = calibrate(sem_raw)
    argmax_units = sem.argmax_units()

    weights = np.array([r.weight for r in jd.requirements], dtype=np.float32)
    total_w = float(weights.sum()) or 1.0

    scored: list[CandidateScore] = []
    for j, resume in enumerate(resumes):
        evidence: list[Evidence] = []
        hits: dict[str, float] = {}

        for i, req in enumerate(jd.requirements):
            hit, terms = kw.hit(req, resume)
            hits[req.id] = hit
            unit_text, unit_section = argmax_units[i][j]
            evidence.append(Evidence(
                requirement_id=req.id,
                requirement_text=req.text,
                required=req.required,
                kw_hit=hit,
                kw_terms=terms,
                sem_raw=float(sem_raw[i, j]),
                sem_cal=float(sem_cal[i, j]),
                unit_text=unit_text,
                unit_section=unit_section,
            ))

        keyword_score, bm25 = kw.score(resume, hits)
        semantic_score = float((sem_cal[:, j] * weights).sum() / total_w)

        scored.append(CandidateScore(
            resume=resume,
            keyword_score=keyword_score,
            semantic_score=semantic_score,
            bm25_norm=bm25,
            evidence=evidence,
        ))

    apply_final(scored)
    return scored


def apply_final(scored: list[CandidateScore]) -> list[CandidateScore]:
    """Fuse both channels, apply the gate, sort, rank.

    Separate from fuse() so calibrate.py can re-score under different weights without
    re-embedding -- that's what makes the grid search take seconds.
    """
    for cand in scored:
        base = (config.W_KEYWORD * cand.keyword_score
                + config.W_SEMANTIC * cand.semantic_score)
        penalty = max(
            config.HARD_MISS_FLOOR,
            1.0 - config.HARD_MISS_PENALTY * len(cand.hard_misses),
        )
        cand.final_score = round(base * penalty * 100, 2)

    scored.sort(key=lambda c: -c.final_score)
    for i, cand in enumerate(scored, 1):
        cand.rank = i
    return scored
