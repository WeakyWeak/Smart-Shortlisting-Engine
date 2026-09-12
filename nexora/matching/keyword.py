"""Stage 4 - keyword channel. Does the resume literally contain what the JD names?

Two signals: per-requirement skill hits (precision -- stops vaguely related
experience from satisfying a named tool) and cohort BM25 (recall -- vocabulary
overlap our taxonomy doesn't know about).

BM25 over TF-IDF for term saturation (the tenth "React" adds nothing) and length
normalisation (these resumes run 2k-5.6k chars).
"""
from __future__ import annotations

import re

from .. import config
from ..schemas import JobDescription, ParsedResume, Requirement

_TOKEN_RE = re.compile(r"[a-z0-9+#.]+")
_STOP = {
    "and", "or", "the", "a", "an", "of", "to", "in", "for", "with", "on", "at", "by",
    "is", "are", "be", "as", "from", "that", "this", "it", "we", "you", "our", "your",
    "will", "can", "has", "have", "using", "use", "work", "working", "experience",
    "knowledge", "strong", "good", "familiar", "familiarity", "ability", "skills",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOP and len(t) > 1]


class KeywordMatcher:
    """Fits BM25 over the cohort once, then scores each resume."""

    def __init__(self, jd: JobDescription, resumes: list[ParsedResume]) -> None:
        self.jd = jd
        self.resumes = resumes
        self._bm25_norm = self._fit_bm25()

    # ------------------------------------------------------------------ BM25
    def _fit_bm25(self) -> dict[str, float]:
        """BM25 of the JD against every resume, min-max normalised. Raw BM25 has no
        fixed scale, so only the ordering within this pool means anything."""
        from rank_bm25 import BM25Okapi

        corpus = [tokenize(r.raw_text) for r in self.resumes]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(tokenize(self.jd.raw_text))

        lo, hi = float(min(scores)), float(max(scores))
        span = hi - lo
        return {
            r.stem: ((float(s) - lo) / span if span > 1e-9 else 0.5)
            for r, s in zip(self.resumes, scores)
        }

    # ------------------------------------------------------------------ per requirement
    def hit(self, req: Requirement, resume: ParsedResume) -> tuple[float, list[str]]:
        """Fraction of this requirement's skills the resume shows. Checks the skill
        set first, then scans the raw text -- a skill mentioned only in an experience
        bullet still counts."""
        if not req.keywords:
            return 0.0, []
        text = resume.raw_text.lower()
        matched = [
            kw for kw in req.keywords
            if kw in resume.skills
            or re.search(r"(?<![a-z0-9+#.])" + re.escape(kw) + r"(?![a-z0-9+#])", text)
        ]
        return len(matched) / len(req.keywords), matched

    # ------------------------------------------------------------------ document score
    def score(self, resume: ParsedResume, hits: dict[str, float]) -> tuple[float, float]:
        """Per-requirement hits + BM25 into S_kw. `hits` is passed in because fuse()
        already computed it for the evidence records."""
        total_w = sum(r.weight for r in self.jd.requirements)
        weighted = (
            sum(hits.get(r.id, 0.0) * r.weight for r in self.jd.requirements) / total_w
            if total_w else 0.0
        )
        bm25 = self._bm25_norm.get(resume.stem, 0.0)
        combined = (config.KW_SKILL_HIT_WEIGHT * weighted
                    + config.KW_BM25_WEIGHT * bm25)
        return combined, bm25
