"""All dataclasses live here -- the contract between stages."""
from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------- Stage 1-2

@dataclass
class Unit:
    """A bullet or sentence from a resume. We embed these, not whole documents --
    encoders truncate, and this way every score has a quotable source."""
    text: str
    section: str          # normalised: summary | experience | projects | education | skills | other


@dataclass
class ParsedResume:
    stem: str                            # dedup key; filename without extension
    name: str
    email: str | None
    family: str                          # job family from the filename -> eval label
    source_format: str                   # the format we actually parsed
    formats_available: list[str]         # every format this resume came in
    skills: set[str] = field(default_factory=set)
    units: list[Unit] = field(default_factory=list)
    raw_text: str = ""

    @property
    def display(self) -> str:
        return self.name or self.stem


# --------------------------------------------------------------------------- Stage 3

@dataclass
class Requirement:
    """One thing the JD asks for. Everything is scored per-requirement, so the
    ranking is weighted coverage rather than one opaque number."""
    id: str
    text: str
    kind: str                            # skill | responsibility | qualification
    required: bool
    keywords: list[str] = field(default_factory=list)   # canonical skill tokens
    weight: float = 1.0


@dataclass
class JobDescription:
    title: str
    company: str
    raw_text: str
    requirements: list[Requirement] = field(default_factory=list)

    @property
    def required(self) -> list[Requirement]:
        return [r for r in self.requirements if r.required]


# --------------------------------------------------------------------------- Stage 4-7

@dataclass
class Evidence:
    """Why one requirement scored what it did. `unit_text` is the argmax of the
    semantic match -- free to capture, and it's what the explanation quotes."""
    requirement_id: str
    requirement_text: str
    required: bool
    kw_hit: float                        # 0-1 fraction of the requirement's keywords found
    kw_terms: list[str]                  # which terms actually matched
    sem_raw: float                       # cosine, before calibration
    sem_cal: float                       # after z-normalising across the cohort
    unit_text: str                       # the best-matching resume sentence
    unit_section: str
    ce_score: float | None = None        # cross-encoder, top-N only

    @property
    def matched(self) -> bool:
        from . import config
        return self.kw_hit >= config.MATCH_KW_THRESHOLD or self.sem_cal >= config.MATCH_SEM_THRESHOLD

    @property
    def hard_miss(self) -> bool:
        from . import config
        return self.required and self.kw_hit == 0.0 and self.sem_cal < config.HARD_MISS_SEM_THRESHOLD


@dataclass
class CandidateScore:
    resume: ParsedResume
    keyword_score: float                 # 0-1
    semantic_score: float                # 0-1
    bm25_norm: float
    evidence: list[Evidence]
    final_score: float = 0.0             # 0-100, after gate + optional rerank
    rank: int = 0
    reranked: bool = False
    explanation: str | None = None
    explanation_source: str = "none"     # llm | template

    @property
    def delta(self) -> float:
        """semantic - keyword. Large positive = someone keyword search would miss."""
        return self.semantic_score - self.keyword_score

    @property
    def matched(self) -> list[Evidence]:
        return [e for e in self.evidence if e.matched]

    @property
    def missing(self) -> list[Evidence]:
        return [e for e in self.evidence if not e.matched]

    @property
    def hard_misses(self) -> list[Evidence]:
        return [e for e in self.evidence if e.hard_miss]

    @property
    def coverage(self) -> tuple[int, int]:
        req = [e for e in self.evidence if e.required]
        return sum(1 for e in req if e.matched), len(req)
