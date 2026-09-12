"""Stage 8a - scored candidate to a sheet of verified facts.

The LLM downstream sees ONLY what this emits: no resume text, no JD text. Every
number and quote here came from our own matching code, so the model can narrate but
cannot score.
"""
from __future__ import annotations

from .. import config
from ..schemas import CandidateScore, JobDescription


def build(cand: CandidateScore, jd: JobDescription,
          next_cand: CandidateScore | None = None) -> dict:
    """Facts about one candidate, ready for the explainer."""
    matched = sorted(cand.matched, key=lambda e: (-e.required, -e.sem_cal))
    missing = sorted(cand.missing, key=lambda e: (-e.required, e.sem_cal))
    covered, total = cand.coverage

    facts = {
        "rank": cand.rank,
        "name": cand.resume.display,
        "final_score": cand.final_score,
        "keyword_score": round(cand.keyword_score * 100, 1),
        "semantic_score": round(cand.semantic_score * 100, 1),
        "required_coverage": f"{covered}/{total}",
        "reranked": cand.reranked,
        "job_title": jd.title,
        "matched": [
            {
                "requirement": ev.requirement_text,
                "required": ev.required,
                # Which channel found it -- stated per requirement, not implied.
                "how": _how(ev),
                "matched_terms": ev.kw_terms,
                "evidence": ev.unit_text,
                "evidence_section": ev.unit_section,
                "semantic": round(ev.sem_cal, 3),
            }
            for ev in matched[:8]
        ],
        "missing": [
            {
                "requirement": ev.requirement_text,
                "required": ev.required,
                "hard_miss": ev.hard_miss,
            }
            for ev in missing[:6]
        ],
    }

    if next_cand is not None:
        facts["ranked_above"] = {
            "name": next_cand.resume.display,
            "final_score": next_cand.final_score,
            "required_coverage": "{}/{}".format(*next_cand.coverage),
        }
    return facts


def _how(ev) -> str:
    kw = ev.kw_hit >= config.MATCH_KW_THRESHOLD
    sem = ev.sem_cal >= config.MATCH_SEM_THRESHOLD
    if kw and sem:
        return "both channels"
    if kw:
        return "keyword only"
    return "semantic only"


def delta_table(a: CandidateScore, b: CandidateScore) -> dict:
    """Per-requirement comparison of two candidates, for the chat layer. Handing the
    model the requirements where they differ is what makes "why is X above Y"
    answerable with specifics."""
    rows = []
    by_id = {e.requirement_id: e for e in b.evidence}
    for ea in a.evidence:
        eb = by_id.get(ea.requirement_id)
        if eb is None:
            continue
        if ea.matched == eb.matched:
            continue
        rows.append({
            "requirement": ea.requirement_text,
            "required": ea.required,
            "winner": a.resume.display if ea.matched else b.resume.display,
            f"{a.resume.display}": "matched" if ea.matched else "no evidence",
            f"{b.resume.display}": "matched" if eb.matched else "no evidence",
            "evidence": (ea if ea.matched else eb).unit_text,
        })
    return {
        "a": {"name": a.resume.display, "rank": a.rank, "score": a.final_score,
              "coverage": "{}/{}".format(*a.coverage),
              "keyword": round(a.keyword_score * 100, 1),
              "semantic": round(a.semantic_score * 100, 1)},
        "b": {"name": b.resume.display, "rank": b.rank, "score": b.final_score,
              "coverage": "{}/{}".format(*b.coverage),
              "keyword": round(b.keyword_score * 100, 1),
              "semantic": round(b.semantic_score * 100, 1)},
        "differences": rows[:10],
    }
