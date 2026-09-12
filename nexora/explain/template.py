"""Stage 8c - explanations with no LLM at all.

Insurance against a missing key, a rate limit or dead wifi. Also proves the
fact-sheet alone contains everything needed to explain a ranking.
"""
from __future__ import annotations


def explain(facts: dict) -> str:
    name = facts["name"]
    parts: list[str] = []

    covered, total = facts["required_coverage"].split("/")
    parts.append(
        f"{name} ranks #{facts['rank']} with a final score of {facts['final_score']}, "
        f"covering {covered} of {total} required criteria "
        f"(keyword {facts['keyword_score']}, semantic {facts['semantic_score']})."
    )

    both = [m for m in facts["matched"] if m["how"] == "both channels"]
    sem_only = [m for m in facts["matched"] if m["how"] == "semantic only"]

    if both:
        terms = sorted({t for m in both for t in m["matched_terms"]})[:6]
        if terms:
            parts.append(
                "Direct matches on " + _join(terms)
                + ", confirmed by both literal and meaning-based matching."
            )

    if sem_only and sem_only[0]["evidence"]:
        top = sem_only[0]
        parts.append(
            f"They also satisfy \"{_trim(top['requirement'])}\" through related work "
            f"rather than the exact term - evidence: \"{_trim(top['evidence'], 160)}\"."
        )
    elif both and both[0]["evidence"]:
        parts.append(f"Supporting evidence: \"{_trim(both[0]['evidence'], 160)}\".")

    hard = [m["requirement"] for m in facts["missing"] if m["hard_miss"]]
    soft = [m["requirement"] for m in facts["missing"] if not m["hard_miss"]]
    if hard:
        parts.append("No evidence found for required criteria: "
                     + _join([_trim(h) for h in hard[:3]]) + ".")
    elif soft:
        parts.append("Gaps are limited to preferred criteria: "
                     + _join([_trim(s) for s in soft[:3]]) + ".")
    else:
        parts.append("No required criteria are unmet.")

    return " ".join(parts)


def _trim(text: str, limit: int = 70) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "..."


def _join(items: list[str]) -> str:
    items = [str(i) for i in items]
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]
