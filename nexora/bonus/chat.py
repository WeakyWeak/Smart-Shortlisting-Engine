"""Recruiter chat over the finished ranking.

No vector store. The artifact is small and questions name candidates, so retrieval is
name resolution, not similarity search. FAISS would add setup and latency without
improving an answer.

What does improve answers is the delta table: for "why is X above Y" we hand the
model the requirements where they actually differ.
"""
from __future__ import annotations

from ..explain import llm
from ..schemas import CandidateScore

_MAX_CANDIDATES_IN_CONTEXT = 6


def resolve_names(question: str, artifact: dict) -> list[dict]:
    """Candidates named in the question, by name or by #rank."""
    import re

    found: list[dict] = []
    by_rank = {c["rank"]: c for c in artifact["candidates"]}

    for token in re.findall(r"#\s*(\d+)|\brank\s+(\d+)|\bcandidate\s+(\d+)", question, re.I):
        num = next((int(t) for t in token if t), None)
        if num in by_rank and by_rank[num] not in found:
            found.append(by_rank[num])

    try:
        from rapidfuzz import fuzz
    except ImportError:
        return found

    low = question.lower()
    # Score everyone and take the best -- never first-match-wins. Names repeat here
    # (three Karthiks), and a first-hit loop silently answers about the wrong person.
    scored: list[tuple[int, int, dict]] = []
    for cand in artifact["candidates"]:
        if cand in found:
            continue
        name = cand["name"].lower()
        if name in low:
            scored.append((3, len(name), cand))
            continue
        tokens = [t for t in name.split() if len(t) > 2]
        present = [t for t in tokens if t in low]
        # A shared first name isn't enough -- need the full name or surname+1.
        if len(present) >= 2:
            scored.append((2, sum(len(t) for t in present), cand))
        elif present and tokens and present[0] == tokens[-1]:
            scored.append((1, len(present[0]), cand))

    scored.sort(key=lambda t: (-t[0], -t[1]))
    # Got a confident hit? Drop the surname-only matches -- different people who
    # just share a surname, and they pollute the context.
    if scored and scored[0][0] >= 2:
        scored = [t for t in scored if t[0] >= 2]

    seen_names = {c["name"] for c in found}
    for _, _, cand in scored:
        if cand["name"] in seen_names:
            continue
        seen_names.add(cand["name"])
        found.append(cand)

    return found[:_MAX_CANDIDATES_IN_CONTEXT]


def _slim(cand: dict, max_evidence: int = 8) -> dict:
    """One candidate, trimmed to what an answer needs."""
    matched = [e for e in cand["evidence"] if e["matched"]]
    missing = [e for e in cand["evidence"] if not e["matched"]]
    return {
        "rank": cand["rank"],
        "name": cand["name"],
        "family": cand["family"],
        "final_score": cand["final_score"],
        "keyword_score": cand["keyword_score"],
        "semantic_score": cand["semantic_score"],
        "required_coverage": cand["required_coverage"],
        "skills": cand["skills"][:25],
        "matched": [
            {"requirement": e["requirement"], "required": e["required"],
             "how": "keyword+semantic" if e["kw_hit"] > 0 else "semantic only",
             "terms": e["kw_terms"], "evidence": e["evidence"],
             "semantic": e["semantic"]}
            for e in sorted(matched, key=lambda e: (-e["required"], -e["semantic"]))[:max_evidence]
        ],
        "missing": [
            {"requirement": e["requirement"], "required": e["required"],
             "hard_miss": e["hard_miss"]}
            for e in sorted(missing, key=lambda e: -e["required"])[:6]
        ],
    }


def _delta_rows(a: dict, b: dict) -> list[dict]:
    """Requirements where only one of the two has evidence."""
    by_id = {e["id"]: e for e in b["evidence"]}
    rows = []
    for ea in a["evidence"]:
        eb = by_id.get(ea["id"])
        if eb is None or ea["matched"] == eb["matched"]:
            continue
        winner, loser = (a, b) if ea["matched"] else (b, a)
        rows.append({
            "requirement": ea["requirement"],
            "required": ea["required"],
            "only_matched_by": winner["name"],
            "no_evidence_from": loser["name"],
            "evidence": (ea if ea["matched"] else eb)["evidence"],
        })
    rows.sort(key=lambda r: -r["required"])
    return rows[:12]


def build_context(question: str, artifact: dict) -> dict:
    """Assemble the facts relevant to this question."""
    named = resolve_names(question, artifact)
    job = {
        "title": artifact["job"]["title"],
        "company": artifact["job"]["company"],
        "required_requirements": [r["text"] for r in artifact["job"]["requirements"]
                                  if r["required"]],
        "preferred_requirements": [r["text"] for r in artifact["job"]["requirements"]
                                   if not r["required"]],
        "total_candidates": len(artifact["candidates"]),
    }

    if len(named) >= 2:
        return {"job": job,
                "candidates": [_slim(c) for c in named[:2]],
                "differences": _delta_rows(named[0], named[1])}
    if len(named) == 1:
        return {"job": job, "candidates": [_slim(named[0])]}

    # Nobody named -- answer about the shortlist as a whole.
    return {
        "job": job,
        "leaderboard": [
            {"rank": c["rank"], "name": c["name"], "family": c["family"],
             "final_score": c["final_score"], "keyword_score": c["keyword_score"],
             "semantic_score": c["semantic_score"],
             "required_coverage": c["required_coverage"]}
            for c in artifact["candidates"][:15]
        ],
        "candidates": [_slim(c, max_evidence=5) for c in artifact["candidates"][:3]],
    }


def answer(question: str, artifact: dict) -> dict:
    """Answer a question. Always returns something usable."""
    context = build_context(question, artifact)
    text = llm.chat(context, question)
    if text:
        return {"answer": text, "source": "llm",
                "candidates_used": [c["name"] for c in context.get("candidates", [])]}
    return {"answer": _fallback(context, question), "source": "template",
            "candidates_used": [c["name"] for c in context.get("candidates", [])]}


def _fallback(context: dict, question: str) -> str:
    """Fallback when the LLM is unavailable."""
    cands = context.get("candidates", [])
    if len(cands) >= 2:
        a, b = cands[0], cands[1]
        lines = [
            f"{a['name']} is ranked #{a['rank']} ({a['final_score']}) and "
            f"{b['name']} is #{b['rank']} ({b['final_score']}). "
            f"Required coverage: {a['required_coverage']} vs {b['required_coverage']}. "
            f"Keyword {a['keyword_score']} vs {b['keyword_score']}, "
            f"semantic {a['semantic_score']} vs {b['semantic_score']}."
        ]
        diffs = context.get("differences", [])
        if diffs:
            lines.append("Requirements where they differ:")
            for d in diffs[:5]:
                tag = "required" if d["required"] else "preferred"
                lines.append(f"  - [{tag}] {d['requirement']}: "
                             f"only {d['only_matched_by']} has evidence.")
        return "\n".join(lines)
    if len(cands) == 1:
        c = cands[0]
        return (f"{c['name']} is ranked #{c['rank']} with {c['final_score']}, covering "
                f"{c['required_coverage']} required criteria "
                f"(keyword {c['keyword_score']}, semantic {c['semantic_score']}).")
    board = context.get("leaderboard", [])[:5]
    return "Top candidates: " + "; ".join(
        f"#{c['rank']} {c['name']} ({c['final_score']})" for c in board)


def compare_by_rank(artifact: dict, rank_a: int, rank_b: int) -> str:
    """Backs `run.py --why A B`."""
    by_rank = {c["rank"]: c for c in artifact["candidates"]}
    if rank_a not in by_rank or rank_b not in by_rank:
        return f"ranks {rank_a} and {rank_b} are not both in the ranking"
    a, b = by_rank[rank_a], by_rank[rank_b]
    return answer(f"Why is {a['name']} ranked above {b['name']}?", artifact)["answer"]


def to_scores(_: list[CandidateScore]) -> None:  # pragma: no cover - placeholder
    raise NotImplementedError
