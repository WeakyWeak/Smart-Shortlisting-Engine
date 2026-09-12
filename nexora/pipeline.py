"""Runs stages 1-8 and writes out/ranking.json -- the single artifact the API, UI and
chat all read. Engine runs once; every consumer is a pure renderer over it."""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from . import config
from .explain import factsheet, llm, template
from .ingest.dedup import collect
from .ingest.jd import load_jd
from .ingest.resume import parse
from .matching.fuse import fuse
from .matching.rerank import rerank
from .schemas import CandidateScore, JobDescription, ParsedResume


def load_resumes(data_dir: Path | None = None, verbose: bool = True) -> list[ParsedResume]:
    items = collect(data_dir)
    out: list[ParsedResume] = []
    for stem, path, formats in items:
        parsed = parse(path, stem, formats)
        if parsed is not None:
            out.append(parsed)
    if verbose:
        skipped = len(items) - len(out)
        print(f"  parsed {len(out)} resumes"
              + (f" ({skipped} unreadable)" if skipped else ""))
    return out


def run(jd_path: Path | None = None, data_dir: Path | None = None,
        limit: int | None = None, do_rerank: bool = True,
        do_explain: bool = True, verbose: bool = True) -> dict:
    """Full pipeline. Returns the ranking artifact."""
    started = time.time()

    if verbose:
        print("Stage 1-2: ingesting resumes...")
    resumes = load_resumes(data_dir, verbose)
    if not resumes:
        raise RuntimeError(f"no readable resumes in {data_dir or config.DATA_DIR}")

    if verbose:
        print("Stage 3: decomposing the JD into requirements...")
    jd = load_jd(jd_path.parent if jd_path else None)
    if jd_path:
        from .ingest.jd import parse_jd
        jd = parse_jd(jd_path)
    if verbose:
        print(f"  {jd.title} @ {jd.company}: {len(jd.requirements)} requirements "
              f"({len(jd.required)} required)")

    if verbose:
        print("Stage 4-6: keyword + semantic matching, calibration, fusion...")
    scored = fuse(jd, resumes, verbose=verbose)

    if do_rerank:
        if verbose:
            print("Stage 7: cross-encoder re-rank...")
        scored = rerank(scored, verbose=verbose)

    if limit:
        scored = scored[:limit]
        for i, cand in enumerate(scored, 1):
            cand.rank = i

    if do_explain:
        if verbose:
            src = "Groq" if llm.available() else "template (no GROQ_API_KEY)"
            print(f"Stage 8: explanations for top {config.TOP_K_EXPLAIN} via {src}...")
        explain_top(scored, jd)

    elapsed = time.time() - started
    if verbose:
        print(f"Done in {elapsed:.1f}s")
    return build_artifact(jd, scored, elapsed)


def explain_top(scored: list[CandidateScore], jd: JobDescription) -> None:
    """Explain the top K, falling back to templates on any failure."""
    for i, cand in enumerate(scored[:config.TOP_K_EXPLAIN]):
        nxt = scored[i + 1] if i + 1 < len(scored) else None
        facts = factsheet.build(cand, jd, nxt)
        text = llm.explain(facts)
        if text:
            cand.explanation, cand.explanation_source = text, "llm"
        else:
            cand.explanation, cand.explanation_source = template.explain(facts), "template"


def build_artifact(jd: JobDescription, scored: list[CandidateScore],
                   elapsed: float = 0.0) -> dict:
    from .bonus.bias import audit

    return {
        "job": {
            "title": jd.title,
            "company": jd.company,
            "requirements": [asdict(r) for r in jd.requirements],
        },
        "config": {
            "w_keyword": config.W_KEYWORD,
            "w_semantic": config.W_SEMANTIC,
            "bi_encoder": config.BI_ENCODER,
            "cross_encoder": config.CROSS_ENCODER,
            "hard_miss_penalty": config.HARD_MISS_PENALTY,
        },
        "bias": audit(jd, scored),
        "elapsed_seconds": round(elapsed, 2),
        "candidates": [_candidate_dict(c) for c in scored],
    }


def _candidate_dict(c: CandidateScore) -> dict:
    covered, total = c.coverage
    return {
        "rank": c.rank,
        "stem": c.resume.stem,
        "name": c.resume.display,
        "email": c.resume.email,
        "family": c.resume.family,
        "source_format": c.resume.source_format,
        "formats_available": c.resume.formats_available,
        "final_score": c.final_score,
        "keyword_score": round(c.keyword_score * 100, 2),
        "semantic_score": round(c.semantic_score * 100, 2),
        "delta": round(c.delta * 100, 2),
        "bm25": round(c.bm25_norm, 3),
        "reranked": c.reranked,
        "required_coverage": f"{covered}/{total}",
        "hard_misses": len(c.hard_misses),
        "skills": sorted(c.resume.skills),
        "explanation": c.explanation,
        "explanation_source": c.explanation_source,
        "evidence": [
            {
                "id": e.requirement_id,
                "requirement": e.requirement_text,
                "required": e.required,
                "matched": e.matched,
                "hard_miss": e.hard_miss,
                "kw_hit": round(e.kw_hit, 3),
                "kw_terms": e.kw_terms,
                "semantic": round(e.sem_cal, 3),
                "semantic_raw": round(e.sem_raw, 3),
                "cross_encoder": round(e.ce_score, 3) if e.ce_score is not None else None,
                "evidence": e.unit_text,
                "section": e.unit_section,
            }
            for e in c.evidence
        ],
    }


def save(artifact: dict, path: Path | None = None) -> Path:
    path = path or config.RANKING_JSON
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False))
    return path
