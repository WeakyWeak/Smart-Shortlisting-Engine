"""Flag bias and narrow phrasing in the JD.

Three checks. Two are lexicon/rule based. The third reuses the ranking engine: for
each required tool, count candidates scoring high semantically with zero keyword hit.
Those people do the thing but name a different tool -- MySQL not PostgreSQL, Vue not
React. If a requirement excludes a lot of them it's phrased too narrowly, and we can
say so with a count from the actual pool. Only possible because the two channels are
independent.
"""
from __future__ import annotations

import json
import re

from .. import config
from ..schemas import CandidateScore, JobDescription

_LEX = json.loads((config.RESOURCES / "bias_lexicon.json").read_text())

# Only meaningful for a named tool. "Strong problem solving" can't be broadened.
_SKILLS_BY_CAT = json.loads((config.RESOURCES / "skills.json").read_text())
_TOOL_CATEGORIES = {"frontend", "backend", "database", "devops", "data_ml",
                    "security", "testing", "design_media", "languages"}
_TOOL_SKILLS: set[str] = {
    s for cat in _TOOL_CATEGORIES for s in _SKILLS_BY_CAT.get(cat, [])
}
_NOTES = _LEX.get("notes", {})
_CATEGORIES = [k for k in _LEX if not k.startswith("_") and k != "notes"]

_YEARS_RE = re.compile(r"(\d+)\+?\s*(?:-\s*\d+\s*)?year", re.I)
_JUNIOR_RE = re.compile(r"\b(intern|internship|junior|entry[- ]level|fresher|trainee|graduate)\b", re.I)


def audit(jd: JobDescription, scored: list[CandidateScore]) -> dict:
    findings: list[dict] = []
    findings += _lexicon(jd)
    findings += _experience_contradiction(jd)
    findings += _narrow_requirements(jd, scored)

    # JDs name the same tool in both Responsibilities and Required Skills; keep the
    # worst instance rather than reporting it twice.
    severity_rank = {"high": 0, "medium": 1, "low": 2}
    findings.sort(key=lambda f: (severity_rank.get(f["severity"], 3),
                                 -f.get("excluded_count", 0)))
    deduped, seen = [], set()
    for f in findings:
        key = (f["type"], f.get("term", "").lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(f)
    findings = deduped
    return {
        "checked": len(jd.requirements),
        "flags": len(findings),
        "findings": findings,
    }


def _lexicon(jd: JobDescription) -> list[dict]:
    """Coded language that suppresses applications from some groups."""
    out: list[dict] = []
    text = jd.raw_text
    for category in _CATEGORIES:
        if category == "feminine_coded":
            continue  # too weak on its own, too many false positives
        for term in _LEX[category]:
            match = re.search(r"(?<![a-z])" + re.escape(term) + r"(?![a-z])", text, re.I)
            if not match:
                continue
            out.append({
                "type": "coded_language",
                "category": category,
                "term": match.group(0),
                "severity": "low" if category == "feminine_coded" else "medium",
                "context": _context(text, match.start()),
                "why": _NOTES.get(category, "Wording may deter qualified applicants."),
                "suggestion": f"Rephrase or remove \"{match.group(0)}\".",
            })
    return out


def _experience_contradiction(jd: JobDescription) -> list[dict]:
    """Years-of-experience demands that contradict a junior title."""
    if not _JUNIOR_RE.search(jd.title):
        return []
    out: list[dict] = []
    for req in jd.requirements:
        match = _YEARS_RE.search(req.text)
        if match and int(match.group(1)) > 1:
            out.append({
                "type": "experience_contradiction",
                "severity": "high",
                "requirement": req.text,
                "term": match.group(0),
                "why": (f"The role is titled \"{jd.title}\" but this asks for "
                        f"{match.group(1)}+ years. Intern and junior applicants cannot "
                        f"satisfy it, so it filters out the intended audience."),
                "suggestion": "Drop the year count or express it as familiarity.",
            })
    return out


def _narrow_requirements(jd: JobDescription, scored: list[CandidateScore]) -> list[dict]:
    """Requirements excluding candidates who can demonstrably do the work.

    Two gates keep this honest. Test the RAW cosine, not the calibrated score --
    calibration is relative, so ~40% of any pool clears a fixed threshold and every
    requirement would look narrow. And only count otherwise-competitive candidates: a
    video editor missing React proves nothing about the React requirement.
    """
    if not scored:
        return []
    cutoff = sorted((c.final_score for c in scored), reverse=True)[
        max(0, int(len(scored) * config.BIAS_NARROW_TOP_FRACTION) - 1)]
    min_excluded = max(config.BIAS_NARROW_MIN_ABSOLUTE,
                       round(len(scored) * config.BIAS_NARROW_MIN_FRACTION))

    out: list[dict] = []
    for req in jd.requirements:
        tools = [k for k in req.keywords if k in _TOOL_SKILLS]
        if not req.required or not tools:
            continue

        excluded = [
            c for c in scored
            if c.final_score >= cutoff
            for e in c.evidence
            if e.requirement_id == req.id
            and e.kw_hit == 0.0
            and e.sem_raw >= config.BIAS_NARROW_SEM_THRESHOLD
        ]
        if len(excluded) < min_excluded:
            continue

        # What did they use instead? Their other skills tell us what the
        # requirement is really asking for.
        named = set(tools)
        alt: dict[str, int] = {}
        for cand in excluded:
            for skill in cand.resume.skills:
                if skill not in named:
                    alt[skill] = alt.get(skill, 0) + 1
        top = [s for s, n in sorted(alt.items(), key=lambda kv: -kv[1])
               if n >= len(excluded) * 0.4][:5]

        out.append({
            "type": "overly_narrow",
            "severity": "medium",
            "requirement": req.text,
            "term": ", ".join(tools),
            "excluded_count": len(excluded),
            "alternatives_used": top,
            "why": (f"{len(excluded)} candidates match this requirement semantically "
                    f"(raw cosine >= {config.BIAS_NARROW_SEM_THRESHOLD}) and rank in the top "
                    f"{int(config.BIAS_NARROW_TOP_FRACTION * 100)}% overall, but use none of its named "
                    f"tools. They demonstrate the capability with different technology."),
            "suggestion": (f"Broaden to the capability rather than the specific tool, "
                           f"e.g. accept " + (", ".join(top[:3]) if top else "equivalents")
                           + "."),
        })
    return out


def _context(text: str, pos: int, width: int = 60) -> str:
    start, end = max(0, pos - width), min(len(text), pos + width)
    return " ".join(text[start:end].split())
