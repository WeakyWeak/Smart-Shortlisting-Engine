"""Stage 3 - JD to a list of atomic requirements.

Scoring per-requirement instead of blob-vs-blob means coverage IS the ranking, and
"which skills are missing" falls out for free. Rule-based, so the requirement list
is identical on every run.
"""
from __future__ import annotations

import re
from pathlib import Path

from .. import config
from ..schemas import JobDescription, Requirement
from .readers import read
from .resume import _SKILL_PATTERNS, _match_header

# Do the bullets under this header count as mandatory?
_REQUIRED_HEADERS = re.compile(
    r"required|must[- ]have|essential|minimum|core (skills|requirements)|soft skills|"
    r"what you.ll need|responsibilit|qualification|requirements", re.I)
_PREFERRED_HEADERS = re.compile(
    r"preferred|nice[- ]to[- ]have|bonus|good[- ]to[- ]have|desirable|plus|optional|"
    r"advantage|we.d love", re.I)

# For JDs that mix both in one list instead of splitting by header.
_PREFERRED_INLINE = re.compile(
    r"\b(nice to have|preferred|bonus|a plus|optional|desirable|exposure to|"
    r"familiarity with .{0,30}\bis a plus)\b", re.I)
_REQUIRED_INLINE = re.compile(r"\b(must|required|essential|strong|proficien\w+)\b", re.I)

_KIND_HEADERS = {
    "responsibility": re.compile(r"responsibilit|what you.ll do|the role|day[- ]to[- ]day|duties", re.I),
    "qualification": re.compile(r"qualification|education|eligibilit|who (can|should) apply", re.I),
}

_ROLE_RE = re.compile(
    r"\b(developer|engineer|intern|internship|manager|analyst|designer|scientist|"
    r"architect|consultant|specialist|lead|associate|executive)\b", re.I)

_BULLET_RE = re.compile(r"^\s*(?:[-*]\s+|[\u2022\u25cf\u25aa\u2023]\s*)")
_NOISE = re.compile(r"^(about|we are|company|benefits|perks|how to apply|location|"
                    r"stipend|salary|duration|apply)\b", re.I)


def _keywords_in(text: str) -> list[str]:
    """Skill tokens this requirement names. A requirement with none is still scored
    semantically, it just can't produce a lexical hit."""
    found: list[str] = []
    for pattern, canon in _SKILL_PATTERNS:
        if canon not in found and pattern.search(text):
            found.append(canon)
    return found


def parse_jd(path: Path) -> JobDescription:
    text = read(path)
    lines = text.splitlines()

    # Skip any HTML comment block before looking for the title.
    body, in_comment = [], False
    for line in lines:
        if "<!--" in line:
            in_comment = True
        if in_comment:
            if "-->" in line:
                in_comment = False
            continue
        body.append(line)
    lines = body

    # First two lines are role and employer, in either order.
    head: list[str] = []
    for line in lines[:12]:
        stripped = line.strip().lstrip("#").strip().strip("*")
        if stripped:
            head.append(stripped)
        if len(head) == 2:
            break
    head += ["", ""]
    title, company = head[0], head[1]
    if _ROLE_RE.search(company) and not _ROLE_RE.search(title):
        title, company = company, title
    # Strip "| Bengaluru | 6 months" style qualifiers.
    title = title.split("|")[0].strip()
    company = company.split("|")[0].strip()

    requirements: list[Requirement] = []
    seen: set[str] = set()
    section_required, section_kind = True, "skill"
    # Did the header state required/preferred outright? If so it wins: under
    # "MUST-HAVE SKILLS", "(React preferred)" names a framework, it doesn't make
    # the requirement optional.
    section_explicit = False

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("<!--") or line.startswith("-->"):
            continue

        # Section header -- switches the required/preferred context.
        header = line.lstrip("#").strip() if line.startswith("#") else None
        if header is None and not _BULLET_RE.match(raw):
            if _match_header(line):
                header = line
            elif (line.isupper() and 2 <= len(line.split()) <= 5
                  and (_REQUIRED_HEADERS.search(line) or _PREFERRED_HEADERS.search(line)
                       or _KIND_HEADERS["responsibility"].search(line))):
                header = line
        if header:
            if _PREFERRED_HEADERS.search(header):
                section_required, section_explicit = False, True
            elif _REQUIRED_HEADERS.search(header):
                section_required, section_explicit = True, True
            else:
                section_explicit = False
            section_kind = next(
                (k for k, pat in _KIND_HEADERS.items() if pat.search(header)), "skill")
            continue

        if not _BULLET_RE.match(raw):
            continue
        body = _BULLET_RE.sub("", line).strip().rstrip(".")
        if len(body) < 10 or _NOISE.match(body):
            continue

        key = body.lower()
        if key in seen:
            continue
        seen.add(key)

        # Only consult inline cues if the header didn't settle it.
        required = section_required
        if not section_explicit:
            if _PREFERRED_INLINE.search(body):
                required = False
            elif _REQUIRED_INLINE.search(body):
                required = True

        requirements.append(Requirement(
            id=f"R{len(requirements) + 1:02d}",
            text=body,
            kind=section_kind,
            required=required,
            keywords=_keywords_in(body),
            weight=config.REQUIRED_WEIGHT if required else config.PREFERRED_WEIGHT,
        ))

    return JobDescription(title=title or path.stem, company=company,
                          raw_text=text, requirements=requirements)


def load_jd(jd_dir: Path | None = None) -> JobDescription:
    """Load the JD from data/jd/. Prefers a real PDF/DOCX over markdown."""
    jd_dir = jd_dir or config.JD_DIR
    candidates = [p for p in sorted(jd_dir.iterdir())
                  if p.is_file() and p.suffix.lower() in {".pdf", ".docx", ".md", ".txt"}]
    if not candidates:
        raise FileNotFoundError(f"no JD found in {jd_dir}")
    real = [p for p in candidates if p.suffix.lower() in {".pdf", ".docx"}]
    return parse_jd(real[0] if real else candidates[0])
