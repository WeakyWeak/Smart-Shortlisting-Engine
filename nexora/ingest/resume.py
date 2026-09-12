"""Stage 2 - text to evidence units (bullets/sentences, not whole documents).
Encoders truncate at 512 tokens and these resumes run past that."""
from __future__ import annotations

import json
import re
from pathlib import Path

from .. import config
from ..schemas import ParsedResume, Unit
from .dedup import family_of
from .readers import read

# --------------------------------------------------------------------------- taxonomy

_SKILLS_RAW = json.loads((config.RESOURCES / "skills.json").read_text())
SKILLS: set[str] = {s for k, v in _SKILLS_RAW.items() if k != "_comment" for s in v}
ALIASES: dict[str, str] = {
    k: v for k, v in json.loads((config.RESOURCES / "aliases.json").read_text()).items()
    if not k.startswith("_")
}
# Everything we can recognise on the surface, mapped to canonical form.
VOCAB: dict[str, str] = {s: s for s in SKILLS} | ALIASES

# Longest first so "react native" wins over "react".
_SKILL_PATTERNS = [
    (re.compile(r"(?<![a-z0-9+#.])" + re.escape(term) + r"(?![a-z0-9+#])", re.I), canon)
    for term, canon in sorted(VOCAB.items(), key=lambda kv: -len(kv[0]))
]

# --------------------------------------------------------------------------- sections

SECTION_SYNONYMS: dict[str, list[str]] = {
    "summary": ["summary", "professional summary", "profile", "about", "objective",
                "career objective", "about me", "overview"],
    "experience": ["experience", "work experience", "professional experience",
                   "work history", "employment", "employment history", "internship",
                   "internships", "professional background", "career history"],
    "projects": ["projects", "personal projects", "key projects", "academic projects",
                 "selected projects", "portfolio", "project work"],
    "education": ["education", "academic background", "academics", "qualifications",
                  "educational qualifications", "academic qualifications"],
    "skills": ["skills", "technical skills", "core skills", "key skills", "expertise",
               "technologies", "tech stack", "competencies", "core competencies",
               "skills summary", "technical proficiencies", "tools"],
    "certifications": ["certifications", "certificates", "courses", "licenses",
                       "certifications and courses", "training"],
    "achievements": ["achievements", "awards", "honors", "honours", "accomplishments",
                     "activities", "extracurricular", "leadership", "volunteering",
                     "publications", "interests", "hobbies", "languages", "contact"],
}
_FLAT_SYNONYMS = [(syn, sect) for sect, syns in SECTION_SYNONYMS.items() for syn in syns]

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_PHONE_RE = re.compile(r"(\+?\d[\d\s\-()]{7,}\d)")
_URL_RE = re.compile(r"(https?://\S+|(?:www\.|linkedin\.com|github\.com)\S*)", re.I)
_BULLET_RE = re.compile(r"^\s*(?:[-*]\s+|[\u2022\u25cf\u25aa\u2023]\s*)")
# Contact blocks and bare dates prove nothing, and can win a MaxSim -- which would
# put a phone number in an explanation.
_CONTACT_RE = re.compile(
    r"linkedin\.com|github\.com|https?://|@[\w-]+\.\w+|\+\d[\d\s-]{7,}", re.I)
_DATE_ONLY_RE = re.compile(
    r"^[\s\d/.-]*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|present|expected)?"
    r"[\s\d/.,-]*(to|[-])?[\s\d/.,-]*"
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|present|expected)?[\s\d/.,-]*$", re.I)


def _is_noise(text: str) -> bool:
    if _CONTACT_RE.search(text):
        return True
    if _DATE_ONLY_RE.match(text):
        return True
    letters = sum(c.isalpha() for c in text)
    return letters < len(text) * 0.5


def _match_header(line: str) -> str | None:
    """Section header? Fuzzy, since these are spelled every possible way. Gate on
    length first so a bullet containing "experience" isn't mistaken for one."""
    stripped = line.strip().strip(":").strip()
    if not stripped or len(stripped) > 45 or len(stripped.split()) > 5:
        return None
    if _BULLET_RE.match(line) or _EMAIL_RE.search(line):
        return None
    low = re.sub(r"[^a-z& ]", " ", stripped.lower())
    low = re.sub(r"\s+", " ", low).strip()
    if not low:
        return None
    for syn, sect in _FLAT_SYNONYMS:
        if low == syn:
            return sect
    # Only fuzzy-match ALL-CAPS or Title Case -- how headers actually appear.
    if not (stripped.isupper() or stripped.istitle()):
        return None
    try:
        from rapidfuzz import fuzz, process
        hit = process.extractOne(low, [s for s, _ in _FLAT_SYNONYMS],
                                 scorer=fuzz.ratio, score_cutoff=config.HEADER_FUZZ_THRESHOLD)
    except ImportError:
        return None
    return dict(_FLAT_SYNONYMS)[hit[0]] if hit else None


def _split_units(block: str, section: str) -> list[Unit]:
    """Bullets are already atomic; prose splits on sentence boundaries. Anything
    still too long gets chunked to stay inside the encoder window."""
    units: list[Unit] = []
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        line = _BULLET_RE.sub("", line).strip()
        if len(line) < config.MIN_UNIT_CHARS or _is_noise(line):
            continue
        pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z])", line) if len(line) > config.MAX_UNIT_CHARS else [line]
        for piece in pieces:
            piece = piece.strip()
            while len(piece) > config.MAX_UNIT_CHARS:
                cut = piece.rfind(" ", 0, config.MAX_UNIT_CHARS)
                cut = cut if cut > 0 else config.MAX_UNIT_CHARS
                head, piece = piece[:cut].strip(), piece[cut:].strip()
                if len(head) >= config.MIN_UNIT_CHARS:
                    units.append(Unit(head, section))
            if len(piece) >= config.MIN_UNIT_CHARS:
                units.append(Unit(piece, section))
    return units


def extract_skills(text: str, skills_block: str = "") -> set[str]:
    """Two passes. Regex is exact-or-alias over the whole resume (high precision,
    feeds the keyword channel). Fuzzy runs only inside the skills section, where
    typos are likely and ambiguity is low -- over the whole document it invents
    matches and is slow."""
    found = {canon for pattern, canon in _SKILL_PATTERNS if pattern.search(text)}

    if skills_block:
        try:
            from rapidfuzz import fuzz, process
        except ImportError:
            return found
        vocab = list(VOCAB)
        for entry in re.split(r"[,;|\n]", skills_block):
            entry = re.sub(r"\([^)]*\)", "", entry)          # drop "(basic)" etc.
            entry = re.sub(r"^[A-Za-z &/]{0,25}:", "", entry).strip().lower()
            if len(entry) < config.SKILL_FUZZ_MIN_LEN or len(entry) > 30:
                continue
            hit = process.extractOne(entry, vocab, scorer=fuzz.ratio,
                                     score_cutoff=config.SKILL_FUZZ_THRESHOLD)
            if hit:
                found.add(VOCAB[hit[0]])
    return found


def _extract_name(text: str, stem: str) -> str:
    """Name from the top of the document, falling back to the filename."""
    for line in text.splitlines()[:6]:
        line = line.strip()
        if not line or _EMAIL_RE.search(line) or _PHONE_RE.search(line) or _URL_RE.search(line):
            continue
        if _match_header(line) or len(line) > 45:
            continue
        words = line.replace(".", "").split()
        if 1 < len(words) <= 4 and all(w.replace("-", "").isalpha() for w in words):
            return line.title() if line.isupper() else line

    tail = re.split(r"resume|_\d+_|__", stem, flags=re.I)[-1]
    words = [w for w in re.split(r"[_\s-]+", tail) if w.isalpha() and len(w) > 1]
    return " ".join(w.capitalize() for w in words) if len(words) >= 2 else stem


def parse(path: Path, stem: str, formats: list[str]) -> ParsedResume | None:
    """Full Stage 1 + 2 for one candidate."""
    text = read(path)
    if len(text) < 120:
        return None

    # Walk the document, attributing every line to the section header above it.
    sections: dict[str, list[str]] = {}
    current = "summary"
    for line in text.splitlines():
        header = _match_header(line)
        if header:
            current = header
            continue
        sections.setdefault(current, []).append(line)

    units: list[Unit] = []
    for section, lines in sections.items():
        if section in {"education", "certifications", "achievements"}:
            continue  # rarely shows capability; keeps the unit set clean
        units.extend(_split_units("\n".join(lines), section))

    skills_block = "\n".join(sections.get("skills", []))
    email = _EMAIL_RE.search(text)

    return ParsedResume(
        stem=stem,
        name=_extract_name(text, stem),
        email=email.group(0) if email else None,
        family=family_of(stem),
        source_format=path.suffix.lower().lstrip("."),
        formats_available=formats,
        skills=extract_skills(text, skills_block),
        units=units,
        raw_text=text,
    )
