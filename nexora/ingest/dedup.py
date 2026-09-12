"""Stage 1b - one record per candidate. The same resume often exists as .docx,
.pdf, .txt and .xml; parse the cleanest, remember the rest."""
from __future__ import annotations

import json
import re
from pathlib import Path

from .. import config
from .readers import SUPPORTED

_FAMILIES = json.loads((config.RESOURCES / "families.json").read_text())
_PATTERNS: list[tuple[str, list[str]]] = [(f, pats) for f, pats in _FAMILIES["patterns"]]
RELEVANCE: dict[str, dict[str, int]] = _FAMILIES["relevance"]


def family_of(stem: str) -> str:
    """Job family from the filename -- our relevance label. Encoded three different
    ways in the corpus, so normalise separators and match longest patterns first."""
    norm = re.sub(r"[_\-]+", " ", stem.lower())
    norm = re.sub(r"\s+", " ", norm).strip()
    for fam, patterns in _PATTERNS:
        for pat in patterns:
            if re.sub(r"[_\-]+", " ", pat) in norm:
                return fam
    return "unknown"


def collect(data_dir: Path | None = None) -> list[tuple[str, Path, list[str]]]:
    """(stem, path to parse, formats available) per candidate, sorted by stem."""
    data_dir = data_dir or config.RESUME_DIR
    by_stem: dict[str, dict[str, Path]] = {}

    for path in sorted(data_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            continue
        by_stem.setdefault(path.stem, {})[path.suffix.lower().lstrip(".")] = path

    out: list[tuple[str, Path, list[str]]] = []
    for stem, formats in sorted(by_stem.items()):
        best = next((f for f in config.FORMAT_PREFERENCE if f in formats), None)
        if best is None:
            continue
        out.append((stem, formats[best], sorted(formats)))
    return out
