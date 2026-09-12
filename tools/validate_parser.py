#!/usr/bin/env python3
"""Check the heuristic parser against ground truth.

22 corpus resumes exist as both tagged .xml and .pdf/.docx/.txt. The XML says what
the correct extraction is, so we can parse the messy formats and measure how close we
get instead of eyeballing a few files and hoping.

    python tools/validate_parser.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexora import config  # noqa: E402
from nexora.ingest.readers import SUPPORTED  # noqa: E402
from nexora.ingest.resume import parse  # noqa: E402


def main() -> None:
    by_stem: dict[str, dict[str, Path]] = defaultdict(dict)
    for path in sorted(config.CORPUS_DIR.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            by_stem[path.stem][path.suffix.lower().lstrip(".")] = path

    multi = {s: f for s, f in by_stem.items() if "xml" in f and len(f) > 1}
    if not multi:
        sys.exit("no stems with both XML and another format")

    print(f"{len(multi)} resumes available in XML (ground truth) + other formats\n")
    print(f"{'resume':<42}{'fmt':<6}{'skill recall':<14}{'precision':<12}{'name':<6}")
    print("-" * 82)

    recalls: list[float] = []
    precisions: list[float] = []
    name_ok = 0
    name_total = 0

    for stem, formats in sorted(multi.items()):
        truth = parse(formats["xml"], stem, sorted(formats))
        if truth is None:
            continue

        for fmt, path in sorted(formats.items()):
            if fmt == "xml":
                continue
            got = parse(path, stem, sorted(formats))
            if got is None:
                print(f"{stem[:40]:<42}{fmt:<6}UNREADABLE")
                continue

            if truth.skills:
                recall = len(truth.skills & got.skills) / len(truth.skills)
                recalls.append(recall)
            else:
                recall = float("nan")
            if got.skills:
                precision = len(truth.skills & got.skills) / len(got.skills)
                precisions.append(precision)
            else:
                precision = float("nan")

            same_name = truth.name.lower().strip() == got.name.lower().strip()
            name_ok += same_name
            name_total += 1

            print(f"{stem[:40]:<42}{fmt:<6}{recall:<14.2f}{precision:<12.2f}"
                  f"{'ok' if same_name else 'DIFF':<6}")

            missed = truth.skills - got.skills
            if missed and recall < 0.85:
                print(f"{'':42}  missed: {', '.join(sorted(missed)[:8])}")

    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0  # noqa: E731
    print("-" * 82)
    print(f"mean skill recall vs XML ground truth : {mean(recalls):.3f}  "
          f"({len(recalls)} comparisons)")
    print(f"mean skill precision                  : {mean(precisions):.3f}")
    print(f"name extraction agreement             : {name_ok}/{name_total}")
    print()
    if mean(recalls) >= 0.85:
        print("PASS - heuristic parsing of messy formats matches the structured source.")
    else:
        print("BELOW TARGET (0.85) - the taxonomy or section splitter has gaps; "
              "see the 'missed' lines above.")


if __name__ == "__main__":
    main()
