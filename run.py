#!/usr/bin/env python3
"""Smart Shortlisting Engine - CLI.

    python run.py                  rank data/resumes/, write out/ranking.json
    python run.py --demo18         top 18 only
    python run.py --explain-only   redo explanations from the existing artifact
    python run.py --no-rerank      skip the cross-encoder stage
    python run.py --why 3 5        why is #3 above #5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nexora import config, pipeline  # noqa: E402


def _table(artifact: dict, limit: int) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console(width=132)
    cands = artifact["candidates"][:limit]

    table = Table(title=f"{artifact['job']['title']} @ {artifact['job']['company']}"
                        f"  -  {len(artifact['candidates'])} candidates ranked",
                  header_style="bold")
    table.add_column("#", justify="right", no_wrap=True)
    table.add_column("Candidate", no_wrap=True)
    show_family = any(c["family"] != "unknown" for c in cands)   # only for the corpus
    if show_family:
        table.add_column("Family", no_wrap=True)
    table.add_column("Score", justify="right", no_wrap=True)
    table.add_column("Kw", justify="right", no_wrap=True)
    table.add_column("Sem", justify="right", no_wrap=True)
    # Delta is the interesting column: large positive = someone the keyword channel
    # alone would have missed.
    table.add_column("Delta", justify="right", no_wrap=True)
    table.add_column("Req", justify="center", no_wrap=True)
    table.add_column("Fmt", no_wrap=True)

    for c in cands:
        delta = c["delta"]
        colour = "green" if delta > 8 else "red" if delta < -8 else "dim"
        score_colour = ("bold green" if c["final_score"] >= 70
                        else "yellow" if c["final_score"] >= 50 else "dim")
        row = [str(c["rank"]), c["name"][:22]]
        if show_family:
            row.append(c["family"])
        table.add_row(
            *row,
            f"[{score_colour}]{c['final_score']:.1f}[/]",
            f"{c['keyword_score']:.1f}", f"{c['semantic_score']:.1f}",
            f"[{colour}]{delta:+.1f}[/]",
            c["required_coverage"], c["source_format"],
        )
    console.print(table)

    scores = [c["final_score"] for c in artifact["candidates"]]
    console.print(f"[dim]score spread: {min(scores):.1f} - {max(scores):.1f} "
                  f"(range {max(scores) - min(scores):.1f})[/]")


def _explanations(artifact: dict) -> None:
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    console.print()
    for c in artifact["candidates"][:config.TOP_K_EXPLAIN]:
        if not c["explanation"]:
            continue
        matched = [e for e in c["evidence"] if e["matched"]]
        sem_only = [e for e in matched if e["kw_hit"] == 0]
        body = c["explanation"]
        body += (f"\n\n[dim]matched {len(matched)}/{len(c['evidence'])} requirements"
                 f"  -  {len(sem_only)} via semantic matching alone"
                 f"  -  source: {c['explanation_source']}[/]")
        console.print(Panel(body, title=f"#{c['rank']}  {c['name']}  ({c['final_score']})",
                            border_style="cyan"))


def _bias(artifact: dict) -> None:
    from rich.console import Console
    console = Console()
    bias = artifact.get("bias", {})
    if not bias.get("findings"):
        console.print("\n[green]JD bias audit: no issues found.[/]")
        return
    console.print(f"\n[bold]JD bias audit[/] - {bias['flags']} finding(s):")
    for f in bias["findings"]:
        colour = {"high": "red", "medium": "yellow"}.get(f["severity"], "dim")
        console.print(f"  [{colour}]* {f['type']}[/] ({f['severity']}): "
                      f"[bold]{f.get('term', '')}[/]")
        console.print(f"    {f['why']}")
        console.print(f"    [dim]-> {f['suggestion']}[/]")


def _why(artifact: dict, rank_a: int, rank_b: int) -> None:
    from nexora.bonus.chat import compare_by_rank
    from rich.console import Console
    Console().print(compare_by_rank(artifact, rank_a, rank_b))


def main() -> None:
    ap = argparse.ArgumentParser(description="Smart Shortlisting Engine")
    ap.add_argument("--jd", type=Path, help="path to a specific JD file")
    ap.add_argument("--data", type=Path, help="resume directory (default data/)")
    ap.add_argument("--limit", type=int, help="rank only the first N candidates")
    ap.add_argument("--demo18", action="store_true",
                    help=f"show the top {config.DEMO_COHORT_SIZE}")
    ap.add_argument("--top", type=int, default=25, help="rows to print (default 25)")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--no-explain", action="store_true")
    ap.add_argument("--explain-only", action="store_true",
                    help="reuse out/ranking.json, regenerate explanations only")
    ap.add_argument("--why", nargs=2, type=int, metavar=("A", "B"),
                    help="explain why rank A is above rank B")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.explain_only or args.why:
        if not config.RANKING_JSON.exists():
            sys.exit("no out/ranking.json yet - run `python run.py` first")
        artifact = json.loads(config.RANKING_JSON.read_text())
        if args.why:
            _why(artifact, *args.why)
            return
    else:
        artifact = pipeline.run(
            jd_path=args.jd, data_dir=args.data, limit=args.limit,
            do_rerank=not args.no_rerank, do_explain=not args.no_explain,
            verbose=not args.quiet,
        )
        path = pipeline.save(artifact)
        if not args.quiet:
            print(f"wrote {path}")

    _table(artifact, config.DEMO_COHORT_SIZE if args.demo18 else args.top)
    _explanations(artifact)
    _bias(artifact)


if __name__ == "__main__":
    main()
