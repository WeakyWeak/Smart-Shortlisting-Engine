# Smart Shortlisting Engine

Ranks a pool of resumes against one Job Description using **two genuinely independent
matching channels** — lexical and semantic — and explains the result per requirement,
citing the exact resume sentence behind every claim.

```bash
source .venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU wheel, ~200MB
python scripts/download_models.py        # once; verifies both models actually work

python run.py                            # rank data/resumes/ -> out/ranking.json  (~32s)
python run.py --why 1 13                 # why is #1 above #13
uvicorn app.api:app                      # UI at http://127.0.0.1:8000
```

**Docs:** [`docs/SUMMARY.md`](docs/SUMMARY.md) one page ·
[`docs/WALKTHROUGH.md`](docs/WALKTHROUGH.md) file by file ·
[`docs/DEMO.md`](docs/DEMO.md) demo runbook

---

## The design decision everything follows from

Most approaches embed the whole resume, embed the whole JD, take a cosine, and blend
it with BM25. That has two problems we built around.

**It is technically wrong.** Sentence encoders truncate (256 word-pieces for MiniLM,
512 for ours). These resumes run 2,000–5,600 characters. A "whole resume vs whole JD"
cosine silently compares the first ~180 words of each — usually the header and
education block.

**It produces no spread.** Cosines between two same-domain documents cluster in
~0.45–0.80, so every candidate lands within a few points of every other. The brief
explicitly demands "meaningful spread, not all-similar scores."

**So the JD's individual requirements are the spine, not the JD blob.** We decompose
the JD into atomic requirements and score every candidate against each one, on both
channels, recording the winning resume sentence. Ranking is weighted requirement
coverage. Everything else falls out of that: "missing skills" is just the unmatched
required set, and every explanation has a citation because the argmax was free.

## Pipeline

| Stage | Module | What it produces |
|---|---|---|
| 1 | `ingest/readers.py` | 4 formats (pdf/docx/xml/txt) → clean text; ftfy, ligature repair, OCR fallback |
| 2 | `ingest/resume.py` | Evidence **units** (bullets/sentences) + canonical skills |
| 3 | `ingest/jd.py` | JD → `Requirement[]`, required vs preferred |
| 4 | `matching/keyword.py` | `S_kw` — per-requirement alias-exact skill hits + cohort BM25 |
| 5 | `matching/semantic.py` | `S_sem` — MaxSim of each requirement over every unit, **+ the evidence sentence** |
| 6 | `matching/fuse.py` | Per-requirement cohort z-calibration, required-skill gate, final score |
| 7 | `matching/rerank.py` | Cross-encoder re-scores the top 10 |
| 8 | `explain/` | Fact-sheet → Groq, with a deterministic template fallback |

### Why the two channels are genuinely independent

The alias map (`resources/aliases.json`) holds **surface variants only** — `js↔javascript`,
`postgres↔postgresql`, `k8s↔kubernetes`. `Express → Node.js` is deliberately *absent*:
that is a semantic relation and belongs to the other channel. Had we put semantic
neighbours in the alias map, the two "independent" channels would collapse into one
signal computed twice.

### Where the spread comes from

Per-requirement **z-normalisation across the cohort**. For each requirement we take
its mean and std over all candidates and convert to a z-score. A requirement every
applicant satisfies carries no information and stops contributing to separation; one
that splits the pool dominates it. Then a **required-skill gate**: a required
requirement with no keyword hit *and* weak semantic evidence is a hard miss, and hard
misses scale the score down.

Result on the supplied 18-resume cohort: scores span **6.3 → 58.9**, degrading
monotonically with required-requirement coverage (11/11 at the top, 1/11 at the bottom). Note the score is *cohort-relative*
by construction — it answers "how does this candidate compare to this pool", not
"what percentage fit is this in the abstract".

## Measured results

**Parser vs ground truth** (`tools/validate_parser.py`) — 22 resumes ship as both XML
(semantically tagged) and pdf/docx/txt, so the heuristic parse can be checked against
the structured source:

```
mean skill recall    0.997   (66 format comparisons)
mean skill precision 1.000
name agreement       66/66
```

**Ranking quality** (`tools/calibrate.py`) — the 154-resume practice corpus in
`data/corpus/` has its job family in each filename, which is a real relevance label.
Evaluated across **six** JDs so nothing is tuned to one JD's wording:

```
strict NDCG@10 (exact family only)   0.850
graded NDCG@10 (adjacent credited)   0.978
per-JD: web_dev .967  app_dev .968  content .1000  cyber .1000  data_sci 1.000  sales .935
```

### An honest negative result

The channel ablation says:

```
keyword only    0.852
semantic only   0.796
both channels   0.850
```

**Keyword alone ties the blend on this metric, and we are not going to pretend
otherwise.** The reason is that cross-family separation is a largely *lexical* task —
telling a web developer from a video editor barely needs semantics. What the family
labels cannot measure is ordering *within* the relevant family, which is exactly where
the semantic channel earns its place, and where no ground truth exists in this corpus.

What we can show directly is that the semantic channel changes the shortlist: 3–5 of
the top 20 per JD are candidates a keyword-only ranking drops entirely (e.g. for the
app-dev JD, a candidate scoring keyword 16 / semantic 78). Whether those swaps are
*improvements* is a claim the available labels cannot settle.

The gate (`HARD_MISS_PENALTY`) is likewise kept at 0.12 even though the grid search
prefers 0.00, because the metric is blind to what the gate is for: it produces the
score spread and the missing-skills list, neither of which cross-family NDCG can see.

## The LLM does not score anything

`explain/factsheet.py` is the boundary. The model receives **only** that dict — no
resume text, no JD text. Every number and every quoted sentence was computed by our
own matching code. The LLM's job is strictly to narrate, and it is structurally
incapable of scoring because it never sees the inputs.

`explain/template.py` produces the same explanations deterministically with no API
call, so a dead key or dead wifi never costs the explanation criterion. Verify with
`GROQ_API_KEY= python run.py`.

## Reading the output

The **Δ column** (`semantic − keyword`) is the demo. Strongly positive means the
candidate proves the requirement through related work without naming the tool — the
case pure keyword search misses. Strongly negative means they name the tools but show
thinner supporting evidence.

```
 1  Priya Menon      58.9   kw 77.1  sem 40.6   -36.4   11/11
 4  Farhan Sheikh    45.6   kw 48.9  sem 42.2    -6.7   10/11
```

Priya leads on literal skill overlap with full required coverage. Candidates with a
strongly positive Δ are ones the keyword channel alone would have buried.

## Bonus features

- **JD bias audit** (`bonus/bias.py`) — coded-language lexicon, intern-title vs
  years-of-experience contradiction, and a narrowness check that reuses the ranking
  engine: count candidates who score high on *raw* cosine but zero on keyword for a
  required tool, restricted to the top 35% overall. That grounds the claim in the real
  applicant pool rather than an LLM's opinion, and names the tools they used instead.
- **Recruiter chat** (`bonus/chat.py`) — name resolution, not vector search. For
  "why is X above Y" it builds a per-requirement **delta table** so the answer cites
  the requirements where they actually differ.
- **Messy formats** — 4 readers, fuzzy section headers, typo-tolerant skill matching,
  OCR fallback. Validated at 0.997 recall above.

## Layout

```
nexora/          engine      config.py = every tunable number; schemas.py = every dataclass
  ingest/        stages 1-3
  matching/      stages 4-7
  explain/       stage 8 + template fallback
  bonus/         bias, chat
app/             FastAPI + single-file React (CDN, no build step)
tools/           calibrate.py, validate_parser.py
scripts/         download_models.py
out/ranking.json the one artifact the API and UI read
```

`config.py` is the only place weights live — `tools/calibrate.py` rewrites it, so an
inlined weight anywhere else is a bug calibration cannot see.

## Data layout

```
data/resumes/   the 18 resumes shipped with the problem statement -- the ranked cohort
data/corpus/    154 practice resumes, 25 job families -- calibration only, never ranked
data/jd/        Sample_JD.pdf -- the real JD
data/jd/eval/   6 JDs used only by tools/calibrate.py
```

Stage 3 reads whatever single JD lives in `data/jd/`, preferring a real PDF/DOCX, so
swapping the JD needs no code change.

## Choosing models

Model choice here is a measurement, not an argument. `tools/compare_models.py` scores
alternatives against the labelled corpus, and `NEXORA_BI` / `NEXORA_CE` override them:

```bash
python tools/compare_models.py              # bi-encoders
python tools/compare_models.py --rerankers  # cross-encoders
```

The bi-encoder result is why `bge-large` is the default:

| bi-encoder | strict NDCG@10 | graded | corpus runtime |
|---|---|---|---|
| `BAAI/bge-small-en-v1.5` (33M) | 0.8319 | 0.9541 | 83s |
| **`BAAI/bge-large-en-v1.5` (335M)** | **0.8802** | **0.9733** | 780s |

The 9.4× runtime is on the 154-resume corpus, where encoding dominates. On the actual
18-resume cohort it costs **7 seconds** (31.7s → 38.8s), because at that size model
*loading* dominates instead. +0.048 NDCG for 7 seconds is an easy trade — but note it
would be the wrong trade at corpus scale, which is exactly why it was measured rather
than assumed.

Swap back with `NEXORA_BI=BAAI/bge-small-en-v1.5` if you need the faster run.
