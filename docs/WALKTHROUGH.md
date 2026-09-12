# File-by-file walkthrough

Read in execution order. One JD + N resumes go in one end; a ranked, explained
shortlist comes out the other.

---

## Entry points

### `run.py`
The CLI. Parses arguments, calls `nexora.pipeline.run()`, writes `out/ranking.json`,
then renders three things to the terminal: the ranked table, the top-3 explanations,
and the JD bias audit.

The table's **Δ column** (`semantic − keyword`) is the demo instrument — it makes
visible which candidates each channel found. `--why A B` shortcuts into the chat
module to compare two ranks.

```
python run.py               # rank data/resumes/ against data/jd/
python run.py --demo18      # top 18
python run.py --why 1 4     # why #1 beats #4
python run.py --no-rerank   # skip stage 7, for timing comparisons
```

### `app/api.py`
FastAPI. Deliberately thin: the engine runs once at startup, and every endpoint is a
read over the cached `out/ranking.json`. Nothing re-parses resumes while a judge is
watching.

`/api/ranking` (list, evidence stripped for payload size) · `/api/candidate/{stem}`
(full evidence) · `/api/bias` · `/api/chat` · `/api/rerun` · `/` serves the UI.

### `app/static/index.html`
The whole UI in one file — React and htm from a CDN, no build step, no npm, no CORS.
Ranked table with inline score bars; click a row to expand per-requirement evidence
colour-coded by which channel matched it; top-3 explanation cards; chat panel; bias
panel. A pure renderer — it contains no scoring logic.

---

## The engine — `nexora/`

### `config.py`
**Every tunable number in the system, and nothing else.** `tools/calibrate.py`
rewrites the weights here, so a weight inlined anywhere else is a bug calibration
cannot see. Also picks the torch device at runtime, so a CPU-only install needs no
code change, and reads `NEXORA_BI` / `NEXORA_CE` so models can be A/B-tested.

### `schemas.py`
Every dataclass: `Unit`, `ParsedResume`, `Requirement`, `JobDescription`, `Evidence`,
`CandidateScore`. The contract between stages — one file tells you what any function
receives and returns. `Evidence.matched` / `.hard_miss` and `CandidateScore.delta` /
`.coverage` are derived properties, so those definitions live in exactly one place.

### `pipeline.py`
Orchestrates stages 1→8 and serialises the result. `build_artifact()` defines
`ranking.json`, which is the contract with the API, the UI and the chat layer.

---

### Stage 1 — ingestion

**`ingest/readers.py`** · 4 formats behind one interface: `pymupdf` for PDF,
`python-docx` for DOCX (paragraphs *and* tables — resumes often put skills in a
table), `ElementTree` for the structured XML, plain read for TXT/MD.

`clean()` runs on everything: ftfy for mojibake, ligature repair, NFKC normalisation,
and a re-join for the case where PDF extraction puts the bullet glyph on its own line
with the bullet's text on the next (the real Sample_JD.pdf does exactly this — left
unhandled, every bullet parses as an empty marker plus an orphan sentence).

A PDF yielding under 100 characters is assumed scanned and gets an optional
`pytesseract` pass. Unreadable files return `""` rather than raising — one bad file
must not kill a 154-resume run.

**`ingest/dedup.py`** · The corpus ships the same resume in up to 4 formats. Collapses
by filename stem, keeping the highest-fidelity format but recording all of them.
`family_of()` extracts the job family from the filename — the relevance label that
makes `tools/calibrate.py` possible.

### Stage 2 — structuring

**`ingest/resume.py`** · Text → `ParsedResume`. Walks the document attributing each
line to the section header above it; headers are matched fuzzily (`rapidfuzz` ≥ 85)
against a synonym table, because this corpus spells them every possible way.

Sections become **evidence units** — individual bullets and sentences. This is the
single most important structural choice in the system: units, not documents, are what
gets embedded in Stage 5.

`extract_skills()` runs two passes with different jobs. A regex pass, exact-or-alias
over the whole resume, gives high precision and feeds the keyword channel. A fuzzy
pass runs *only* inside the skills section, where typos are likely and ambiguity is
low — running rapidfuzz over the whole document would be slow and would invent
matches. `_is_noise()` drops contact blocks and bare dates so a phone number can never
win a semantic match and end up quoted in an explanation.

### Stage 3 — the JD becomes requirements

**`ingest/jd.py`** · The spine. Splits the JD into atomic `Requirement` objects,
classifying required vs preferred from the enclosing header ("MUST-HAVE SKILLS" /
"GOOD-TO-HAVE SKILLS") with inline cues as a fallback.

An explicit header **wins** over an inline cue — under "MUST-HAVE SKILLS", the phrase
"(React preferred)" names the preferred framework, it does not make the requirement
optional. Rule-based rather than an LLM call, so the requirement list is byte-identical
every run — which matters when a judge re-runs your demo.

---

### Stage 4 — the keyword channel

**`matching/keyword.py`** · Two sub-signals doing different jobs.

*Per-requirement skill hit* — the fraction of a requirement's named skills the resume
demonstrates, via alias-expanded exact matching plus typo tolerance. The **precision**
signal: what stops loosely-related experience from satisfying an explicitly named tool.

*Cohort BM25* — the JD as a query over all resumes, min-max normalised. The **recall**
signal: vocabulary overlap beyond our 260-term taxonomy. BM25 over TF-IDF for term
saturation (the 10th "React" adds nothing) and length normalisation.

### Stage 5 — the semantic channel

**`matching/semantic.py`** · Where "built REST APIs with Express" gets credited
against "Node.js backend development" despite zero shared keywords.

Every evidence unit and every requirement is embedded in two batched calls, then one
matmul gives the full similarity matrix. Per requirement we take **MaxSim** over the
resume's units — the best-matching evidence, not a document average.

Three reasons this beats whole-document cosine: encoders truncate (512 tokens) so most
of a resume would never be seen; one excellent bullet survives instead of being
averaged into noise; and the `argmax` **is** the sentence the explanation quotes, which
turns explanation accuracy from a prompting problem into an array lookup.

### Stage 6 — calibration and fusion

**`matching/fuse.py`** · Where ranking spread comes from.

`calibrate()` z-normalises **each requirement across the cohort**, then squashes
through a logistic. Principled, not cosmetic: the task is ranking *this* pool against
*this* job, so a requirement every applicant satisfies carries no information and
should stop contributing to separation, while one that splits the pool should dominate.
Without this, every candidate lands within a few points of every other.

Then the **required-skill gate**: a required requirement with no keyword hit *and*
weak semantic evidence is a hard miss, and hard misses scale the final score down.
Doing two jobs — enforcing the brief's "explicitly requested skills shouldn't be
satisfied by loosely related experience", and producing the missing-skills list that
Stage 8 needs.

`apply_final()` is split out so calibration can re-score under different weights
without re-embedding — that is why the grid search takes seconds.

### Stage 7 — cross-encoder re-rank

**`matching/rerank.py`** · Stages 5–6 use a *bi*-encoder: the requirement and the
resume are embedded separately, so the model never sees them together. A *cross*-encoder
takes both as one joint input and attends across them — far more accurate, but a
forward pass per pair instead of a lookup. So it runs only on the top 10.

It is fed `(requirement, best evidence sentence)`, not `(requirement, whole resume)`:
short-query-against-short-passage is precisely its MS MARCO training distribution, and
a whole resume would truncate anyway. Blended rather than substituted — it refines the
ordering, it does not overrule the keyword channel.

---

### Stage 8 — explanations

**`explain/factsheet.py`** · The compliance boundary. `build()` emits a dict of
verified claims — scores, matched requirements with the channel that found each and
the evidence sentence, missing requirements. The LLM receives **only** this: no resume
text, no JD text. It is therefore structurally incapable of scoring; it can only
narrate. `delta_table()` does the same for two candidates, for the chat layer.

**`explain/prompts.py`** · The exact wording sent to the model, isolated so it is
auditable in one place when a judge asks.

**`explain/llm.py`** · Groq client. Lazy, and never raises — every failure returns
`None` so the caller falls back. Passes `reasoning_effort` for gpt-oss models, which
spend tokens on hidden reasoning before emitting content (a low `max_tokens` returns
an empty string rather than a short answer).

**`explain/template.py`** · The same explanations, assembled deterministically with no
API call. Insurance: a dead key or dead wifi cannot cost the explanation criterion. It
also proves the fact-sheet genuinely contains everything needed — the cleanest answer
to "are you sure the LLM isn't doing the scoring?"

---

### Bonus features — `nexora/bonus/`

**`bias.py`** · Three checks. A coded-language lexicon (gendered, age-proxy,
exclusionary). A contradiction check for years-of-experience demands in an intern
title. And the interesting one: **narrowness measured against the real applicant
pool**, reusing the ranking engine. For each required tool, count candidates scoring
high on *raw* cosine but zero on keyword — people who demonstrably do the thing while
naming a different tool — restricted to the top 35% so a video editor missing React
isn't counted as evidence. Grounded in data, not in a model's taste, and only possible
because the two channels are genuinely independent.

**`chat.py`** · No vector store. The artifact is small and questions name candidates,
so retrieval is **name resolution**, not similarity search. Every candidate is scored
and the best taken — never first-match-wins, since surnames repeat in this corpus and
a first-match loop silently answers about the wrong person. For "why is X above Y" it
builds a per-requirement **delta table** so the answer cites where they actually
differ instead of restating scores.

---

### Resources — `nexora/resources/`

- **`skills.json`** — ~260 canonical skills in 16 categories, mined from the corpus's
  own XML `<skillGroup>` tags rather than guessed.
- **`aliases.json`** — **surface variants only**: `js↔javascript`, `k8s↔kubernetes`.
  `Express → Node.js` is deliberately *absent* — that is a semantic relation and
  belongs to the other channel. Mixing them would collapse two "independent" channels
  into one signal computed twice. Single common English words are excluded too:
  `resolve` matched "Debug and resolve issues" against DaVinci Resolve until it was
  removed.
- **`families.json`** — filename → job family patterns, plus graded relevance per JD.
  The evaluation labels.
- **`bias_lexicon.json`** — coded-language terms with sourced rationale.

---

### Tools

**`tools/calibrate.py`** · Fits the fusion weights against real labels. Grid-searches
`(w_keyword, hard-miss penalty, floor)` to maximise NDCG@10 across **six** JDs — six
rather than one so no threshold can be tuned to a single JD's wording. Features are
computed once and cached, so hundreds of configurations evaluate in seconds. Reports a
channel ablation, which is where the honest negative result in the README comes from.

**`tools/validate_parser.py`** · 22 corpus resumes exist as *both* XML (semantically
tagged) and PDF/DOCX/TXT. Parses the messy formats heuristically and diffs against the
structured source. Turns "handles messy formatting" from an assertion into a number:
**0.997 skill recall, 1.000 precision, 66/66 names** across 66 comparisons.

**`scripts/download_models.py`** · Pre-downloads both models via `huggingface_hub`
(resumable) and smoke-tests them on the exact case the project rests on — asserting
that "Built REST APIs with Express and MongoDB" outscores an irrelevant sentence
against a "Node.js backend" requirement, with zero keyword overlap. If that assertion
passes, Stage 5 will work.

---

## Data layout

```
data/resumes/     the 18 shipped with the problem statement -- the cohort we rank
data/corpus/      154 practice resumes, 25 job families -- calibration only, never ranked
data/jd/          Sample_JD.pdf -- the real JD
data/jd/eval/     6 JDs used only by calibrate.py
out/ranking.json  the one artifact the API and UI read
```
