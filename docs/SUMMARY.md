# Smart Shortlisting Engine — one-page summary

**Problem.** Rank a pool of resumes against one Job Description, using both semantic
and keyword matching, and explain the top 3. An LLM is explicitly not allowed to do
the scoring.

---

## The idea

Most solutions embed the whole resume, embed the whole JD, take a cosine, blend with
BM25. That fails twice: sentence encoders **truncate at 512 tokens**, so a "whole
resume" cosine silently compares only the first ~180 words; and cosines between two
same-domain documents cluster in 0.45–0.80, so every candidate scores within a few
points of every other.

**We make the JD's individual requirements the spine instead of the JD blob.** The JD
is decomposed into atomic requirements; every candidate is scored against each one, on
both channels, recording the resume sentence that best matched. Ranking is weighted
requirement coverage.

Everything else falls out of that choice: "missing skills" is just the unmatched
required set, and every explanation ships with a citation because the argmax was free.

## Pipeline

```
PDF/DOCX/XML/TXT → evidence units ─┬→ keyword channel  (alias-exact hits + BM25)  ─┐
                                   └→ semantic channel (MaxSim per requirement)   ─┤
JD → atomic requirements ──────────────────────────────────────────────────────────┤
                                                                                   ▼
                          cohort z-calibration + required-skill gate → ranked list
                                                                                   ▼
                          cross-encoder re-rank (top 10) → fact-sheet → explanations
```

Two models, both transfer learning: a **bi-encoder** retrieves over the whole cohort,
a **cross-encoder** re-ranks the top 10 on `(requirement, evidence sentence)` pairs —
the standard retrieve-then-rerank pattern.

## The three decisions that matter

**The channels are genuinely independent.** The alias map holds *surface variants
only* — `js↔javascript`, `k8s↔kubernetes`. `Express → Node.js` is deliberately absent:
that is a semantic relation and belongs to the other channel. Mixing them would
collapse two "independent" signals into one computed twice.

**Spread comes from cohort-relative calibration.** Each requirement is z-normalised
across the applicant pool. A requirement everyone satisfies carries no information and
stops separating candidates; one that splits the pool dominates. A required-skill gate
then penalises requirements with neither keyword nor semantic evidence — which also
produces the missing-skills list.

**The LLM cannot score.** `explain/factsheet.py` is a hard boundary: the model receives
only verified facts our code computed — no resume text, no JD text. A deterministic
template fallback produces the same explanations with no API call at all.

## Measured results

| Check | Result |
|---|---|
| Parser vs XML ground truth (66 comparisons) | **0.997** recall, **1.000** precision, 66/66 names |
| Ranking quality, 6 JDs | strict NDCG@10 **0.880**, graded **0.973** |
| Score spread on the 18-resume cohort | **6.3 – 58.9** |
| Runtime | ~39s for 18 resumes (~50s with explanations) |

**Model choice was measured, not assumed.** `bge-large` beats `bge-small` by +0.048
strict NDCG@10 (0.832 → 0.880). That costs 9.4× runtime on the 154-resume corpus but
only **7 seconds** on the actual 18-resume cohort, where model loading dominates
encoding — so it is the right default here and would be the wrong one at scale.

**An honest negative result.** Channel ablation on the calibration corpus: keyword-only
**0.852**, semantic-only **0.796**, blended **0.850**. Keyword alone ties the blend —
because cross-family separation is a *lexical* task; telling a web developer from a
video editor barely needs semantics. What the family labels cannot measure is ordering
*within* the relevant family, which is where semantics pays and where no ground truth
exists. What we can show is that 3–5 of each top-20 are candidates keyword-only drops
entirely. We don't claim more than that.

## Reading the output

The **Δ column** (`semantic − keyword`) is the instrument. On the real cohort, **Priya Menon** at #1 covers 11/11 required criteria on
77.1 keyword. Candidates with a large positive Δ are the ones a keyword-only system
would bury — they prove a requirement through related work without naming the tool.

## Bonus features

- **JD bias audit** — coded-language lexicon, intern-title vs years-of-experience
  contradiction, and a narrow-phrasing check that *reuses the ranking engine*: count
  candidates scoring high semantically but zero on keyword for a required tool. The
  claim is grounded in the real applicant pool, not an LLM's opinion.
- **Recruiter chat** — name resolution rather than vector search; "why is X above Y"
  gets a per-requirement delta table so answers cite specifics.
- **Messy formats** — 4 readers, fuzzy section headers, typo-tolerant skills, OCR
  fallback, validated at 0.997 recall above.

## Stack

Python 3.13 · `BAAI/bge-large-en-v1.5` (bi-encoder) · `cross-encoder/ms-marco-MiniLM-L-6-v2`
(re-ranker) · `rank_bm25` · `rapidfuzz` · FastAPI · React via CDN · Groq for phrasing only.
Runs on CPU in under a minute; no training, no GPU required.
