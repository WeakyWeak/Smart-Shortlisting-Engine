# Demo runbook

## Before judges arrive

```bash
cd Nexora_ && source .venv/bin/activate
python scripts/download_models.py     # must print the OK line
python run.py                         # warms the model cache; ~50s cold
```

Leave `out/ranking.json` **in place** while you set up, then see below.

---

## Should you delete `out/ranking.json` before demoing?

**Yes — delete it and run live.** A cold run on the 18 resumes takes **~50 seconds**,
which is the right length: long enough that judges see real work happening, short
enough that nobody gets restless. You also get the stage-by-stage progress output,
which is free narration for exactly the thing judges are there to assess:

```
Stage 1-2: ingesting resumes...          parsed 18 resumes
Stage 3: decomposing the JD into requirements...
  Junior Full Stack Developer Intern @ TechNova Solutions: 17 requirements (11 required)
Stage 4-6: keyword + semantic matching, calibration, fusion...
  encoding 331 evidence units from 18 resumes on cpu...
Stage 7: cross-encoder re-rank...
Stage 8: explanations for top 3 via Groq...
```

Talk over it — that is your walkthrough of how the matching works.

**Two caveats.**

The 50s assumes the **models are already in `~/.cache/huggingface`** (bge-large is 1.3GB). On a cold cache
it is several minutes of downloading, so never let the first-ever run be the demo run.
Run it once beforehand.

Keep a **known-good `ranking.json` backed up** so a failed live run is not fatal:

```bash
cp out/ranking.json out/ranking.backup.json    # before the demo
rm out/ranking.json && python run.py           # the live run
cp out/ranking.backup.json out/ranking.json    # if anything goes wrong
```

For the **UI**, do the opposite — start `uvicorn` with the artifact already present so
the page loads instantly. Re-ranking live in a browser tab just looks like a hang.

---

## The three-minute walkthrough

**1. Run it live** (~50s). Narrate the stages as they print.

**2. Open the ranked table.** Point at the **Δ column** — this is the single best
answer to "did you really use both semantic and keyword matching?"

```
 1  Priya Menon      58.9   kw 77.1  sem 40.6   -36.4   11/11
 4  Farhan Sheikh    45.6   kw 48.9  sem 42.2    -6.7   10/11
```

Priya leads on literal skill overlap with full required coverage. Look down the Δ
column for the strongly positive rows — those are candidates a keyword-only system
would bury. Both channels are computed independently and both move the final score.

**3. Expand a candidate row** in the UI. Every requirement shows which channel matched
it and the exact resume sentence behind it, colour-coded: green = both channels,
blue = semantic only, red = no evidence. The **blue rows are the proof** — a
requirement satisfied with zero shared keywords.

**4. Show the top-3 explanations.** Then say the important part: *the LLM never sees a
resume or the JD.* Open `nexora/explain/factsheet.py` — it receives only that dict of
verified facts our code computed. It cannot score; it can only narrate.

**5. Kill the network** (or `unset GROQ_API_KEY`) and re-run. Explanations still appear
from the deterministic template. This proves the LLM is doing no scoring work.

**6. Chat:** "Why is Divya Krishnan ranked above Aakash Jain?" The answer cites the
specific requirements where they differ, because the engine hands it a per-requirement
delta table rather than two summaries.

---

## Questions judges will ask

**"Show me the semantic matching actually working."**
`python run.py --why 1 4`, or expand any candidate and read a blue-tagged row.

**"Is this just an LLM wrapper?"**
`nexora/explain/factsheet.py` is the boundary — it is the only thing the model ever
sees. Also demo the offline fallback (step 5).

**"How do you know your ranking is any good?"**
`python tools/calibrate.py`. The 154-resume corpus has job families in its filenames,
so we have real relevance labels: strict NDCG@10 **0.850** across six JDs. And be
ready with the honest part — keyword-only ties the blend on that metric, because
cross-family separation is lexical. Say so; the ablation is in the README.

**"How did you pick your weights?"**
Grid search over six JDs, not intuition. Same command.

**"Why this embedding model?"**
`python tools/compare_models.py` — we measured it. `bge-large` beat `bge-small` by
+0.048 strict NDCG@10, and costs only 7s more at this cohort size. At corpus scale it
costs 9.4×, so it would be the wrong default there. That is the point of measuring.

**"What if a resume is badly formatted?"**
`python tools/validate_parser.py` — 22 resumes exist as both tagged XML and PDF/DOCX/TXT,
so we diff our heuristic parse against ground truth: **0.997 recall, 1.000 precision,
66/66 names**.

**"Why did your bias checker find nothing?"**
Because the supplied JD is well written — it already says "SQL or NoSQL databases
(MySQL, PostgreSQL, MongoDB)". Not crying wolf is correct behaviour. Then run
`python run.py --jd data/jd/eval/web_dev.md` to show it firing on a JD that does have
problems.

---

## If something breaks

| Symptom | Fix |
|---|---|
| Explanations say `template` | Groq key/model issue — harmless, everything still works |
| Model download starts during the demo | `~/.cache/huggingface` was cleared; restore the backup artifact |
| Live run errors out | `cp out/ranking.backup.json out/ranking.json` and present from that |
| UI blank | Check `uvicorn` is running and `out/ranking.json` exists |
