"""Stage 5 - semantic channel. Does the meaning match?

"Built REST APIs with Express and MongoDB" shares no keyword with "Node.js backend
development", so Stage 4 scores it zero. Here it scores ~0.7.

We MaxSim over evidence units rather than embedding whole resumes. Whole-document
embedding breaks three ways: the encoder truncates at 512 tokens, one good bullet
gets averaged into noise, and there's nothing left to quote. The argmax doubles as
the evidence sentence.

bge is an asymmetric retrieval model -- short query against passage, which is our
requirement-against-bullet shape. Hence QUERY_PREFIX on one side only.
"""
from __future__ import annotations

import numpy as np

from .. import config
from ..schemas import JobDescription, ParsedResume

_MODEL = None
_LOADED: str | None = None


def get_model():
    """Load once per process."""
    global _MODEL, _LOADED
    if _MODEL is None or _LOADED != config.BI_ENCODER:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(config.BI_ENCODER, device=config.device())
        _LOADED = config.BI_ENCODER
    return _MODEL


class SemanticMatcher:
    """Encodes the cohort once, then answers (requirement, resume) queries.
    Everything goes through two batched calls -- batching is where the speed is."""

    def __init__(self, jd: JobDescription, resumes: list[ParsedResume],
                 verbose: bool = True) -> None:
        self.jd = jd
        self.resumes = resumes
        model = get_model()

        # Queries: the JD requirements.
        req_texts = [config.QUERY_PREFIX + r.text for r in jd.requirements]
        self.req_emb = model.encode(req_texts, normalize_embeddings=True,
                                    batch_size=64, show_progress_bar=False)

        # Passages: every evidence unit, one flat batch.
        flat: list[str] = []
        self.spans: dict[str, tuple[int, int]] = {}
        for r in resumes:
            start = len(flat)
            flat.extend(u.text for u in r.units)
            self.spans[r.stem] = (start, len(flat))

        if verbose:
            print(f"  encoding {len(flat)} evidence units from {len(resumes)} resumes "
                  f"on {config.device()}...")
        self.unit_emb = model.encode(flat, normalize_embeddings=True,
                                     batch_size=128, show_progress_bar=False)

        # sim[requirement, unit] for the whole cohort, one matmul.
        self.sim = self.req_emb @ self.unit_emb.T

    def best(self, resume: ParsedResume, req_idx: int) -> tuple[float, str, str]:
        """Best evidence for this requirement. Averages the top-k rather than a
        single max so one lucky sentence can't carry a requirement; we still report
        the top unit for provenance."""
        start, end = self.spans[resume.stem]
        if start == end:
            return 0.0, "", ""

        row = self.sim[req_idx, start:end]
        k = min(config.SEM_TOP_K_UNITS, row.shape[0])
        top = np.argpartition(-row, k - 1)[:k]
        top = top[np.argsort(-row[top])]

        score = float(np.mean(row[top]))
        unit = resume.units[int(top[0])]
        return score, unit.text, unit.section

    def raw_matrix(self) -> np.ndarray:
        """[n_requirements, n_resumes] of best-match similarity. Stage 6 needs the
        whole cohort at once to z-normalise each requirement across it."""
        out = np.zeros((len(self.jd.requirements), len(self.resumes)), dtype=np.float32)
        for j, resume in enumerate(self.resumes):
            start, end = self.spans[resume.stem]
            if start == end:
                continue
            block = self.sim[:, start:end]
            k = min(config.SEM_TOP_K_UNITS, block.shape[1])
            part = np.partition(-block, k - 1, axis=1)[:, :k]
            out[:, j] = -np.mean(part, axis=1)
        return out

    def argmax_units(self) -> list[list[tuple[str, str]]]:
        """(unit_text, section) of the top match. Costs one argmax and it's the
        sentence every explanation quotes."""
        out: list[list[tuple[str, str]]] = []
        for i in range(len(self.jd.requirements)):
            row: list[tuple[str, str]] = []
            for resume in self.resumes:
                start, end = self.spans[resume.stem]
                if start == end:
                    row.append(("", ""))
                    continue
                idx = int(np.argmax(self.sim[i, start:end]))
                unit = resume.units[idx]
                row.append((unit.text, unit.section))
            out.append(row)
        return out
