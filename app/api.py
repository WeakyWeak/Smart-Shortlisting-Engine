"""FastAPI layer over the ranking artifact.

Thin on purpose: the engine runs once at startup and every endpoint reads the cached
out/ranking.json. Nothing re-parses resumes mid-request.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nexora import config, pipeline  # noqa: E402
from nexora.bonus import chat as chat_mod  # noqa: E402

app = FastAPI(title="Smart Shortlisting Engine")
STATIC = Path(__file__).resolve().parent / "static"

_ARTIFACT: dict | None = None


def artifact() -> dict:
    """Cached ranking. Loads from disk, computes only if absent."""
    global _ARTIFACT
    if _ARTIFACT is None:
        if config.RANKING_JSON.exists():
            _ARTIFACT = json.loads(config.RANKING_JSON.read_text())
        else:
            _ARTIFACT = pipeline.run()
            pipeline.save(_ARTIFACT)
    return _ARTIFACT


@app.on_event("startup")
def _warm() -> None:
    data = artifact()
    print(f"ready: {len(data['candidates'])} candidates ranked for "
          f"{data['job']['title']}")


@app.get("/api/ranking")
def ranking(limit: int = 200) -> dict:
    """Ranked list. Evidence stripped to keep the payload small -- the UI fetches it
    per candidate on expand."""
    data = artifact()
    return {
        "job": data["job"],
        "config": data["config"],
        "bias": data["bias"],
        "count": len(data["candidates"]),
        "candidates": [
            {k: v for k, v in c.items() if k != "evidence"}
            for c in data["candidates"][:limit]
        ],
    }


@app.get("/api/candidate/{stem}")
def candidate(stem: str) -> dict:
    for c in artifact()["candidates"]:
        if c["stem"] == stem:
            return c
    raise HTTPException(404, f"no candidate {stem!r}")


@app.get("/api/bias")
def bias() -> dict:
    return artifact()["bias"]


class ChatRequest(BaseModel):
    question: str


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict:
    if not req.question.strip():
        raise HTTPException(400, "empty question")
    return chat_mod.answer(req.question, artifact())


@app.post("/api/rerun")
def rerun() -> dict:
    """Re-run the engine. Useful after swapping the JD."""
    global _ARTIFACT
    _ARTIFACT = pipeline.run()
    pipeline.save(_ARTIFACT)
    return {"status": "ok", "count": len(_ARTIFACT["candidates"])}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
