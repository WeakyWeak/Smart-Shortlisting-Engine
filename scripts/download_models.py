#!/usr/bin/env python3
"""Pre-download and smoke-test both models.

    python scripts/download_models.py

Goes to ~/.cache/huggingface/hub, resumable.
"""
import os
import time

from huggingface_hub import snapshot_download

# Override with NEXORA_BI / NEXORA_CE to try alternatives.
BI_ENCODER = os.environ.get("NEXORA_BI", "BAAI/bge-large-en-v1.5")        # ~1.3GB
CROSS_ENCODER = os.environ.get("NEXORA_CE",
                               "cross-encoder/ms-marco-MiniLM-L-6-v2")    # ~80MB

# bge-large is default because it measured +0.048 NDCG@10 better, not because it's
# bigger. NEXORA_BI=BAAI/bge-small-en-v1.5 is ~7s faster and measurably worse.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def fetch(repo_id: str) -> str:
    print(f"\n> {repo_id}")
    t = time.time()
    path = snapshot_download(repo_id=repo_id)
    print(f"  ok  ({time.time() - t:.1f}s)  ->  {path}")
    return path


def main() -> None:
    import torch
    print(f"torch {torch.__version__}  |  cuda: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    fetch(BI_ENCODER)
    fetch(CROSS_ENCODER)

    # Smoke test: the exact case the whole project rests on.
    from sentence_transformers import CrossEncoder, SentenceTransformer
    print("\nloading models...")
    bi = SentenceTransformer(BI_ENCODER)
    ce = CrossEncoder(CROSS_ENCODER)

    requirement = "Experience with Node.js backend development"
    good = "Built REST APIs with Express and MongoDB"       # zero shared keywords
    bad = "Designed marketing posters in Adobe Photoshop"

    emb = bi.encode([QUERY_PREFIX + requirement, good, bad], normalize_embeddings=True)
    sim_good, sim_bad = float(emb[0] @ emb[1]), float(emb[0] @ emb[2])
    ce_good, ce_bad = ce.predict([(requirement, good), (requirement, bad)])

    print(f"\n  bi-encoder   relevant={sim_good:.3f}  irrelevant={sim_bad:.3f}")
    print(f"  cross-enc    relevant={ce_good:.3f}  irrelevant={ce_bad:.3f}")

    assert sim_good > sim_bad and ce_good > ce_bad, "models load but ranking is wrong"
    print("\nOK - both models working: semantic matching catches Express->Node.js "
          "with zero keyword overlap.\n")


if __name__ == "__main__":
    main()
