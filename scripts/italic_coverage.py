"""How much of ITALIC is in pinocchio? Best-neighbor distribution + threshold sweep.

Encodes both sets once, then for every ITALIC question reports its single best
semantic neighbor in pinocchio (with option/content agreement). This separates
"present but below a strict threshold" from "genuinely absent", and bounds the
true overlap between the two extremes.

  python scripts/italic_coverage.py --pool data/pinocchio_text.jsonl \
      --italic ~/ai/ITALIC/italic.jsonl --out data/italic_coverage.json
"""

from __future__ import annotations

import argparse
import html
import json

import numpy as np
from sentence_transformers import SentenceTransformer

from match_italic_semantic import content_tokens, jaccard, option_texts  # noqa: E402


def load(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--italic", required=True)
    ap.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    ap.add_argument("--out", default="data/italic_coverage.json")
    args = ap.parse_args()

    pool, italic = load(args.pool), load(args.italic)
    m = SentenceTransformer(args.model, device="cuda")
    pe = m.encode([html.unescape(r["question"]) for r in pool], batch_size=256,
                  normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    ie = m.encode([html.unescape(r["question"]) for r in italic], batch_size=256,
                  normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    pool_opts = [option_texts(r["options"]) for r in pool]

    best_sim = np.zeros(len(italic), dtype=np.float32)
    best_opt = np.zeros(len(italic), dtype=np.float32)
    best_ctok = np.zeros(len(italic), dtype=np.float32)
    for i, it in enumerate(italic):
        sims = pe @ ie[i]
        bi = int(sims.argmax())
        best_sim[i] = float(sims[bi])
        best_opt[i] = jaccard(option_texts(it["options"]), pool_opts[bi])
        best_ctok[i] = jaccard(content_tokens(it["question"]), content_tokens(pool[bi]["question"]))

    N = len(italic)
    # best-semantic-neighbor histogram
    edges = [0.0, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.01]
    hist = {f"{edges[i]:.2f}-{edges[i+1]:.2f}": int(((best_sim >= edges[i]) & (best_sim < edges[i+1])).sum())
            for i in range(len(edges) - 1)}

    # overlap bounds: verified (sem AND (options or content)) vs semantic-only ceiling
    verified = int(((best_sim >= 0.9) & ((best_opt >= 0.4) | (best_ctok >= 0.5))).sum())
    sem_085 = int((best_sim >= 0.85).sum())
    sem_090 = int((best_sim >= 0.90).sum())
    sem_095 = int((best_sim >= 0.95).sum())

    report = {
        "italic": N, "pinocchio": len(pool),
        "best_neighbor_semantic_histogram": hist,
        "italic_with_near_dup": {
            "verified (sem>=.9 & option/content agree)": verified,
            "semantic>=0.85 (loose ceiling)": sem_085,
            "semantic>=0.90": sem_090,
            "semantic>=0.95 (near-exact)": sem_095,
        },
        "pct": {
            "verified": round(100 * verified / N, 1),
            "sem_0.85": round(100 * sem_085 / N, 1),
            "sem_0.90": round(100 * sem_090 / N, 1),
            "sem_0.95": round(100 * sem_095 / N, 1),
        },
    }
    with open(args.out, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps(report["best_neighbor_semantic_histogram"], indent=1))
    print("\nITALIC questions with a pinocchio near-dup:")
    for k, v in report["italic_with_near_dup"].items():
        print(f"  {k}: {v} ({round(100*v/N,1)}%)")
    print(f"\nreport -> {args.out}")


if __name__ == "__main__":
    main()
