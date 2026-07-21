"""Drop pool rows that are near-duplicates of ANY ITALIC question.

For every pool row, compute its max cosine similarity to the 10k ITALIC
questions (paraphrase-multilingual embeddings). Drop the row if that max is
>= threshold. Asymmetric cost: over-removing wastes cheap data, under-removing
leaks the test set, so we drop on the semantic signal alone (conservative).

  python scripts/decontaminate_pool.py --pool data/pinocchio_text.jsonl \
      --italic ~/ai/ITALIC/italic.jsonl --threshold 0.80 \
      --out data/pinocchio_text_clean.jsonl
"""

from __future__ import annotations

import argparse
import html
import json

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def load(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, nargs="+", help="one or more pool jsonl files")
    ap.add_argument("--italic", required=True)
    ap.add_argument("--threshold", type=float, default=0.80)
    ap.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pool = [r for p in args.pool for r in load(p)]
    italic = load(args.italic)
    print(f"pool={len(pool)} rows, italic={len(italic)} questions, drop threshold={args.threshold}")

    m = SentenceTransformer(args.model, device="cuda")
    pe = m.encode([html.unescape(r["question"]) for r in pool], batch_size=256,
                  normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    ie = m.encode([html.unescape(r["question"]) for r in italic], batch_size=256,
                  normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    it = torch.tensor(ie, device="cuda")
    max_sim = np.empty(len(pool), dtype=np.float32)
    for i in range(0, len(pool), 4096):
        chunk = torch.tensor(pe[i : i + 4096], device="cuda")
        max_sim[i : i + 4096] = (chunk @ it.T).max(dim=1).values.cpu().numpy()

    keep = max_sim < args.threshold
    dropped = int((~keep).sum())
    with open(args.out, "w") as f:
        for r, k in zip(pool, keep):
            if k:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # distribution of what we dropped, for transparency
    for lo, hi in [(0.80, 0.85), (0.85, 0.90), (0.90, 0.95), (0.95, 1.01)]:
        n = int(((max_sim >= lo) & (max_sim < hi)).sum())
        print(f"  dropped in [{lo:.2f},{hi:.2f}): {n}")
    print(f"\nkept {int(keep.sum())} / {len(pool)}  (dropped {dropped}, "
          f"{100*dropped/len(pool):.1f}%) -> {args.out}")


if __name__ == "__main__":
    main()
