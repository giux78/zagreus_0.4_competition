"""Near-duplicate contamination check: training pool vs ITALIC test set.

ITALIC is derived from pinocchio (per its creator), which is also our pool's
main source — so exact-hash dedup (opd.pool.question_hash) can miss rewordings,
truncations, and punctuation variants that are effectively the test question.
This measures that leakage with MinHash/LSH (datasketch) and, crucially, tells
us how much of our PUBLISHED eval was contaminated.

  python scripts/dedup_lsh_check.py \
      --pool data/prompts_v3.jsonl \
      --italic ~/ai/ITALIC/italic.jsonl \
      --out data/contamination_report.json

Reports, at several Jaccard thresholds:
  - how many pool rows are near-dups of some ITALIC question (train-side leakage)
  - how many ITALIC questions have a near-dup in the pool (eval-side contamination
    — the number that matters for the published score)
  - sample matched pairs for eyeballing
  - the list of contaminated ITALIC question-hashes (to re-score on the clean subset)
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata

from datasketch import MinHash, MinHashLSH


def normalize(s: str) -> str:
    """lower() + accent-strip + collapse whitespace (per ITALIC creator's advice)."""
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s).strip()


def shingles(text: str, k: int = 4) -> set[str]:
    """Word k-shingles; falls back to the whole (short) question."""
    words = normalize(text).split()
    if len(words) <= k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def minhash(text: str, num_perm: int, k: int) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for sh in shingles(text, k):
        m.update(sh.encode())
    return m


def load_questions(path: str, field: str = "question") -> list[str]:
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line)[field])
    return out


def exact_hash(text: str) -> str:
    import hashlib

    return hashlib.md5("".join(c for c in normalize(text) if c.isalnum()).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--italic", required=True)
    ap.add_argument("--field", default="question")
    ap.add_argument("--num-perm", type=int, default=128)
    ap.add_argument("--shingle-k", type=int, default=4)
    ap.add_argument("--thresholds", default="1.0,0.9,0.8,0.7")
    ap.add_argument("--out", default="data/contamination_report.json")
    args = ap.parse_args()

    pool = load_questions(args.pool, args.field)
    italic = load_questions(args.italic, args.field)
    print(f"pool={len(pool)} rows, italic={len(italic)} questions")

    # exact-hash baseline (what we deduped with originally)
    italic_exact = {exact_hash(q) for q in italic}
    exact_hits = sum(1 for q in pool if exact_hash(q) in italic_exact)
    print(f"\nexact-hash leakage (original method): {exact_hits} pool rows")

    # MinHash every ITALIC question once, index by threshold-specific LSH
    print("computing ITALIC minhashes...")
    italic_mh = [minhash(q, args.num_perm, args.shingle_k) for q in italic]
    pool_mh_cache: dict[int, MinHash] = {}

    report = {"pool": len(pool), "italic": len(italic),
              "exact_hash_pool_hits": exact_hits, "thresholds": {}}

    for thr in [float(t) for t in args.thresholds.split(",")]:
        lsh = MinHashLSH(threshold=thr, num_perm=args.num_perm)
        for i, m in enumerate(italic_mh):
            lsh.insert(str(i), m)

        contaminated_italic: set[int] = set()
        leaked_pool = 0
        samples = []
        for pi, q in enumerate(pool):
            if pi not in pool_mh_cache:
                pool_mh_cache[pi] = minhash(q, args.num_perm, args.shingle_k)
            hits = lsh.query(pool_mh_cache[pi])
            if hits:
                leaked_pool += 1
                for h in hits:
                    contaminated_italic.add(int(h))
                if len(samples) < 12 and float(pool_mh_cache[pi].jaccard(italic_mh[int(hits[0])])) < 0.999:
                    samples.append({"jaccard": round(float(pool_mh_cache[pi].jaccard(italic_mh[int(hits[0])])), 3),
                                    "pool": q[:160], "italic": italic[int(hits[0])][:160]})

        pct = 100 * len(contaminated_italic) / len(italic)
        report["thresholds"][str(thr)] = {
            "leaked_pool_rows": leaked_pool,
            "contaminated_italic_questions": len(contaminated_italic),
            "contaminated_italic_pct": round(pct, 2),
            "contaminated_hashes": [exact_hash(italic[i]) for i in sorted(contaminated_italic)],
            "near_dup_samples": samples,
        }
        print(f"threshold {thr}: {leaked_pool} pool rows leak; "
              f"{len(contaminated_italic)} ITALIC questions contaminated ({pct:.1f}%)")

    with open(args.out, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"\nreport -> {args.out}")


if __name__ == "__main__":
    main()
