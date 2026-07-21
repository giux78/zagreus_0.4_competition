"""Find ITALIC questions inside a training pool via ANSWER-OPTION fingerprints.

ITALIC is derived from pinocchio (our pool's source), but exact-question hashing
and surface-question LSH both fail on the real leakage: rewordings (interrogative
-> fill-in-blank) share almost no question n-grams, while template siblings
("sinonimo di calesse" vs "di ottemperanza") share almost all of them. Neither
signal separates "same question, reworded" from "same template, different item".

The answer OPTIONS do. Two rows are the same question iff they offer the same
answer choices, regardless of how the stem is phrased or which letter maps where.
So we fingerprint each row by its normalized set of option texts:

  - reworded duplicate      -> same option set        -> MATCH   (surface LSH missed)
  - template sibling        -> different option set    -> reject  (surface LSH false-positived)
  - generic options (Vero/Falso, numbers): option set collides across unrelated
    questions -> we require the question stem to also be similar (surface Jaccard).

Speed: an inverted index (option-text -> rows) yields candidates sharing >=1
option; Jaccard is computed only on those. 10k x 49k stays sub-minute.

  python scripts/match_italic_options.py \
      --pool data/prompts_v3.jsonl --italic ~/ai/ITALIC/italic.jsonl \
      --out data/italic_pool_matches.json
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s).lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s).strip()


def option_texts(options) -> frozenset[str]:
    """Normalized set of option TEXTS. Accepts [[letter,text],...] or [{letter:text},...]."""
    out = set()
    for o in options:
        if isinstance(o, dict):
            out.update(norm(v) for v in o.values())
        else:  # [letter, text]
            out.add(norm(o[1]))
    return frozenset(t for t in out if t)


def q_shingles(text: str, k: int = 3) -> set[str]:
    w = norm(text).split()
    if len(w) <= k:
        return {" ".join(w)} if w else set()
    return {" ".join(w[i : i + k]) for i in range(len(w) - k + 1)}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


# generic option-sets that collide across unrelated questions -> need stem check
GENERIC = [frozenset({"vero", "falso"}), frozenset({"si", "no"}),
           frozenset({"vero", "falso", "non so"})]


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--italic", required=True)
    ap.add_argument("--opt-threshold", type=float, default=0.6, help="min option-set Jaccard")
    ap.add_argument("--q-threshold", type=float, default=0.5, help="min stem Jaccard for generic options")
    ap.add_argument("--out", default="data/italic_pool_matches.json")
    args = ap.parse_args()

    pool = load(args.pool)
    italic = load(args.italic)
    print(f"pool={len(pool)} rows, italic={len(italic)} questions")

    pool_opts = [option_texts(r["options"]) for r in pool]
    pool_qsh = [q_shingles(r["question"]) for r in pool]
    index: dict[str, list[int]] = defaultdict(list)
    for i, opts in enumerate(pool_opts):
        for t in opts:
            index[t].append(i)

    matched, reworded, samples = 0, 0, []
    per_option_count = 0
    for it in italic:
        iopts = option_texts(it["options"])
        ish = q_shingles(it["question"])
        cand = set()
        for t in iopts:
            cand.update(index.get(t, ()))
        best, best_j = None, 0.0
        for ci in cand:
            oj = jaccard(iopts, pool_opts[ci])
            if oj > best_j:
                best_j, best = oj, ci
        if best is None:
            continue
        qj = jaccard(ish, pool_qsh[best])
        is_generic = any(len(iopts & g) >= 2 and len(iopts) <= 3 for g in GENERIC)
        ok = best_j >= args.opt_threshold and (not is_generic or qj >= args.q_threshold)
        if ok:
            matched += 1
            if qj < 0.6:  # same options but stem reworded — the case surface LSH misses
                reworded += 1
            if len(samples) < 15:
                samples.append({"opt_jaccard": round(best_j, 3), "q_jaccard": round(qj, 3),
                                "italic_q": it["question"][:150], "pool_q": pool[best]["question"][:150]})

    report = {
        "pool_rows": len(pool), "italic_questions": len(italic),
        "matched": matched, "matched_pct": round(100 * matched / len(italic), 2),
        "reworded_subset": reworded,
        "opt_threshold": args.opt_threshold, "q_threshold": args.q_threshold,
        "samples": samples,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"\nITALIC questions found in pool: {matched} ({report['matched_pct']}%)")
    print(f"  of which stem-reworded (surface-LSH would miss): {reworded}")
    print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
