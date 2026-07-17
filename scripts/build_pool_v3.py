"""Build the v3 training pool from the teacher-scored pool (experiment policy).

  python scripts/build_pool_v3.py --scored data/prompts_scored.jsonl --out data/prompts_v3.jsonl

Policy applied on top of `pgs distill-score` annotations (mechanism lives in
palingenesis.opd.score_pool; the choices below are ITALIC-specific):

  1. keep only teacher-correct rows — pure reverse KL distills the teacher's
     errors too, and nesso-3B is right only ~half the time
  2. upweight Italian-language rows (vocabulary/grammar/comprehension) by
     duplication — ITALIC is 40% "language capability" but the pool has only
     ~1k such rows (~0.9%), our weakest domain (31.8% vs 37.9% culture).
     Foreign-language quizzes (lingua_straniera, inglese*) are NOT upweighted:
     they don't exercise Italian language capability.

Note the honest limit: 990 language rows upweighted x4 is still only ~3.5% of
draws. The pool simply lacks Italian-language material; fixing that needs new
sources, not resampling.
"""

from __future__ import annotations

import argparse
import collections
import json
import re

LANG_KEY = re.compile(
    r"grammat|italiano|lingua|lessic|ortograf|sintass|congiuntiv|coniugaz|pronomi"
    r"|avverbi|aggettiv|sinonim|contrar|etimolog|fonolog|morfolog|punteggiatur"
    r"|comprensione|vocabol|analisi_grammaticale"
)
FOREIGN_KEY = re.compile(r"straniera|inglese|francese|tedesco|spagnolo")


def is_italian_language(category: str) -> bool:
    return bool(LANG_KEY.search(category)) and not FOREIGN_KEY.search(category)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", default="data/prompts_scored.jsonl")
    ap.add_argument("--out", default="data/prompts_v3.jsonl")
    ap.add_argument("--lang-factor", type=int, default=4,
                    help="total copies of each Italian-language row (1 = no upweight)")
    args = ap.parse_args()

    kept, dropped = [], 0
    acc_by_source = collections.Counter()
    n_by_source = collections.Counter()
    with open(args.scored) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            n_by_source[row["source"]] += 1
            acc_by_source[row["source"]] += row["teacher_correct"]
            if row["teacher_correct"]:
                kept.append(row)
            else:
                dropped += 1

    n_lang = sum(1 for r in kept if is_italian_language(r["category"]))
    out_rows = []
    for row in kept:
        copies = args.lang_factor if is_italian_language(row["category"]) else 1
        out_rows.extend([row] * copies)

    with open(args.out, "w") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = len(kept) + dropped
    print(f"scored rows: {total}  teacher-correct kept: {len(kept)} ({100*len(kept)/total:.1f}%)  dropped: {dropped}")
    for src in n_by_source:
        print(f"  teacher acc on {src:<14s} {acc_by_source[src]/n_by_source[src]:.3f} ({n_by_source[src]} rows)")
    print(f"italian-language rows kept: {n_lang} -> x{args.lang_factor} = "
          f"{n_lang*args.lang_factor}/{len(out_rows)} rows ({100*n_lang*args.lang_factor/len(out_rows):.1f}% of draws)")
    print(f"wrote {len(out_rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
