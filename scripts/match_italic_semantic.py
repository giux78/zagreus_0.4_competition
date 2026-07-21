"""Robust ITALIC-in-pool detection: semantic question similarity + option verification.

Two cheap methods each fail (see match_italic_options.py and dedup_lsh_check.py):
  - surface question LSH: misses rewordings (fill-in-blank), false-positives on
    templates ("sinonimo di X" with different X).
  - option-set matching: catches rewordings, but false-positives when unrelated
    questions draw options from a shared small space (integers, city/region names,
    Vero/Falso).

A genuine duplicate has BOTH: the questions mean the same thing AND they offer the
same answer choices. So:
  1. embed questions with a multilingual PARAPHRASE model (handles reword/HTML-entity/
     casing noise that surface n-grams cannot), cosine kNN pool<-italic.
  2. confirm with option-text Jaccard.
  match = qsim >= q_hi   OR   (qsim >= q_lo AND option_jaccard >= opt_min)

The city/number collisions die (low qsim); the rewordings survive (high qsim,
same options). Output: matched ITALIC questions + the pool row ids to drop.

  python scripts/match_italic_semantic.py \
      --pool data/prompts_v3.jsonl --italic ~/ai/ITALIC/italic.jsonl \
      --out data/italic_matches_semantic.json
"""

from __future__ import annotations

import argparse
import html
import json
import re
import unicodedata

import numpy as np
from sentence_transformers import SentenceTransformer


def norm(s: str) -> str:
    s = html.unescape(str(s))
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s).strip()


def option_texts(options) -> frozenset[str]:
    out = set()
    for o in options:
        vals = o.values() if isinstance(o, dict) else [o[1]]
        out.update(norm(v) for v in vals)
    return frozenset(t for t in out if t)


def jaccard(a, b) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


# Frame words shared by question templates; excluded so the DISTINGUISHING
# content tokens (the quoted lemma in "sinonimo di X") drive the guard.
_STOP = set("il lo la i gli le un uno una di a da in con su per tra fra e o che chi cui "
            "quale quali qual come dove quando quanto quanti cosa non si e' è del della dei "
            "delle dal dalla al alla ai alle nel nella sul sé seguenti seguente termini "
            "parola verbi verbo tra fra questi queste questo indicati proposti sinonimo "
            "contrario significato individuare esatta corretta frase termine".split())


def content_tokens(text: str) -> set[str]:
    """Salient tokens: length>=4, not a template frame word (keeps quoted lemmas, names, numbers)."""
    return {w for w in norm(text).split() if len(w) >= 4 and w not in _STOP}


def load(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--italic", required=True)
    ap.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    ap.add_argument("--q-hi", type=float, default=0.92, help="qsim above this = match on stem alone")
    ap.add_argument("--q-lo", type=float, default=0.75, help="qsim above this + option overlap = match")
    ap.add_argument("--opt-min", type=float, default=0.4)
    ap.add_argument("--ctok-min", type=float, default=0.5, help="min salient-content-token Jaccard when matching on stem")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--out", default="data/italic_matches_semantic.json")
    args = ap.parse_args()

    pool = load(args.pool)
    italic = load(args.italic)
    print(f"pool={len(pool)} rows, italic={len(italic)} questions")

    model = SentenceTransformer(args.model, device="cuda")
    print("embedding pool...")
    pe = model.encode([html.unescape(r["question"]) for r in pool], batch_size=256,
                      normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
    print("embedding italic...")
    ie = model.encode([html.unescape(r["question"]) for r in italic], batch_size=256,
                      normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)

    pool_opts = [option_texts(r["options"]) for r in pool]
    pe = pe.astype(np.float32)
    ie = ie.astype(np.float32)

    matched, reworded, samples = 0, 0, []
    matched_pool_ids = set()
    for i, it in enumerate(italic):
        sims = pe @ ie[i]                       # cosine (both normalized)
        top = np.argpartition(sims, -args.topk)[-args.topk:]
        iopts = option_texts(it["options"])
        ictok = content_tokens(it["question"])
        best, best_score = None, -1.0
        for pi in top:
            qs = float(sims[pi])
            oj = jaccard(iopts, pool_opts[pi])
            # salient-content overlap: rejects "sinonimo di X" vs "di Y" (share
            # only frame words) while keeping true rewordings (share art/68/names).
            cj = jaccard(ictok, content_tokens(pool[pi]["question"]))
            hit = (oj >= args.opt_min and qs >= args.q_lo) or (qs >= args.q_hi and cj >= args.ctok_min)
            if hit and qs > best_score:
                best, best_score, best_oj = pi, qs, oj
        if best is None:
            continue
        matched += 1
        matched_pool_ids.add(int(best))
        # surface stem overlap, to count how many surface-LSH would have missed
        import difflib
        surf = difflib.SequenceMatcher(None, norm(it["question"]), norm(pool[best]["question"])).ratio()
        if surf < 0.6:
            reworded += 1
        if len(samples) < 20:
            samples.append({"qsim": round(best_score, 3), "opt_jac": round(best_oj, 3),
                            "surface": round(surf, 2),
                            "italic_q": html.unescape(it["question"])[:150],
                            "pool_q": pool[best]["question"][:150]})

    report = {"pool_rows": len(pool), "italic_questions": len(italic),
              "matched": matched, "matched_pct": round(100 * matched / len(italic), 2),
              "reworded_surface_lt_0.6": reworded,
              "unique_pool_rows_to_drop": len(matched_pool_ids),
              "thresholds": {"q_hi": args.q_hi, "q_lo": args.q_lo, "opt_min": args.opt_min},
              "pool_ids_to_drop": sorted(matched_pool_ids), "samples": samples}
    with open(args.out, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"\nITALIC found in pool: {matched}/{len(italic)} ({report['matched_pct']}%)")
    print(f"  stem-reworded (surface <0.6, LSH would miss): {reworded}")
    print(f"  unique pool rows to drop: {len(matched_pool_ids)}")
    print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
