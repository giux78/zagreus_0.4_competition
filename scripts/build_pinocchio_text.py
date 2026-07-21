"""Download efederici/pinocchio (config `text`) and normalize to the pool schema.

This is the corpus ITALIC is derived from — the correct source for both
decontamination (match ITALIC against it) and building a clean training pool.

  python scripts/build_pinocchio_text.py --out data/pinocchio_text.jsonl

Output rows: {"question", "options": [["A","text"],...], "answer", "category", "source"}.
"""

from __future__ import annotations

import argparse
import json

from datasets import load_dataset

LETTERS = "ABCDEFGHIJ"


def norm_options(options) -> list[list[str]]:
    """efederici options are [{"key":"A","value":"text"}, ...] -> [["A","text"], ...]."""
    out = []
    for i, o in enumerate(options):
        if isinstance(o, dict) and "value" in o:
            out.append([o.get("key") or LETTERS[i], str(o["value"]).strip()])
        elif isinstance(o, dict):  # {letter: text}
            (k, v), = o.items()
            out.append([k, str(v).strip()])
        else:
            out.append([LETTERS[i], str(o).strip()])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="efederici/pinocchio")
    ap.add_argument("--config", default="text")
    ap.add_argument("--out", default="data/pinocchio_text.jsonl")
    args = ap.parse_args()

    ds = load_dataset(args.dataset, args.config)  # dict of splits
    n = 0
    with open(args.out, "w") as f:
        for split, rows in ds.items():
            for r in rows:
                out = {
                    "question": (r.get("question") or "").strip(),
                    "options": norm_options(r.get("options") or []),
                    "answer": r.get("answer", ""),
                    "category": r.get("category") or r.get("macro") or split,
                    "source": f"pinocchio_text/{split}",
                }
                if out["question"] and out["options"]:
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")
                    n += 1
    print(f"wrote {n} rows -> {args.out}")


if __name__ == "__main__":
    main()
