"""Select the SFT subset from DeepMount00/OpenItalianData (experiment policy).

Budget-driven: at ~25k tok/s on the A100, one epoch over ~1.3M rows (~400M
tokens) costs ~4.5-5h, leaving the 10h pipeline room for the OPD stage.
Policy: structural validity, length caps (avoid truncation waste at
max_seq_length 2048), first-user-turn dedup, seeded reservoir sample.

  python scripts/select_openita.py --out data/openita_selected.jsonl --budget 1300000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import unicodedata


def norm_key(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return "".join(c for c in s if c.isalnum())[:400]


def valid(conv) -> bool:
    if not isinstance(conv, list) or len(conv) < 2:
        return False
    roles = [m.get("role") for m in conv]
    if roles[0] not in ("system", "user") or roles[-1] != "assistant":
        return False
    if any(r not in ("system", "user", "assistant") for r in roles):
        return False
    assistant_chars = sum(len(m.get("content") or "") for m in conv if m["role"] == "assistant")
    total_chars = sum(len(m.get("content") or "") for m in conv)
    # too-short answers teach nothing; too-long rows waste the 2048 window
    return 20 <= assistant_chars and total_chars <= 7000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/openita_selected.jsonl")
    ap.add_argument("--budget", type=int, default=1_300_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from datasets import load_dataset

    rng = random.Random(args.seed)
    ds = load_dataset("DeepMount00/OpenItalianData", split="train", streaming=True)

    seen: set[str] = set()
    reservoir: list[dict] = []
    n_valid = 0
    stats = {"total": 0, "invalid": 0, "dup": 0}
    for row in ds:
        stats["total"] += 1
        conv = row.get("conversation")
        if not valid(conv):
            stats["invalid"] += 1
            continue
        first_user = next((m["content"] for m in conv if m["role"] == "user"), "")
        key = hashlib.md5(norm_key(first_user).encode()).hexdigest()
        if key in seen:
            stats["dup"] += 1
            continue
        seen.add(key)
        n_valid += 1
        item = {"conversation": conv}
        if len(reservoir) < args.budget:
            reservoir.append(item)
        else:
            j = rng.randrange(n_valid)
            if j < args.budget:
                reservoir[j] = item
        if stats["total"] % 250_000 == 0:
            print(f"  scanned {stats['total']}, kept-so-far {len(reservoir)}", flush=True)

    rng.shuffle(reservoir)
    with open(args.out, "w") as f:
        for item in reservoir:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"scanned {stats['total']}  invalid {stats['invalid']}  dup {stats['dup']}  "
          f"unique-valid {n_valid}  selected {len(reservoir)} -> {args.out}")


if __name__ == "__main__":
    main()
