"""Build the training prompt pool from the three sources, deduped against ITALIC.

  python scripts/prepare_prompts.py --out data/prompts.jsonl

Sources:
  - mii-llm/pinocchio-raw (raw-splits/*.jsonl, streamed + reservoir-sampled)
  - efederici/MMLU-Pro-ita
  - sapienzanlp/mmlu_italian

Also downloads ITALIC's italic.jsonl (for dedup + final eval) and 5_shots.jsonl
(few-shot prefix) into data/.
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from opd.data import (  # noqa: E402
    load_italic_hashes,
    normalize_mmlu_italian,
    normalize_mmlu_pro_ita,
    normalize_pinocchio,
    question_hash,
    write_pool,
)

ITALIC_RAW = "https://raw.githubusercontent.com/Crisp-Unimib/ITALIC/main"
PINOCCHIO = "mii-llm/pinocchio-raw"


def hf_token() -> str | None:
    from huggingface_hub import get_token
    return os.environ.get("HF_TOKEN") or get_token()


def ensure_italic_files(data_dir: str) -> str:
    os.makedirs(data_dir, exist_ok=True)
    for name in ("italic.jsonl", "5_shots.jsonl"):
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            print(f"downloading {name} from ITALIC repo")
            r = requests.get(f"{ITALIC_RAW}/{name}", timeout=60)
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)
    return os.path.join(data_dir, "italic.jsonl")


def _http_lines(url: str, token: str | None):
    with requests.get(url, headers={"Authorization": f"Bearer {token}"},
                      stream=True, timeout=120) as r:
        r.raise_for_status()
        yield from r.iter_lines()


def _xet_lines(path: str):
    """Fallback for xet-backed files the CDN refuses to stream: download to
    the HF cache, iterate, then delete the blob (disk on the box is tight)."""
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(PINOCCHIO, path, repo_type="dataset")
    try:
        with open(local, "rb") as f:
            yield from f
    finally:
        blob = os.path.realpath(local)
        for p in (local, blob):
            try:
                os.unlink(p)
            except OSError:
                pass


def stream_pinocchio(cap_per_split: int, max_scan_lines: int, seed: int,
                     xet_fallback: bool = False):
    """Reservoir-sample each raw-splits/*.jsonl."""
    import json

    from huggingface_hub import HfApi

    token = hf_token()
    api = HfApi(token=token)
    files = [
        f for f in api.list_repo_files(PINOCCHIO, repo_type="dataset")
        if f.startswith("raw-splits/") and f.endswith(".jsonl")
    ]
    rng = random.Random(seed)
    for path in sorted(files):
        url = f"https://huggingface.co/datasets/{PINOCCHIO}/resolve/main/{path}"
        reservoir: list[dict] = []
        seen = 0
        transports = [_http_lines(url, token)]
        if xet_fallback:
            transports.append(_xet_lines(path))
        for lines in transports:
            reservoir, seen = [], 0
            try:
                for line in lines:
                    if not line.strip():
                        continue
                    seen += 1
                    if seen > max_scan_lines:
                        break
                    try:
                        row = normalize_pinocchio(json.loads(line))
                    except (json.JSONDecodeError, KeyError, TypeError):
                        continue
                    if row is None:
                        continue
                    if len(reservoir) < cap_per_split:
                        reservoir.append(row)
                    else:
                        j = rng.randint(0, seen - 1)
                        if j < cap_per_split:
                            reservoir[j] = row
                break  # this transport worked
            except Exception as e:
                print(f"  WARN {path}: {type(e).__name__}: {str(e)[:120]} — trying fallback")
        print(f"  {path}: scanned {seen} lines, kept {len(reservoir)}")
        yield from reservoir


def load_mmlu_datasets():
    from datasets import load_dataset

    print("loading efederici/MMLU-Pro-ita")
    try:
        ds = load_dataset("efederici/MMLU-Pro-ita")
        for split in ds:
            for raw in ds[split]:
                row = normalize_mmlu_pro_ita(raw)
                if row:
                    yield row
    except Exception as e:
        print(f"  WARN MMLU-Pro-ita failed ({e}) — inspect its schema and adjust "
              f"normalize_mmlu_pro_ita in opd/data.py")

    print("loading sapienzanlp/mmlu_italian")
    try:
        ds = load_dataset("sapienzanlp/mmlu_italian")
        for split in ds:
            for raw in ds[split]:
                row = normalize_mmlu_italian(raw)
                if row:
                    yield row
    except Exception as e:
        print(f"  WARN mmlu_italian failed ({e}) — inspect its schema and adjust "
              f"normalize_mmlu_italian in opd/data.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/prompts.jsonl")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--pinocchio-cap-per-split", type=int, default=5000)
    ap.add_argument("--pinocchio-max-scan-lines", type=int, default=300_000)
    ap.add_argument("--skip-pinocchio", action="store_true")
    ap.add_argument("--xet-fallback", action="store_true",
                    help="download xet-backed pinocchio files that refuse to stream "
                         "(needs disk headroom; deleted after sampling)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    italic_path = ensure_italic_files(args.data_dir)
    italic_hashes = load_italic_hashes(italic_path)
    print(f"ITALIC dedup set: {len(italic_hashes)} question hashes")

    seen: set[str] = set()
    rows = []
    dropped_italic = dropped_dup = 0

    def add(row):
        nonlocal dropped_italic, dropped_dup
        h = question_hash(row["question"])
        if h in italic_hashes:
            dropped_italic += 1
        elif h in seen:
            dropped_dup += 1
        else:
            seen.add(h)
            rows.append(row)

    for row in load_mmlu_datasets():
        add(row)
    if not args.skip_pinocchio:
        print("streaming mii-llm/pinocchio-raw (this takes a while, one-time)")
        for row in stream_pinocchio(args.pinocchio_cap_per_split,
                                    args.pinocchio_max_scan_lines, args.seed,
                                    xet_fallback=args.xet_fallback):
            add(row)

    random.Random(args.seed).shuffle(rows)
    n = write_pool(rows, args.out)
    by_source: dict[str, int] = {}
    for r in rows:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    print(f"\nwrote {n} prompts -> {args.out}")
    print(f"  by source: {by_source}")
    print(f"  dropped: {dropped_italic} ITALIC-overlap (!), {dropped_dup} duplicates")


if __name__ == "__main__":
    main()
