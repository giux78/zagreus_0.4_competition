"""Verify student/teacher tokenizer compatibility before training.

  python scripts/check_compat.py

Downloads only the tokenizer files (a few MB), no model weights.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from opd import token_bridge  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="mii-llm/nesso-0.4B-agentic")
    ap.add_argument("--teacher", default="Coloss/nesso-3B")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    print(f"student: {args.student}\nteacher: {args.teacher}\n")
    s_tok = AutoTokenizer.from_pretrained(args.student)
    t_tok = AutoTokenizer.from_pretrained(args.teacher)

    # 1. fast checks + probe strings (raises on failure)
    token_bridge.assert_compatible(s_tok, t_tok)

    # 2. exhaustive: base vocab + merges must be identical
    s_raw = json.load(open(hf_hub_download(args.student, "tokenizer.json")))
    t_raw = json.load(open(hf_hub_download(args.teacher, "tokenizer.json")))
    sv, tv = s_raw["model"]["vocab"], t_raw["model"]["vocab"]
    mismatches = [k for k, v in tv.items() if sv.get(k) != v]
    assert not mismatches, f"{len(mismatches)} base-vocab id mismatches, e.g. {mismatches[:5]}"
    assert s_raw["model"]["merges"] == t_raw["model"]["merges"], "BPE merges differ!"
    print(f"[exhaustive] base vocab identical ({len(tv)} entries), merges identical")

    sa = {a["id"]: a["content"] for a in s_raw.get("added_tokens", [])}
    ta = {a["id"]: a["content"] for a in t_raw.get("added_tokens", [])}
    diff_s = {i: c for i, c in sorted(sa.items()) if ta.get(i) != c}
    diff_t = {i: c for i, c in sorted(ta.items()) if sa.get(i) != c}
    print(f"[added tokens] student-only/changed: {diff_s}")
    print(f"[added tokens] teacher-only/changed: {diff_t}")

    student_only_ids = set(diff_s) - set(ta)
    unmapped = student_only_ids - set(token_bridge.Bridge().swap)
    print(f"[bridge] swap map covers <|im_end|>; unmapped student-only ids "
          f"(truncated if sampled): {sorted(unmapped)}")
    print("\nAll checks passed — exact per-token KL is valid on the shared vocab.")


if __name__ == "__main__":
    main()
