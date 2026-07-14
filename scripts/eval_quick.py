"""Quick ITALIC-style accuracy eval of any HF checkpoint (no vLLM server needed).

  python scripts/eval_quick.py --model runs/opd/final --limit 1000

Uses ITALIC's exact prompt format, 5-shot fast mode, greedy decoding.
For official leaderboard numbers still run Crisp-Unimib/ITALIC's run_eval.py
against a vLLM server; this script is for fast iteration tracking.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from opd.formatting import build_messages, load_italic_shots  # noqa: E402

LETTER_RE = re.compile(r"\b([A-J])\b")


def load_italic_rows(path: str):
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            rows.append({
                "question": raw["question"],
                "options": [(k, v) for opt in raw["options"] for k, v in opt.items()],
                "answer": raw["answer"],
                "category": raw["category"],
                "macro": raw.get("macro_category", ""),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/italic.jsonl")
    ap.add_argument("--shots", default="data/5_shots.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--zero-shot", action="store_true")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    from opd.trainer import load_causal_lm

    tok = AutoTokenizer.from_pretrained(args.model)
    model = load_causal_lm(
        args.model, torch.bfloat16 if device != "cpu" else torch.float32
    ).to(device).eval()

    shots = None if args.zero_shot else load_italic_shots(args.shots)
    rows = load_italic_rows(args.data)
    if args.limit:
        rows = rows[: args.limit]
    bos = tok.convert_tokens_to_ids("<|begin_of_text|>")
    pad = tok.pad_token_id or tok.eos_token_id
    stop = [i for i in (tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id,
                        tok.convert_tokens_to_ids("<|end_of_text|>")) if i is not None]

    correct = 0
    by_macro = defaultdict(lambda: [0, 0])
    with torch.no_grad():
        for i in range(0, len(rows), args.batch):
            chunk = rows[i : i + args.batch]
            prompts = []
            for r in chunk:
                text = tok.apply_chat_template(
                    build_messages(r, few_shots=shots, fast=True),
                    add_generation_prompt=True, tokenize=False)
                ids = tok.encode(text, add_special_tokens=False)
                prompts.append(ids if ids[:1] == [bos] else [bos] + ids)
            T = max(len(p) for p in prompts)
            ids = torch.full((len(chunk), T), pad, dtype=torch.long)
            mask = torch.zeros((len(chunk), T), dtype=torch.long)
            for j, p in enumerate(prompts):
                ids[j, T - len(p):] = torch.tensor(p)
                mask[j, T - len(p):] = 1
            out = model.generate(
                ids.to(device), attention_mask=mask.to(device),
                do_sample=False, max_new_tokens=8, eos_token_id=stop, pad_token_id=pad)
            for j, r in enumerate(chunk):
                text = tok.decode(out[j, T:], skip_special_tokens=True)
                m = LETTER_RE.search(text)
                ok = bool(m) and m.group(1) == r["answer"]
                correct += ok
                by_macro[r["macro"] or "?"][0] += ok
                by_macro[r["macro"] or "?"][1] += 1
            done = i + len(chunk)
            print(f"\r{done}/{len(rows)} acc={correct / done:.4f}", end="", flush=True)

    print(f"\n\naccuracy: {correct / len(rows):.4f} on {len(rows)} questions")
    for macro, (c, n) in sorted(by_macro.items()):
        print(f"  {macro:35s} {c / n:.4f} ({n})")


if __name__ == "__main__":
    main()
