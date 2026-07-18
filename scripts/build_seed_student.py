"""Build the SFT seed: zagreus-0.4B-ita weights + nesso-3B tokenizer.

The base model ships without a chat template. Adopting the teacher's (Llama-3
template, eos <|eot_id|>) makes student and teacher tokenizer-identical, so
the stage-2 OPD token bridge is the identity: no eos_map, no vocab mismatch.

  python scripts/build_seed_student.py --out models/zagreus-0.4B-seed
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="mii-llm/zagreus-0.4B-ita")
    ap.add_argument("--tokenizer-from", default="giux78/nesso-3B")
    ap.add_argument("--out", default="models/zagreus-0.4B-seed")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer_from)
    assert tok.chat_template, "tokenizer source has no chat template"
    try:
        model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16)
    except TypeError:  # transformers 4.x
        model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
    assert model.config.vocab_size == len(tok), (model.config.vocab_size, len(tok))

    eot = tok.convert_tokens_to_ids("<|eot_id|>")
    model.config.eos_token_id = eot
    if model.generation_config is not None:
        model.generation_config.eos_token_id = eot

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"seed saved: {args.out}  (eos -> <|eot_id|> {eot}, template from {args.tokenizer_from})")


if __name__ == "__main__":
    main()
