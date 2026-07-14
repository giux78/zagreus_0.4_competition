"""ITALIC (Crisp-Unimib) -- protocollo `fast`, 5-shot.

https://github.com/Crisp-Unimib/ITALIC -- 10.000 domande a scelta multipla su cultura,
senso comune e proprieta' linguistica italiana, 12 domini.

Il repo ufficiale valuta via un server vLLM dietro un'API OpenAI-compatible. Qui il backend
e' HF `generate` in locale: installare vLLM (o anche solo le sue dipendenze) nel venv mentre
il training ci gira dentro e' un rischio inutile.

Tutto il resto e' copiato VERBATIM da run_eval.py per non alterare il protocollo:
  - DEFAULT_SYSTEM_MESSAGE
  - QUERY_TEMPLATE_MULTICHOICE_FAST  (fast = risposta diretta, senza catena di ragionamento)
  - format_options / costruzione dei messaggi few-shot (user/assistant alternati)
  - extract_answer_fast + extract_answer  (parsing della risposta, con i fallback)
  - temperature 0.0, max_tokens 350
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- verbatim da run_eval.py -------------------------------------------------
DEFAULT_SYSTEM_MESSAGE = "Sei un assistente utile."

QUERY_TEMPLATE_MULTICHOICE_FAST = """
Rispondi alla seguente domanda a scelta multipla sull'argomento '{topic}'. La tua risposta deve essere nel seguente formato: 'LETTERA' (senza virgolette) dove LETTERA è una tra {merged_letters}. Scrivi solo la lettera corrispondente alla tua risposta senza spiegazioni.

{question}

{options}

Risposta:
""".strip()


def format_options(options):
    formatted = "\n".join(
        [f"{list(i.keys())[0]}) {list(i.values())[0]}" for i in options]
    )
    keys = "".join([list(i.keys())[0] for i in options])
    return formatted, keys


def extract_answer_fast(output: str) -> str:
    LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    min_index = min(
        [output.find(letter) for letter in LETTERS if letter in output], default=-1
    )
    if min_index == -1:
        return ""
    return output[min_index]


def extract_answer(output: str) -> str:
    def _find(pattern, text, ignore_case=True):
        flags = re.DOTALL | (re.IGNORECASE if ignore_case else 0)
        match = re.search(pattern, text, flags)
        if match:
            answer = re.sub(r"[è:)(?+-,;.]", "", match.group(1)).strip()
            answer = re.sub(r"^(?:sarà\s+la\s+|la\s+)?", "", answer).strip()
            return extract_answer_fast(answer) if answer else ""
        return ""

    def_pattern = r"Risposta:\s*(.*?)\s*(?=\n[A-Z]\)|\Z)"
    fallback_patterns = [
        r"quindi, la risposta è\s*(.*?)\s*(?=\n[A-Z]\)|\Z)",
        r"risposta\s*(?:corretta|giusta|appropriata|esatta|migliore|ottimale|finale|definitiva)?\s*[:è]*\s*(.*?)\s*(?=\n[A-Z]\)|\Z)",
        r"risposta\s*più\s*(?:corretta|appropriata)\s*[:è]*\s*(.*?)\s*(?=\n[A-Z]\)|\Z)",
        r"(?:soluzione|opzione|scelta|alternativa)\s*(?:corretta)?\s*[:è]*\s*(.*?)\s*(?=\n[A-Z]\)|\Z)",
        r"(?:quindi|in\s*conclusione,?)?\s*(?:la\s*)?risposta\s*è\s*(.*?)\s*(?=\n[A-Z]\)|\Z)",
        r"(?:la\s*)?(?:risposta|opzione|scelta)\s*(?:corretta|giusta|esatta)\s*è\s*(?:la\s*)?(?:lettera\s*)?([A-Z])",
    ]

    answer = _find(def_pattern, output, ignore_case=False)
    if answer:
        return answer
    if "nessuna delle opzioni" in output.lower():
        return ""
    for pattern in fallback_patterns:
        answer = _find(pattern, output)
        if answer:
            return answer
    return ""
# ------------------------------------------------------------------------------


def build_messages(row, few_shots):
    messages = [{"role": "system", "content": DEFAULT_SYSTEM_MESSAGE}]
    for shot in few_shots:
        o, k = format_options(shot["options"])
        messages.append({"role": "user", "content": QUERY_TEMPLATE_MULTICHOICE_FAST.format(
            topic=shot["category"], question=shot["question"], options=o, merged_letters=k)})
        messages.append({"role": "assistant", "content": shot["answer"]})
    o, k = format_options(row["options"])
    messages.append({"role": "user", "content": QUERY_TEMPLATE_MULTICHOICE_FAST.format(
        topic=row["category"], question=row["question"], options=o, merged_letters=k)})
    return messages


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--italic-dir", default="/tmp/ITALIC")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=8,
                    help="fast = solo la lettera; 350 del config e' un tetto, non un target")
    args = ap.parse_args()

    d = Path(args.italic_dir)
    data = [json.loads(l) for l in (d / "italic.jsonl").read_text().splitlines()]
    shots = [json.loads(l) for l in (d / "5_shots.jsonl").read_text().splitlines()]
    if args.limit:
        # campionamento a passo costante: italic.jsonl e' ordinato per categoria, quindi
        # prendere le prime N darebbe un campione tutto dello stesso dominio
        data = data[:: max(1, len(data) // args.limit)][: args.limit]
    print(f"ITALIC: {len(data)} domande, {len(shots)}-shot, protocollo fast\n")

    results = []
    for name in args.models:
        tok = AutoTokenizer.from_pretrained(name)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token or "<|end_of_text|>"
        model = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.bfloat16, attn_implementation="sdpa"
        ).cuda().eval()

        if not tok.chat_template:
            print(f"!! {name} non ha chat template: uso il fallback testuale grezzo")

        prompts = []
        for row in data:
            msgs = build_messages(row, shots)
            if tok.chat_template:
                prompts.append(tok.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=False))
            else:
                prompts.append("\n\n".join(m["content"] for m in msgs) + "\n")

        correct, blank = 0, 0
        per_macro = defaultdict(lambda: [0, 0])
        for i in range(0, len(prompts), args.batch_size):
            chunk = prompts[i : i + args.batch_size]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                      max_length=3800, add_special_tokens=False).to("cuda")
            out = model.generate(
                **enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
            for j in range(len(chunk)):
                text = tok.decode(out[j, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                pred = extract_answer(text) or extract_answer_fast(text)
                row = data[i + j]
                mc = row["macro_category"]
                per_macro[mc][1] += 1
                if not pred:
                    blank += 1
                if pred == row["answer"]:
                    correct += 1
                    per_macro[mc][0] += 1
            if i % (args.batch_size * 40) == 0:
                print(f"  {i}/{len(prompts)}  acc {correct/max(1,i+len(chunk)):.4f}", flush=True)

        acc = correct / len(data)
        results.append((name, acc, blank, dict(per_macro)))
        print(f"\n=== {name}")
        print(f"    accuracy {acc:.4f}   ({correct}/{len(data)})   risposte non parsate: {blank}")
        for mc, (c, n) in sorted(per_macro.items()):
            print(f"      {mc:<28s} {c/n:.4f}  ({c}/{n})")
        print()
        del model
        torch.cuda.empty_cache()

    print("=" * 66)
    print(f"{'modello':<40s} {'accuracy':>10s} {'vuote':>8s}")
    print("-" * 66)
    for name, acc, blank, _ in results:
        print(f"{name:<40s} {acc:10.4f} {blank:8d}")
    print("\n(caso = 0.25 su 4 opzioni)")


if __name__ == "__main__":
    main()
