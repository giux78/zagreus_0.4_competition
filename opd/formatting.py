"""ITALIC prompt construction.

The templates below are byte-identical to Crisp-Unimib/ITALIC run_eval.py so the
training distribution matches the benchmark's exact prompt format.
"""

from __future__ import annotations

import json
import random
from typing import Any

DEFAULT_SYSTEM_MESSAGE = "Sei un assistente utile."

QUERY_TEMPLATE_MULTICHOICE = """
Rispondi alla seguente domanda a scelta multipla sull'argomento '{topic}'. L'ultima riga della tua risposta deve essere nel seguente formato: 'Risposta: LETTERA' (senza virgolette) dove LETTERA è una tra {merged_letters}. Ragiona brevemente prima di rispondere.

{question}

{options}
""".strip()

QUERY_TEMPLATE_MULTICHOICE_FAST = """
Rispondi alla seguente domanda a scelta multipla sull'argomento '{topic}'. La tua risposta deve essere nel seguente formato: 'LETTERA' (senza virgolette) dove LETTERA è una tra {merged_letters}. Scrivi solo la lettera corrispondente alla tua risposta senza spiegazioni.

{question}

{options}

Risposta:
""".strip()


def format_options(options: list[tuple[str, str]]) -> tuple[str, str]:
    """options: [("A", "text"), ...] -> ("A) text\nB) ...", "ABCD")"""
    formatted = "\n".join(f"{letter}) {text}" for letter, text in options)
    letters = "".join(letter for letter, _ in options)
    return formatted, letters


def build_user_query(row: dict[str, Any], fast: bool = True) -> str:
    options_str, merged_letters = format_options(row["options"])
    template = QUERY_TEMPLATE_MULTICHOICE_FAST if fast else QUERY_TEMPLATE_MULTICHOICE
    return template.format(
        topic=row["category"],
        question=row["question"],
        options=options_str,
        merged_letters=merged_letters,
    )


def build_messages(
    row: dict[str, Any],
    few_shots: list[dict[str, Any]] | None = None,
    fast: bool = True,
    system_message: str | None = None,
) -> list[dict[str, str]]:
    """Full chat message list in ITALIC's structure: system, k few-shot turns, question."""
    messages = [{"role": "system", "content": system_message or DEFAULT_SYSTEM_MESSAGE}]
    for shot in few_shots or []:
        messages.append({"role": "user", "content": build_user_query(shot, fast=fast)})
        # ITALIC fast-mode shots answer with the bare letter; CoT shots too (the
        # official 5_shots.jsonl only stores the letter).
        messages.append({"role": "assistant", "content": shot["answer"]})
    messages.append({"role": "user", "content": build_user_query(row, fast=fast)})
    return messages


def load_italic_shots(path: str) -> list[dict[str, Any]]:
    """Load ITALIC's official 5_shots.jsonl (options as [{"A": "text"}, ...])."""
    shots = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            options = [(k, v) for opt in raw["options"] for k, v in opt.items()]
            shots.append(
                {
                    "question": raw["question"],
                    "options": options,
                    "answer": raw["answer"],
                    "category": raw["category"],
                }
            )
    return shots


class PromptRenderer:
    """Draws a prompt row and renders it as messages with a randomized shot regime.

    Regimes (probabilities from config):
      - "italic": ITALIC's official 5 shots — exactly what the benchmark harness sends
      - "pool":   k random shots drawn from the training pool (format generalization)
      - "zero":   no shots
    """

    def __init__(
        self,
        pool_rows: list[dict[str, Any]],
        italic_shots: list[dict[str, Any]],
        p_italic_shots: float = 0.5,
        p_pool_shots: float = 0.25,
        pool_shots_max_k: int = 5,
        cot_fraction: float = 0.0,
        rng: random.Random | None = None,
    ):
        self.pool_rows = pool_rows
        self.italic_shots = italic_shots
        self.p_italic_shots = p_italic_shots
        self.p_pool_shots = p_pool_shots
        self.pool_shots_max_k = pool_shots_max_k
        self.cot_fraction = cot_fraction
        self.rng = rng or random.Random(0)

    def sample(self) -> tuple[list[dict[str, str]], dict[str, Any], bool]:
        """Returns (messages, row, fast)."""
        row = self.rng.choice(self.pool_rows)
        fast = self.rng.random() >= self.cot_fraction
        u = self.rng.random()
        if u < self.p_italic_shots and self.italic_shots:
            shots = self.italic_shots
        elif u < self.p_italic_shots + self.p_pool_shots:
            k = self.rng.randint(1, self.pool_shots_max_k)
            shots = [s for s in self.rng.sample(self.pool_rows, k + 1) if s is not row][:k]
        else:
            shots = []
        return build_messages(row, few_shots=shots, fast=fast), row, fast
