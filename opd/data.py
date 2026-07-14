"""Prompt-pool construction and loading.

Pool row schema (prompts.jsonl):
  {"question": str, "options": [["A", "text"], ...], "answer": "A",
   "category": str, "source": "pinocchio|mmlu_pro_ita|mmlu_italian"}

Every row is deduped against the ITALIC 10k test set by a normalized question
hash — pinocchio-raw is (very likely) the corpus ITALIC was curated from, so
without this step we would be training on benchmark questions.
"""

from __future__ import annotations

import hashlib
import json
import string
import unicodedata
from typing import Any, Iterable

LETTERS = string.ascii_uppercase


def norm_text(s: str) -> str:
    """Aggressive normalization for near-duplicate detection."""
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return "".join(c for c in s if c.isalnum())


def question_hash(question: str) -> str:
    return hashlib.md5(norm_text(question).encode()).hexdigest()


def load_italic_hashes(italic_path: str) -> set[str]:
    hashes = set()
    with open(italic_path) as f:
        for line in f:
            if line.strip():
                hashes.add(question_hash(json.loads(line)["question"]))
    return hashes


# ---------------------------------------------------------------------------
# Per-source normalizers -> pool row or None (row rejected)
# ---------------------------------------------------------------------------

def _valid(question: str, options: list[tuple[str, str]], answer: str) -> bool:
    if not question or not (2 <= len(options) <= 10):
        return False
    if answer not in {letter for letter, _ in options}:
        return False
    if any(not text.strip() for _, text in options):
        return False
    if len(question) > 1500 or sum(len(t) for _, t in options) > 2000:
        return False
    return True


def normalize_pinocchio(raw: dict[str, Any]) -> dict[str, Any] | None:
    # skip anything that needs an image/table/context to answer
    if raw.get("image") or raw.get("table") or raw.get("context"):
        return None
    if any(o.get("image") for o in raw.get("options", [])):
        return None
    options = [(o["value"], o["text"].strip()) for o in raw.get("options", [])]
    question = (raw.get("question") or "").strip()
    answer = raw.get("answer", "")
    # topic string like ITALIC's category field, e.g. "archeologia_topografia_romana"
    gen_code = raw.get("genCode") or ""
    category = gen_code.split("__", 1)[1] if "__" in gen_code else (raw.get("macro") or "quiz").lower()
    if not _valid(question, options, answer):
        return None
    return {"question": question, "options": options, "answer": answer,
            "category": category, "source": "pinocchio"}


def normalize_mmlu_pro_ita(raw: dict[str, Any]) -> dict[str, Any] | None:
    question = (raw.get("question") or "").strip()
    opts = raw.get("options") or []
    options = [(LETTERS[i], str(t).strip()) for i, t in enumerate(opts)]
    answer = raw.get("answer", "")
    category = str(raw.get("category") or "conoscenze generali").lower().replace(" ", "_")
    if not _valid(question, options, answer):
        return None
    return {"question": question, "options": options, "answer": answer,
            "category": category, "source": "mmlu_pro_ita"}


def normalize_mmlu_italian(raw: dict[str, Any]) -> dict[str, Any] | None:
    question = (raw.get("input_translation") or "").strip()
    choices = raw.get("choices_translation") or []
    options = [(LETTERS[i], str(t).strip()) for i, t in enumerate(choices)]
    gold = raw.get("label")
    if gold is None or not (0 <= int(gold) < len(options)):
        return None
    answer = LETTERS[int(gold)]
    meta = raw.get("metadata") or {}
    subject = meta.get("subject") if isinstance(meta, dict) else None
    category = str(subject or "conoscenze generali").lower().replace(" ", "_")
    if not _valid(question, options, answer):
        return None
    return {"question": question, "options": options, "answer": answer,
            "category": category, "source": "mmlu_italian"}


# ---------------------------------------------------------------------------
# Pool loading
# ---------------------------------------------------------------------------

def write_pool(rows: Iterable[dict[str, Any]], path: str) -> int:
    n = 0
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_pool(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                row["options"] = [tuple(o) for o in row["options"]]
                rows.append(row)
    return rows


def split_pool(rows: list[dict[str, Any]], dev_size: int = 500, seed: int = 0):
    """Deterministic train/dev split by question hash (stable across runs)."""
    ranked = sorted(rows, key=lambda r: question_hash(r["question"]))
    dev = ranked[:dev_size]
    dev_hashes = {question_hash(r["question"]) for r in dev}
    train = [r for r in rows if question_hash(r["question"]) not in dev_hashes]
    return train, dev
