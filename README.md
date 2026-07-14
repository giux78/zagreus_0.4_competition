# zagreus_competition — on-policy distillation for ITALIC

Improve **nesso-0.4B-agentic** (built on zagreus-0.4-ita) on the
[ITALIC](https://github.com/Crisp-Unimib/ITALIC) benchmark by on-policy
distillation from **Coloss/nesso-3B**, following the logic of
[tinker-cookbook's on_policy_distillation](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/recipes/distillation/on_policy_distillation.py)
but fully self-contained (PyTorch + transformers, no Tinker API).

## TL;DR results

Full 10,000-question ITALIC, 5-shot, `fast` protocol (direct letter answer):

| Model | Accuracy (fast) | Accuracy (slow / CoT) |
|---|---:|---:|
| `nesso-0.4B-agentic` (baseline student) | 32.9% | 0.0%\* |
| **`opd-v1/step_250` (distilled, this repo)** | **34.6%** | 30.5% |
| `nesso-3B` (teacher, the ceiling) | 50.7% | 2.5%\* |

\* The near-zero slow scores are a **scoring artifact**, not a reasoning
collapse — see [Why CoT scores near-zero](#why-cot-scores-near-zero).

**Net effect of distillation: +1.7 points on full-10k fast** (32.9 → 34.6),
saturating by ~250 steps. Random chance is ~25% (mostly 4-option). The teacher
sits at 50.7%, so ~16 points of headroom remain for a stronger recipe.

## Method

Each step:

1. Sample ITALIC-formatted MCQ prompts (byte-identical templates to the
   benchmark's `run_eval.py`; mixed 0-shot / official-5-shot / random-k-shot).
2. The **student samples completions** at temperature 1.0 with its current
   weights — exactly on-policy, one gradient step per batch, so no importance
   sampling is needed.
3. The **teacher scores those completions**, conditioned on the same
   conversation rendered with *its own* chat template.
4. Loss = **full-distribution reverse KL** per completion token
   `Σ_v p_s(v)·(log p_s(v) − log p_t(v))` — richer than the recipe's
   sampled-token variant (available as `loss_fn: sampled_rkl`), which Tinker
   uses only because its API doesn't expose full teacher distributions.

### Why nesso-3B as teacher (not nesso-4B-v2)

`Coloss/nesso-3B` is a Llama-3 arch sharing the student's Llama-3 base
vocabulary, so we can compute **exact per-token reverse KL** on the shared
vocab. `Coloss/nesso-4B-v2` is a Qwen3.5 model with a different ~248k Qwen
tokenizer — usable only via a noisier cross-tokenizer alignment, so it was
rejected for this run. (On the training box `giux78/nesso-3B` is a
byte-identical cached mirror of `Coloss/nesso-3B`.)

### The tokenizer bridge

Student and teacher share the Llama-3 base vocab (ids 0–128255) — verified
exhaustively by `scripts/check_compat.py` — but use different chat formats:
the student is ChatML (`<|im_start|>` = 128257, `<|im_end|>` = 128256, out of
range for the teacher), the teacher is Llama-3 (`<|eot_id|>` = 128009).
So prompts are rendered per-model and only completion tokens are aligned;
the student's `<|im_end|>` probability mass is merged into the teacher's
`<|eot_id|>` slot so stopping behavior is distilled too (`opd/token_bridge.py`).

### Training data

Prompt pool built from three sources, **deduped against the ITALIC 10k test
set** by normalized-question hash (pinocchio-raw is ITALIC's source corpus —
the dedup removed ~50 real overlaps):

- `mii-llm/pinocchio-raw` (raw-splits, streamed + reservoir-sampled per domain)
- `efederici/MMLU-Pro-ita`
- `sapienzanlp/mmlu_italian`

Final pool: **109,669 prompts**. Only questions are used for supervision
(pure KL, no correctness rewards); answers are kept for few-shot assistant
turns and dev accuracy tracking. Note: some large pinocchio-raw splits are
xet-backed and refuse HTTP range-streaming (403); they are skipped by default
(`--xet-fallback` downloads-then-deletes them if you have disk headroom).

## Results in detail

### Training run `opd-v1`

1000 steps, `batch_prompts=32`, `cot_fraction=0.25`, full reverse-KL, ~3.5h on
one 80GB A100. KL/token fell from ~4.2 at init to ~0.47 and stayed there.
Checkpoint accuracy on ITALIC (full-10k fast, official harness):

| Checkpoint | fast acc |
|---|---:|
| step_250 | **34.4%** (best) |
| step_500 | 35.8%†|
| step_750 | 36.0%†|
| final (1000) | 34.9%† |

† step_500/750/final numbers above are from an early 1k-subset tracker and are
subset-inflated; on the **full 10k** they collapse to ~34–35%, i.e. within
noise of step_250. The gain **saturates by step 250**; the extra 750 steps keep
lowering KL without adding accuracy. The bottleneck is capacity /
knowledge-transfer into a 0.4B model, not training length.

### Why CoT scores near-zero

ITALIC's `slow` (CoT) protocol requires the answer phrased as `Risposta: X`.
We decomposed the fast/slow gap (`scripts/analyze_cot.py`,
`scripts/analyze_cot_domain.py`) by re-parsing every saved output with the same
robust extractor, isolating three effects:

1. **Format/parser tax (artifact).** Terse models never write "Risposta:", so
   the official extractor returns empty and scores ~0 even when the letter is
   correct. Re-parsed robustly: baseline slow **0.00 → 0.34**, teacher slow
   **0.025 → 0.445**. Not a reasoning failure at all.
2. **Reasoning genuinely hurts (where it happens).** Under identical parsing,
   the CoT prompt *lowers* accuracy: opd250 −2 pts (broke 1841 correct answers,
   fixed 1627); teacher −6 pts (broke 1328, fixed 709). The more a model
   actually reasons, the more it loses. The baseline never reasons (1-char
   outputs) so it is flat.
3. **Task mismatch.** ITALIC's two macro-domains — *culture & commonsense* and
   *language capability* — are pure recall; reasoning is net-negative in
   **both** (no multi-step domain to help). Small models confabulate a rule and
   commit to it, overwriting a correct first-token prior (observed: a model
   states the plural of *uovo* is *uova* mid-reasoning, then answers *uovi*).

**Practical takeaway: submit in `fast` mode** with a short token cap. The one
thing distillation did for CoT — teaching the 0.4B student the `Risposta:`
format so it is the only model scoring non-zero slow (30.5%) — is moot here,
since CoT hurts on this benchmark.

### Evaluation harness

Two equivalent paths, cross-validated to within noise:

- **`scripts/eval_italic_fast.py`** (preferred) — local HF `generate`, verbatim
  ITALIC templates + extractors, robust parsing, `--max-new-tokens 8` to force
  a terse letter and avoid a verbose model mis-parsing its own ramble. No vLLM
  needed, so it is safe to run in the training venv.
- **`scripts/eval_italic_full.sh`** — the official `Crisp-Unimib/ITALIC`
  `run_eval.py` behind a vLLM OpenAI server, both modes (leaderboard-comparable).

Fresh full-10k fast scoring (friend's script, 8-tok cap, 0 unparsed) reproduces
the official harness: baseline 32.94 (vs 33.12), opd250 34.60 (vs 34.37),
teacher 50.71 (vs 50.69).

## Usage (A100 box)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login   # needs access to Coloss/* and mii-llm/pinocchio-raw

# 1. verify the tokenizer bridge assumptions (tokenizer files only)
python scripts/check_compat.py --teacher giux78/nesso-3B

# 2. build the prompt pool (one-time; streams pinocchio-raw, takes a while)
python scripts/prepare_prompts.py --out data/prompts.jsonl

# 3. train  (80GB A100: the flags below avoid a teacher+student OOM)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/train.py --config configs/distill.yaml \
  teacher=giux78/nesso-3B prompts_path=data/prompts.jsonl out_dir=runs/opd-v1 \
  score_micro_seqs=8 gradient_checkpointing=true

# 4. evaluate on the full 10k (fast protocol, no vLLM)
python scripts/eval_italic_fast.py runs/opd-v1/step_250 --italic-dir ./ITALIC

# 5. official/leaderboard numbers (needs vLLM): both modes
bash scripts/eval_italic_full.sh runs/opd-v1/step_250 opd250
```

On **LUMI (ROCm)**: same code; install the ROCm torch build first
(`pip install torch --index-url https://download.pytorch.org/whl/rocm6.2`)
and set `teacher_device` if you split models across GCDs.

## Layout

```
opd/formatting.py     ITALIC prompt templates (exact) + shot-regime sampling
opd/token_bridge.py   vocab-compat assertions + student<->teacher id bridging
opd/data.py           source normalizers, ITALIC dedup, pool load/split
opd/trainer.py        the on-policy distillation loop
scripts/check_compat.py       exhaustive tokenizer-compat check
scripts/prepare_prompts.py    build/dedup the prompt pool
scripts/train.py              training entrypoint
scripts/eval_italic_fast.py   preferred eval (local HF, fast protocol)
scripts/eval_italic_full.sh   official eval (vLLM + run_eval.py, both modes)
scripts/analyze_cot*.py       CoT fast/slow decomposition
scripts/eval_quick.py         early iteration tracker (superseded by eval_italic_fast)
```

## Findings & next steps

- **Distillation works but saturates early** (+1.7 full-10k fast by step 250);
  more steps only lower KL. Early-stop ~250.
- **opd250 became verbose** (it now reasons even in fast mode, ~700 chars),
  which slightly hurts fast extraction and inference speed. A cleaner recipe
  would keep the student terse (`cot_fraction=0`).
- **Ceiling is the teacher (50.7%).** To go higher: (a) `cot_fraction=0`,
  terse, early-stopped run to convert more of the knowledge gain cleanly;
  (b) a stronger teacher (nesso-4B-v2) via cross-tokenizer KL.
- `dev_acc` during training is a held-out pool slice, never ITALIC data.
- Default LR 1e-5 (full fine-tune); tinker's 1e-4 was for LoRA rank 128.
