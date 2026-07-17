# zagreus_competition — on-policy distillation for ITALIC

Improve **nesso-0.4B-agentic** (built on zagreus-0.4-ita) on the
[ITALIC](https://github.com/Crisp-Unimib/ITALIC) benchmark by on-policy
distillation from **Coloss/nesso-3B**, following the logic of
[tinker-cookbook's on_policy_distillation](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/recipes/distillation/on_policy_distillation.py)
but fully self-contained (PyTorch + transformers, no Tinker API).

The `opd/` package in this repo was the v1 prototype; from v2 on, the
library code lives in **[mii-llm/palingenesis](https://github.com/mii-llm/palingenesis)**
(branch `odp`, `pgs distill` / `pgs distill-score`) and this repo is the
experiment record: data policy scripts, eval harnesses, analyses, results.

**Best model released: [giux78/zagreus_0.4_competition](https://huggingface.co/giux78/zagreus_0.4_competition)**
(= `opd_v3/step_550` below).

## TL;DR results

Full 10,000-question ITALIC, 5-shot. `fast` = direct letter answer (the
protocol that matters); slow = CoT. Two cross-validated harnesses: *informal* =
`scripts/eval_italic_fast.py` (verbatim templates, 8-token cap); *official* =
`Crisp-Unimib/ITALIC run_eval.py` via vLLM.

| Model | fast (informal) | fast (official) | slow (official) |
|---|---:|---:|---:|
| `nesso-0.4B-agentic` (baseline student) | 32.94 | 33.12 | 0.00\* |
| `opd-v1/step_250` (v1 prototype, this repo) | 34.60 | 34.37 | 30.49 |
| `opd_v2/step_250` (palingenesis, unfiltered pool) | 35.44 | — | — |
| **`opd_v3/step_550` (palingenesis, filtered pool — released)** | **37.06** | **37.2** | **33.2** |
| `nesso-3B` (teacher, the ceiling) | 50.71 | 50.69 | 2.51\* |

\* The near-zero slow scores are a **scoring artifact**, not a reasoning
collapse — see [Why CoT scores near-zero](#why-cot-scores-near-zero).

**Net effect: +4.1 points official fast** (33.12 → 37.2). Random chance is
~25% (mostly 4-option). The teacher sits at 50.7%, so ~13.5 points of headroom
remain. Per-domain (v3, informal): culture & commonsense 39.7, language
capability 33.1.

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

### Training run `opd_v2` (palingenesis, fast-only)

Same recipe re-run through `palingenesis.opd` (`pgs distill`) with one change:
`cot_fraction 0` — all supervision in the fast format. 600 steps, batch 32,
~50 min on one A100 (needs `score_micro_seqs 8` + `gradient_checkpointing`,
otherwise the student scoring forward OOMs the 80GB card). Full-10k fast:
step_250 **35.44**, step_500 35.38 (tied). +0.8 over v1 purely from not
splitting capacity with CoT.

### Training run `opd_v3` (filtered pool — the released model)

Four combined changes over v2, all data-side:

1. **Teacher-correct filtering** (`pgs distill-score` + `scripts/build_pool_v3.py`):
   pure reverse KL distills the teacher's *errors* too — and nesso-3B is right
   on only **43.7%** of the pool (pinocchio 48.1, mmlu_italian 40.2,
   mmlu_pro_ita 16.2 — 10-option questions). Scoring is one batched forward
   per row (option-letter logits at the last prompt position, no generation);
   the filtered pool keeps 47,933 rows.
2. **Italian-language upweight ×4**: ITALIC is 40% "language capability" but
   the pool has only ~990 such rows pre-filter (570 post) — vocabulary, verbal
   comprehension, grammar. Foreign-language quizzes are not upweighted.
3. **`p_reference_shots 0.8`**: the official harness always sends the same 5
   official shots, so train mostly on exactly that distribution.
4. **`max_new_tokens 8` at train time**: terse supervision; kills the
   verbosity drift v1/v2 showed (and with it the fast-mode misparse tax —
   official fast now reads *at* the informal number, not below it).

600 steps, ~40 min. Dev accuracy climbs monotonically (unlike v2's plateau)
and was **still rising at the last evaluated checkpoint**: full-10k fast
step_250 36.61 → step_350 36.75 → **step_550 37.06** (official: **fast 37.2,
slow 33.2** — both official modes now parse cleanly). Released as
[giux78/zagreus_0.4_competition](https://huggingface.co/giux78/zagreus_0.4_competition).

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

## Reproducing the released model (opd_v3, via palingenesis)

```bash
# this repo: data + eval scripts
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login   # needs access to Coloss/* and mii-llm/pinocchio-raw

# the library (branch odp)
git clone -b odp https://github.com/mii-llm/palingenesis ../palingenesis
export PYTHONPATH=../palingenesis/src

# 1. build the raw prompt pool (one-time; streams pinocchio-raw, takes a while)
python scripts/prepare_prompts.py --out data/prompts.jsonl

# 2. annotate every row with the teacher's own answer (~2h on an A100)
python -m palingenesis.opd.score_pool \
  --config ../palingenesis/configs/distill_opd.yaml \
  --out data/prompts_scored.jsonl \
  --model.teacher giux78/nesso-3B --data.prompts_path data/prompts.jsonl \
  --data.shots_path data/5_shots.jsonl

# 3. policy: keep teacher-correct rows, upweight Italian-language x4
python scripts/build_pool_v3.py --scored data/prompts_scored.jsonl --out data/prompts_v3.jsonl

# 4. train (~40 min on one 80GB A100; memory flags are the config defaults)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m palingenesis.opd.trainer \
  --config ../palingenesis/configs/distill_opd.yaml \
  --model.teacher giux78/nesso-3B \
  --data.prompts_path data/prompts_v3.jsonl --data.shots_path data/5_shots.jsonl \
  --data.p_reference_shots 0.8 --data.p_pool_shots 0.1 \
  --sampling.max_new_tokens 8 \
  --train.output_dir runs/opd_v3 --train.steps 600 \
  --train.eval_every 50 --train.save_steps 50 --train.keep_checkpoints 0

# 5. evaluate all candidate checkpoints on the full 10k (fast protocol, no vLLM)
python scripts/eval_italic_fast.py runs/opd_v3/step_250 runs/opd_v3/step_350 \
  runs/opd_v3/step_550 --italic-dir ./ITALIC --max-new-tokens 8

# 6. official/leaderboard numbers for the winner (needs vLLM): both modes
bash scripts/eval_italic_full.sh runs/opd_v3/step_550 opd-v3-550
```

With a `pip install -e ../palingenesis` the `python -m palingenesis.opd.*`
invocations become `pgs distill-score` / `pgs distill`.

<details>
<summary>v1 prototype reproduction (self-contained opd/ package)</summary>

```bash
python scripts/check_compat.py --teacher giux78/nesso-3B
python scripts/prepare_prompts.py --out data/prompts.jsonl
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python scripts/train.py --config configs/distill.yaml \
  teacher=giux78/nesso-3B prompts_path=data/prompts.jsonl out_dir=runs/opd-v1 \
  score_micro_seqs=8 gradient_checkpointing=true
python scripts/eval_italic_fast.py runs/opd-v1/step_250 --italic-dir ./ITALIC
```

</details>

On **LUMI (ROCm)**: same code; install the ROCm torch build first
(`pip install torch --index-url https://download.pytorch.org/whl/rocm6.2`)
and set `teacher_device` if you split models across GCDs.

## Layout

```
opd/*                 v1 prototype (superseded by palingenesis.opd, kept for the record)
scripts/check_compat.py       exhaustive tokenizer-compat check (v1)
scripts/prepare_prompts.py    build/dedup the raw prompt pool
scripts/build_pool_v3.py      policy: teacher-correct filter + language upweight (v3)
scripts/train.py              v1 training entrypoint (v2/v3 train via palingenesis)
scripts/eval_italic_fast.py   preferred eval (local HF, fast protocol)
scripts/eval_italic_full.sh   official eval (vLLM + run_eval.py, both modes)
scripts/analyze_cot*.py       CoT fast/slow decomposition
scripts/eval_quick.py         early iteration tracker (superseded by eval_italic_fast)
```

## Findings & next steps

- **What moved the needle, in order:** teacher-correct pool filtering +
  eval-distribution matching + terse supervision (v3, +1.6 over v2);
  fast-only training (v2, +0.8 over v1); base on-policy KL (v1, +1.7 over
  baseline). Total: official fast 33.12 → 37.2.
- **Don't distill the teacher's errors.** The teacher is right on only 43.7%
  of the pool; filtering to teacher-correct rows changed the training dynamic
  (dev accuracy climbs monotonically instead of plateauing at step 250) —
  and v3 was **still improving at step 550**, so a longer run at the same
  settings is the cheapest untried gain.
- **Terse supervision fixes verbosity.** v1/v2 drifted to the token cap and
  paid a misparse tax in the official harness; v3 trained at
  `max_new_tokens 8` answers with the bare letter and scores identically
  under both harnesses — and parses in *both* official modes (slow 33.2 vs
  literally 0.00 for the baseline).
- **The remaining structural gap is data, not training:** ITALIC is 40%
  language capability, our pool has ~990 Italian-language rows (570 after
  filtering). v3's language score is 33.1 vs 39.7 culture. New Italian
  grammar/vocabulary sources would move the total more than any knob.
- Other untried levers: a stronger teacher (nesso-4B-v2) via cross-tokenizer
  KL; a small gold-answer CE term alongside the KL (the only route past the
  teacher's 50.7% ceiling).
- `dev_acc` during training is a held-out pool slice, never ITALIC data
  (v3's dev is drawn from the teacher-correct pool, so it reads high —
  in-run signal only, not comparable across runs).
- Default LR 1e-5 (full fine-tune); tinker's 1e-4 was for LoRA rank 128.
