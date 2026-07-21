# ITALIC ↔ pinocchio decontamination

ITALIC is derived from **pinocchio** (confirmed by ITALIC's author), which is
also the main source of our training pool. So any pool built from pinocchio
risks containing the benchmark's own questions — training on the test set.

Our original dedup normalized each question (lowercase, strip accents,
alphanumeric-only) and hashed it. That catches *identical* questions but
**reports 0 overlap** on the real leakage, because ITALIC reworded many stems:

```
ITALIC:    "In base all'art. 68 della Costituzione possono i membri del
            Parlamento essere sottoposti ad intercettazioni…?"
pinocchio: "Per sottoporre i membri del Parlamento ad intercettazioni, in
            qualsiasi forma, di conversazioni o comunicazioni…"
```

Same question, different words — invisible to hashing (surface overlap 0.17).
This document is the method and the results of finding and removing them.

## The right corpus

Use **`efederici/pinocchio`, config `text`** (102,573 rows) — *not*
`mii-llm/pinocchio-raw`, which is a different/partial dump. Download and
normalize to the pool schema:

```bash
python scripts/build_pinocchio_text.py --out data/pinocchio_text.jsonl
```

## Why the obvious methods fail

We tried three cheap matchers; each fails in a different way (see the sample
output in each script's `--out` report):

| method | script | failure |
|---|---|---|
| exact normalized hash | (original) | misses every reworded duplicate (finds 0) |
| surface question LSH (MinHash) | `dedup_lsh_check.py` | template false-positives ("sinonimo di *calesse*" ≈ "*ottemperanza*"); still misses rewordings |
| answer-option set match | `match_italic_options.py` | catches rewordings, but false-positives when unrelated questions draw options from a shared space (integers, city/region names, Vero/Falso) |

The lesson: a genuine duplicate has **both** signals — the questions mean the
same thing **and** they offer the same answer choices. Neither alone separates
"same question reworded" from "same template, different item".

## The robust method

`match_italic_semantic.py` combines them:

1. **Candidate generation** — embed questions with a multilingual *paraphrase*
   model (`paraphrase-multilingual-mpnet-base-v2`), cosine kNN. Semantic
   embeddings catch rewordings that surface n-grams miss.
2. **Verification** — accept only when semantic similarity is high **and**
   (answer-option overlap **or** salient-content-token overlap). The
   content-token guard rejects vocabulary templates: "sinonimo di X" only
   matches same-X, because the quoted lemma is the sole salient token.

```
match  =  (qsim ≥ q_lo AND option_jaccard ≥ opt_min)
       OR (qsim ≥ q_hi AND content_token_jaccard ≥ ctok_min)
```

`italic_coverage.py` reports the *distribution* of each ITALIC question's best
pinocchio neighbor — this is what tells you "present but below threshold" vs
"genuinely absent", and bounds the true overlap.

## Decontaminating a pool

`decontaminate_pool.py` runs the reverse direction: for every **pool** row,
compute its max cosine similarity to *any* ITALIC question and drop it if
≥ threshold. The cost is asymmetric — over-removing wastes cheap data,
under-removing leaks the test set — so we drop on the semantic signal alone at
a conservative **0.80**:

```bash
python scripts/decontaminate_pool.py \
    --pool data/pinocchio_text.jsonl data/mmlu_only.jsonl \
    --italic ~/ai/ITALIC/italic.jsonl --threshold 0.80 \
    --out data/clean_pool_raw.jsonl
```

## Results

### How much of ITALIC is in pinocchio-text

Best-neighbor semantic distribution over the full 102,573-row corpus:

| ITALIC's best pinocchio neighbor | questions | reading |
|---|---:|---|
| ≥ 0.99 (verbatim) | 1,306 | identical |
| 0.95–0.99 | 342 | trivial edits |
| 0.90–0.95 | 846 | clear duplicate |
| 0.85–0.90 | 1,172 | likely reworded |
| 0.70–0.85 | 5,034 | same template, mostly *different* question |
| < 0.70 | 1,300 | unrelated |

- **Verified duplicates** (sem ≥ 0.90 and option/content agree): **1,545 (15.4 %)**.
- Loose upper bound (any ≥ 0.85 neighbor): **3,666 (36.7 %)**.
- So "almost all of ITALIC is in pinocchio" holds only loosely — ~15 % are
  solid duplicates, ≤ 37 % at the most generous threshold. The large 0.70–0.85
  mass is same-topic/template, not the same question.

### How much leaked into our training

The published runs (v3, v5) trained on a pool whose pinocchio rows were
reservoir-sampled and exact-hash-deduped. Matching that pool against ITALIC:

- exact-hash reported **0**;
- the robust matcher finds **168** reworded duplicates that slipped through
  (0.34 % of a ~49 k pool).

### Did the contamination inflate the scores? No.

We rebuilt a fully clean pool — pinocchio-text + the two MMLU sources,
decontaminated at sem ≥ 0.80 (**9,127 rows dropped**, → 118,801 clean; then
teacher-correct filtered + Italian-language ×4 → 55,088-row training pool) —
and re-ran both pipelines with the identical recipe:

| line | contaminated pool | clean pool | inflation |
|---|---:|---:|---:|
| OPD from `nesso-0.4B-agentic` (v3 recipe, 600 steps) | 37.2 official / 37.06 informal | **36.8** | **~0.4** |
| SFT → OPD from `zagreus-0.4B` seed (900 OPD steps) | 32.2 | **31.7** | **~0.5** |

Both independent lines agree: removing every ITALIC near-duplicate costs only
**~0.4–0.5 points**. The reworded leakage exact-hash missed inflated the
headline by well under a point. **The released results are real.**

## Reproduction (A100 box)

```bash
# 1. correct corpus
python scripts/build_pinocchio_text.py --out data/pinocchio_text.jsonl

# 2. (optional) measure ITALIC↔pinocchio overlap and inspect the method
python scripts/italic_coverage.py       --pool data/pinocchio_text.jsonl --italic ITALIC/italic.jsonl
python scripts/match_italic_semantic.py --pool data/pinocchio_text.jsonl --italic ITALIC/italic.jsonl
#   (dedup_lsh_check.py / match_italic_options.py show the methods that fail)

# 3. decontaminate the training pool
python scripts/decontaminate_pool.py \
    --pool data/pinocchio_text.jsonl data/mmlu_only.jsonl \
    --italic ITALIC/italic.jsonl --threshold 0.80 --out data/clean_pool_raw.jsonl

# 4. score + filter + train as usual (palingenesis pgs distill-score / distill)
```

Dependency: `pip install sentence-transformers datasketch`.
Model: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`.
