#!/bin/bash
# The 10-hour two-stage pipeline: SFT (zagreus-0.4B seed + OpenItalianData
# selection, palingenesis) -> on-policy distillation from nesso-3B.
# Prereqs (done outside the clock): seed model built, selection written,
# prompts_v3 pool present.
set -euo pipefail
T0=$(date +%s)
log() { echo "[pipeline +$(( ($(date +%s)-T0)/60 ))min] $*"; }

PY=/home/ecuser/ai/vllm-cu129/bin
export PYTHONPATH=/home/ecuser/ai/palingenesis/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CFG=/home/ecuser/ai/zagreus_competition/configs

log "STAGE 1: SFT (zagreus-0.4B-seed on OpenItalianData selection, lr 1e-3)"
$PY/torchrun --standalone --nproc_per_node=1 -m palingenesis.train \
  --config $CFG/sft_openita.yaml
log "STAGE 1 done -> /home/ecuser/ai/palingenesis/runs/sft_openita/final"

log "STAGE 2: OPD (SFT model <- nesso-3B, ITALIC recipe, 900 steps)"
$PY/python3 -m palingenesis.opd.trainer --config $CFG/distill_stage2.yaml
log "STAGE 2 done -> /home/ecuser/ai/palingenesis/runs/opd_v5"

log "PIPELINE COMPLETE"
