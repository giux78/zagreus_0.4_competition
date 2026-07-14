"""On-policy distillation trainer.

Per step (mirrors tinker-cookbook's on_policy_distillation, self-contained):
  1. draw ITALIC-formatted prompts, render with the STUDENT's chat template
  2. student samples completions (temperature 1.0) with its current weights
     -> exactly on-policy, single gradient step per batch, no importance sampling
  3. teacher scores the same completions conditioned on the SAME conversation
     rendered with the TEACHER's chat template (see token_bridge for why)
  4. loss = full-distribution reverse KL over completion tokens
     sum_v p_student(v) * (log p_student(v) - log p_teacher(v))
     ("sampled_rkl" reproduces tinker's sampled-token REINFORCE variant)
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import shutil
import math
import os
import random
import re
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import token_bridge
from .data import load_pool, split_pool
from .formatting import PromptRenderer, build_messages, load_italic_shots


@dataclass
class DistillConfig:
    student: str = "mii-llm/nesso-0.4B-agentic"
    teacher: str = "Coloss/nesso-3B"
    prompts_path: str = "data/prompts.jsonl"
    italic_shots_path: str = "data/5_shots.jsonl"
    out_dir: str = "runs/opd"

    # batch / sampling
    batch_prompts: int = 32          # prompts per optimizer step
    group_size: int = 1              # rollouts per prompt (>1 only useful for CoT)
    temperature: float = 1.0
    max_new_tokens: int = 16         # fast mode: the answer is a letter
    cot_fraction: float = 0.0        # fraction of prompts using the CoT template
    cot_max_new_tokens: int = 300
    p_italic_shots: float = 0.5
    p_pool_shots: float = 0.25
    pool_shots_max_k: int = 5

    # optimization
    steps: int = 2000
    lr: float = 1e-5
    warmup_steps: int = 50
    lr_schedule: str = "cosine"      # cosine | constant
    grad_clip: float = 1.0
    loss_fn: str = "full_kl"         # full_kl | sampled_rkl
    seed: int = 0

    # memory
    gen_micro_seqs: int = 64         # sequences per generate() call
    score_micro_seqs: int = 32       # sequences per scoring forward
    gradient_checkpointing: bool = False
    teacher_device: str = ""         # default: same device as student

    # bookkeeping
    dev_size: int = 500
    eval_every: int = 200
    eval_dev_samples: int = 200
    save_every: int = 500
    keep_checkpoints: int = 3        # newest step_* dirs kept on disk (0 = keep all)
    log_every: int = 10
    wandb_project: str = ""          # empty = disabled


LETTER_RE = re.compile(r"\b([A-J])\b")


def load_causal_lm(name: str, dtype: torch.dtype):
    """from_pretrained across the transformers 4.x (torch_dtype) / 5.x (dtype) rename."""
    try:
        return AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Trainer:
    def __init__(self, cfg: DistillConfig):
        self.cfg = cfg
        self.device = pick_device()
        self.teacher_device = cfg.teacher_device or self.device
        self.rng = random.Random(cfg.seed)
        torch.manual_seed(cfg.seed)
        os.makedirs(cfg.out_dir, exist_ok=True)

        print(f"[trainer] loading tokenizers ({cfg.student} / {cfg.teacher})")
        self.s_tok = AutoTokenizer.from_pretrained(cfg.student)
        self.t_tok = AutoTokenizer.from_pretrained(cfg.teacher)
        token_bridge.assert_compatible(self.s_tok, self.t_tok)
        self.bridge = token_bridge.Bridge()
        self.s_bos = self.s_tok.convert_tokens_to_ids("<|begin_of_text|>")
        self.s_pad = self.s_tok.pad_token_id or token_bridge.END_OF_TEXT
        self.t_pad = self.t_tok.pad_token_id or token_bridge.END_OF_TEXT

        print(f"[trainer] loading student (fp32+autocast) on {self.device}")
        self.student = load_causal_lm(cfg.student, torch.float32).to(self.device)
        if cfg.gradient_checkpointing:
            self.student.gradient_checkpointing_enable()
        print(f"[trainer] loading teacher (bf16, frozen) on {self.teacher_device}")
        self.teacher = load_causal_lm(cfg.teacher, torch.bfloat16).to(self.teacher_device)
        self.teacher.eval().requires_grad_(False)

        print(f"[trainer] loading prompt pool from {cfg.prompts_path}")
        pool = load_pool(cfg.prompts_path)
        self.train_rows, self.dev_rows = split_pool(pool, cfg.dev_size, cfg.seed)
        print(f"[trainer] pool: {len(self.train_rows)} train / {len(self.dev_rows)} dev")
        self.italic_shots = load_italic_shots(cfg.italic_shots_path)
        self.renderer = PromptRenderer(
            self.train_rows, self.italic_shots,
            p_italic_shots=cfg.p_italic_shots, p_pool_shots=cfg.p_pool_shots,
            pool_shots_max_k=cfg.pool_shots_max_k, cot_fraction=cfg.cot_fraction,
            rng=self.rng,
        )

        self.opt = torch.optim.AdamW(self.student.parameters(), lr=cfg.lr, weight_decay=0.0)
        self.wandb = None
        if cfg.wandb_project:
            import wandb
            self.wandb = wandb
            wandb.init(project=cfg.wandb_project, config=dataclasses.asdict(cfg))

    # ------------------------------------------------------------------ utils

    def _lr_at(self, step: int) -> float:
        cfg = self.cfg
        if step < cfg.warmup_steps:
            return cfg.lr * (step + 1) / cfg.warmup_steps
        if cfg.lr_schedule == "constant":
            return cfg.lr
        t = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
        return cfg.lr * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    def _encode_prompt(self, tok, messages, bos_id: int) -> list[int]:
        text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        ids = tok.encode(text, add_special_tokens=False)
        if not ids or ids[0] != bos_id:
            ids = [bos_id] + ids
        return ids

    @staticmethod
    def _right_pad(seqs: list[list[int]], pad: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        T = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), T), pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), T), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            mask[i, : len(s)] = 1
        return ids.to(device), mask.to(device)

    # -------------------------------------------------------------- generation

    @torch.no_grad()
    def _generate(self, prompt_ids: list[list[int]], max_new_tokens: int) -> list[list[int]]:
        """Sample one completion per prompt (already replicated for group_size)."""
        cfg = self.cfg
        self.student.eval()
        completions: list[list[int]] = []
        for i in range(0, len(prompt_ids), cfg.gen_micro_seqs):
            chunk = prompt_ids[i : i + cfg.gen_micro_seqs]
            # left padding so all prompts end at the same position
            T = max(len(s) for s in chunk)
            ids = torch.full((len(chunk), T), self.s_pad, dtype=torch.long)
            mask = torch.zeros((len(chunk), T), dtype=torch.long)
            for j, s in enumerate(chunk):
                ids[j, T - len(s):] = torch.tensor(s, dtype=torch.long)
                mask[j, T - len(s):] = 1
            with torch.autocast(self.device.split(":")[0], dtype=torch.bfloat16,
                                enabled=self.device != "cpu"):
                out = self.student.generate(
                    ids.to(self.device), attention_mask=mask.to(self.device),
                    do_sample=True, temperature=cfg.temperature, top_p=1.0, top_k=0,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=list(self.bridge.stop_ids),
                    pad_token_id=self.s_pad,
                )
            for j in range(len(chunk)):
                raw = out[j, T:].tolist()
                completions.append(self.bridge.clean_completion(raw))
        self.student.train()
        return completions

    # ----------------------------------------------------------------- scoring

    @staticmethod
    def _gather_logits(model, seqs, plens, lens, pad, device, autocast_dev=None):
        """Forward `seqs`, return lm_head logits only at completion positions.

        Position p predicts token p+1, so for a completion of length L starting
        at index P (= prompt length) we need hidden states at P-1 .. P+L-2.
        Full logits over 128k vocab for every position would not fit; gathering
        hidden states first keeps memory at N_completion_tokens x vocab.
        """
        ids, mask = Trainer._right_pad(seqs, pad, device)
        # grad-vs-no-grad is decided by the caller's context, not here
        ctx = (torch.autocast(autocast_dev, dtype=torch.bfloat16)
               if autocast_dev else contextlib.nullcontext())
        with ctx:
            h = model.model(input_ids=ids, attention_mask=mask).last_hidden_state
            B, T, H = h.shape
            flat = []
            for i, (P, L) in enumerate(zip(plens, lens)):
                flat.extend(range(i * T + P - 1, i * T + P - 1 + L))
            h_sel = h.reshape(B * T, H)[torch.tensor(flat, device=device)]
            logits = model.lm_head(h_sel)  # (N, vocab)
        return logits

    def _loss_on_chunk(self, s_seqs, t_seqs, plens_s, plens_t, lens, targets_t):
        """Loss for a micro-batch of rollouts. Returns (loss, n_tokens, stats)."""
        cfg = self.cfg
        V = self.bridge.shared_vocab_size

        with torch.no_grad():
            t_logits = self._gather_logits(
                self.teacher, t_seqs, plens_t, lens, self.t_pad, self.teacher_device
            )
            logp_t = F.log_softmax(t_logits.float(), dim=-1).to(self.device)  # (N, V)

        s_logits = self._gather_logits(
            self.student, s_seqs, plens_s, lens, self.s_pad, self.device,
            autocast_dev=self.device.split(":")[0] if self.device != "cpu" else None,
        )
        logp_s_full = F.log_softmax(s_logits.float(), dim=-1)  # (N, 128262)

        tgt = torch.tensor(targets_t, dtype=torch.long, device=self.device)  # (N,)

        # merge student's <|im_end|> mass into the teacher's <|eot_id|> slot
        p_full = logp_s_full.exp()
        p_shared = p_full[:, :V].clone()
        p_shared[:, token_bridge.TEACHER_EOT_ID] += p_full[:, token_bridge.STUDENT_IM_END]
        residual = 1.0 - p_shared.sum(-1)  # mass on tool tokens, should be ~0
        logp_s = (p_shared + 1e-12).log()

        kl = (p_shared * (logp_s - logp_t)).sum(-1)  # (N,) full reverse KL
        sampled_kl = (logp_s.gather(-1, tgt[:, None]) - logp_t.gather(-1, tgt[:, None])).squeeze(-1)

        if cfg.loss_fn == "full_kl":
            loss = kl.sum()
        elif cfg.loss_fn == "sampled_rkl":
            # tinker-style: REINFORCE with per-token advantage = -sampled KL
            logp_s_tgt = logp_s.gather(-1, tgt[:, None]).squeeze(-1)
            loss = (logp_s_tgt * sampled_kl.detach()).sum()
        else:
            raise ValueError(cfg.loss_fn)

        stats = {
            "kl": kl.detach().mean().item(),
            "sampled_kl": sampled_kl.detach().mean().item(),
            "residual_mass": residual.detach().mean().item(),
        }
        return loss, len(targets_t), stats

    # ------------------------------------------------------------------- train

    def train(self):
        cfg = self.cfg
        t0 = time.time()
        for step in range(cfg.steps):
            for g in self.opt.param_groups:
                g["lr"] = self._lr_at(step)

            # 1. draw prompts, render for both models
            batch = []
            for _ in range(cfg.batch_prompts):
                messages, row, fast = self.renderer.sample()
                s_prompt = self._encode_prompt(self.s_tok, messages, self.s_bos)
                t_prompt = self._encode_prompt(self.t_tok, messages, self.s_bos)
                for _ in range(cfg.group_size):
                    batch.append({"s_prompt": s_prompt, "t_prompt": t_prompt,
                                  "row": row, "fast": fast})

            # 2. student samples completions (on-policy)
            fast_idx = [i for i, b in enumerate(batch) if b["fast"]]
            cot_idx = [i for i, b in enumerate(batch) if not b["fast"]]
            completions: dict[int, list[int]] = {}
            for idx, mnt in ((fast_idx, cfg.max_new_tokens), (cot_idx, cfg.cot_max_new_tokens)):
                if idx:
                    comps = self._generate([batch[i]["s_prompt"] for i in idx], mnt)
                    completions.update(dict(zip(idx, comps)))

            rollouts = [(batch[i], completions[i]) for i in range(len(batch))
                        if len(completions[i]) > 0]
            if not rollouts:
                print(f"[step {step}] no non-empty completions, skipping")
                continue

            # 3+4. teacher scoring + reverse-KL update (micro-batched, grad accum)
            self.opt.zero_grad(set_to_none=True)
            n_total = sum(len(c) for _, c in rollouts)
            agg = {"kl": 0.0, "sampled_kl": 0.0, "residual_mass": 0.0}
            for i in range(0, len(rollouts), cfg.score_micro_seqs):
                chunk = rollouts[i : i + cfg.score_micro_seqs]
                s_seqs, t_seqs, plens_s, plens_t, lens, targets = [], [], [], [], [], []
                for b, comp in chunk:
                    comp_t = self.bridge.to_teacher(comp)
                    s_seqs.append(b["s_prompt"] + comp[:-1])
                    t_seqs.append(b["t_prompt"] + comp_t[:-1])
                    plens_s.append(len(b["s_prompt"]))
                    plens_t.append(len(b["t_prompt"]))
                    lens.append(len(comp))
                    targets.extend(comp_t)
                loss, n_tok, stats = self._loss_on_chunk(
                    s_seqs, t_seqs, plens_s, plens_t, lens, targets
                )
                (loss / n_total).backward()
                for k in agg:
                    agg[k] += stats[k] * n_tok / n_total

            torch.nn.utils.clip_grad_norm_(self.student.parameters(), cfg.grad_clip)
            self.opt.step()

            # ------------------------------------------------------- logging
            if step % cfg.log_every == 0:
                mean_len = n_total / len(rollouts)
                fmt_ok = sum(
                    1 for b, comp in rollouts
                    if (m := LETTER_RE.search(self.s_tok.decode(comp)))
                    and m.group(1) in {l for l, _ in b["row"]["options"]}
                ) / len(rollouts)
                msg = (f"[step {step}] kl/tok={agg['kl']:.4f} "
                       f"sampled_kl={agg['sampled_kl']:.4f} len={mean_len:.1f} "
                       f"fmt_ok={fmt_ok:.2f} lr={self.opt.param_groups[0]['lr']:.2e} "
                       f"({time.time() - t0:.0f}s)")
                print(msg, flush=True)
                if self.wandb:
                    self.wandb.log({"kl": agg["kl"], "sampled_kl": agg["sampled_kl"],
                                    "residual_mass": agg["residual_mass"],
                                    "completion_len": mean_len, "format_ok": fmt_ok,
                                    "lr": self.opt.param_groups[0]["lr"]}, step=step)

            if cfg.eval_every and step and step % cfg.eval_every == 0:
                acc = self.eval_dev()
                print(f"[step {step}] dev_acc={acc:.4f}", flush=True)
                if self.wandb:
                    self.wandb.log({"dev_acc": acc}, step=step)

            if cfg.save_every and step and step % cfg.save_every == 0:
                self._save(f"step_{step}")

        acc = self.eval_dev()
        print(f"[final] dev_acc={acc:.4f}")
        self._save("final")

    # -------------------------------------------------------------------- eval

    @torch.no_grad()
    def eval_dev(self) -> float:
        """Greedy 5-shot fast-mode accuracy on the held-out dev slice."""
        cfg = self.cfg
        rows = self.dev_rows[: cfg.eval_dev_samples]
        self.student.eval()
        correct = 0
        for i in range(0, len(rows), cfg.gen_micro_seqs):
            chunk = rows[i : i + cfg.gen_micro_seqs]
            prompts = [
                self._encode_prompt(
                    self.s_tok,
                    build_messages(r, few_shots=self.italic_shots, fast=True),
                    self.s_bos,
                )
                for r in chunk
            ]
            T = max(len(p) for p in prompts)
            ids = torch.full((len(chunk), T), self.s_pad, dtype=torch.long)
            mask = torch.zeros((len(chunk), T), dtype=torch.long)
            for j, p in enumerate(prompts):
                ids[j, T - len(p):] = torch.tensor(p, dtype=torch.long)
                mask[j, T - len(p):] = 1
            with torch.autocast(self.device.split(":")[0], dtype=torch.bfloat16,
                                enabled=self.device != "cpu"):
                out = self.student.generate(
                    ids.to(self.device), attention_mask=mask.to(self.device),
                    do_sample=False, max_new_tokens=8,
                    eos_token_id=list(self.bridge.stop_ids), pad_token_id=self.s_pad,
                )
            for j, r in enumerate(chunk):
                text = self.s_tok.decode(
                    self.bridge.clean_completion(out[j, T:].tolist())[:-1]
                    or out[j, T:].tolist()
                )
                m = LETTER_RE.search(text)
                if m and m.group(1) == r["answer"]:
                    correct += 1
        self.student.train()
        return correct / max(1, len(rows))

    def _save(self, name: str):
        path = os.path.join(self.cfg.out_dir, name)
        print(f"[trainer] saving checkpoint -> {path}")
        self.student.save_pretrained(path)
        self.s_tok.save_pretrained(path)
        with open(os.path.join(path, "distill_config.json"), "w") as f:
            json.dump(dataclasses.asdict(self.cfg), f, indent=2)
        self._prune_checkpoints()

    def _prune_checkpoints(self):
        """Keep only the newest keep_checkpoints step_* dirs ("final" is exempt)."""
        keep = self.cfg.keep_checkpoints
        if keep <= 0:
            return
        steps = sorted(
            (d for d in os.listdir(self.cfg.out_dir) if d.startswith("step_")),
            key=lambda d: int(d.split("_")[1]),
        )
        for d in steps[:-keep]:
            victim = os.path.join(self.cfg.out_dir, d)
            print(f"[trainer] pruning old checkpoint {victim}")
            shutil.rmtree(victim, ignore_errors=True)
