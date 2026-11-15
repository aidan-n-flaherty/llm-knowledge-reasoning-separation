"""Shared training loop.

    python -m krs.train --config configs/vanilla.yaml --set seed=1 run_name=vanilla-s1

Resumes automatically from <out_dir>/<run_name>/last.pt if it exists, so a Slurm job that hits its
time limit can simply be resubmitted.
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import yaml

from krs.config import load_config
from krs.data.dataset import collate, load_examples
from krs.evaluate import evaluate, format_results, load_eval_suites
from krs.model.memory import MemoryContext
from krs.model.transformer import Llama3Model
from krs.objectives import MASKED_FACT_LOSSES, lm_loss
from krs.utils import autocast_for, load_data_meta, load_tokenizer, pick_device, save_checkpoint


def get_lr(step, cfg):
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (cfg.lr - cfg.min_lr)


def build_optimizer(model, cfg):
    # memory tables are excluded from weight decay: a rarely visited entity would otherwise decay away
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (decay if p.dim() >= 2 and not name.startswith("memory.") else no_decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr,
        betas=(0.9, 0.95),
    )


class BatchSampler:
    """Deterministic in (seed, step), so a resumed run sees the same batches as an uninterrupted one."""

    def __init__(self, n, batch_size, seed):
        self.n, self.batch_size, self.seed = n, batch_size, seed
        self.steps_per_epoch = max(1, n // batch_size)
        self.epoch, self.perm = -1, None

    def indices(self, step):
        epoch, i = divmod(step, self.steps_per_epoch)
        if epoch != self.epoch:
            self.perm = list(range(self.n))
            random.Random(self.seed * 1_000_003 + epoch).shuffle(self.perm)
            self.epoch = epoch
        return self.perm[i * self.batch_size : (i + 1) * self.batch_size]


def compute_loss(model, batch, obj):
    """Memory-on pass over everything, plus (optionally) a memory-masked pass."""
    logits = model(batch.tokens, MemoryContext(batch.entity_ids))
    parts = {"on": lm_loss(logits, batch.labels)}
    loss = parts["on"]
    if obj.masked_pass:
        logits_off = model(batch.tokens, MemoryContext.off(batch.entity_ids))
        parts["off_nonfact"] = lm_loss(logits_off, batch.labels.masked_fill(batch.fact_mask, -100))
        parts["off_fact"] = MASKED_FACT_LOSSES[obj.masked_fact_loss](logits_off, batch)
        loss = loss + obj.lambda_nonfact * parts["off_nonfact"] + obj.lambda_fact * parts["off_fact"]
    return loss, {k: v.item() for k, v in parts.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="e.g. seed=1 train.lr=1e-4")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    tc = cfg.train

    run_dir = Path(cfg.out_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict()))
    log_path, ckpt_path = run_dir / "log.jsonl", run_dir / "last.pt"

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = pick_device()
    amp = autocast_for(device)
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "none"

    meta, splits = load_data_meta(cfg.data.dir)
    tok = load_tokenizer(cfg.data.dir)
    forget = set(splits["forget"]) if cfg.data.exclude_forget else set()
    train_examples = load_examples(
        Path(cfg.data.dir) / "train.jsonl", tok, cfg.data.max_len, keep=lambda ex: ex["entity_id"] not in forget
    )
    suites = load_eval_suites(cfg.data.dir, tok, cfg.data.max_len, tc.eval_max_examples, cfg.seed)

    model = Llama3Model(cfg, tok.vocab_size, meta["n_people"]).to(device)
    opt = build_optimizer(model, tc)
    step = 0
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        step = ckpt["step"]
        print(f"resumed from {ckpt_path} at step {step}")

    n_mem = sum(p.numel() for p in model.memory.parameters())
    n_all = sum(p.numel() for p in model.parameters())
    print(
        f"run {cfg.run_name} | device {device} ({gpu}) | data {meta['sha256'][:12]} | "
        f"{len(train_examples):,} train examples | params: base {n_all - n_mem:,} memory {n_mem:,}"
    )

    def log(record):
        with open(log_path, "a") as f:
            f.write(json.dumps({"step": step, **record}) + "\n")

    def run_eval():
        results = evaluate(model, suites, tok.pad, device, amp, meta["attr_prior_bits"])
        print(f"eval at step {step}\n{format_results(results)}")
        log({"eval": results})

    sampler = BatchSampler(len(train_examples), tc.batch_size, cfg.seed)
    model.train()
    t0 = time.time()
    while step < tc.max_steps:
        lr = get_lr(step, tc)
        for g in opt.param_groups:
            g["lr"] = lr

        batch = collate([train_examples[i] for i in sampler.indices(step)], tok.pad).to(device)
        with amp:
            loss, parts = compute_loss(model, batch, cfg.objective)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1

        if step % tc.log_every == 0:
            detail = " ".join(f"{k} {v:.4f}" for k, v in parts.items())
            print(f"step {step}/{tc.max_steps} loss {loss.item():.4f} ({detail}) lr {lr:.2e} {time.time() - t0:.0f}s")
            log({"loss": loss.item(), **parts, "lr": lr})
        if step % tc.eval_every == 0 and step < tc.max_steps:
            run_eval()
        if step % tc.ckpt_every == 0 or step == tc.max_steps:
            save_checkpoint(ckpt_path, model, opt, step, cfg, tok.vocab_size, meta["n_people"])

    run_eval()


if __name__ == "__main__":
    main()
