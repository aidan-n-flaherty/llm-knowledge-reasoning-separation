"""Shared evaluation harness. Every number in the final comparison table comes from here.

    python -m krs.evaluate --ckpt runs/vanilla-s0/last.pt

Each suite in eval.jsonl is scored with memory on and with memory masked:
    em         all answer tokens (and the end-of-text token) are the greedy prediction
    ce_bits    cross-entropy of the answer, in bits
    info_bits  closed-book suites: attribute prior entropy minus ce_bits. With memory masked this
               is the leakage estimate: bits about the fact that the base weights alone supply.
    ppl        stories: per-token perplexity

TODO(eval owner): relearning attack, linear probes, paraphrase robustness, editing metrics, seeds/CIs.
"""

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from krs.data.dataset import collate, load_examples
from krs.model.memory import MemoryContext
from krs.utils import autocast_for, load_data_meta, load_model, load_tokenizer, pick_device


def load_eval_suites(data_dir, tok, max_len, max_examples=None, seed=0):
    suites = defaultdict(list)
    for ex in load_examples(Path(data_dir) / "eval.jsonl", tok, max_len):
        # a truncated QA example has lost its answer and would score as trivially correct
        if ex["truncated"] and ex["meta"]["kind"] != "story":
            continue
        suites[ex["meta"]["suite"]].append(ex)
    if max_examples:
        rng = random.Random(seed)
        suites = {k: rng.sample(v, max_examples) if len(v) > max_examples else v for k, v in suites.items()}
    return dict(suites)


@torch.no_grad()
def score_suite(model, examples, pad_id, device, amp, memory_on, prior_bits, batch_size=128):
    ce_sum = n_tok = n_correct = 0
    info = []
    for start in range(0, len(examples), batch_size):
        b = collate(examples[start : start + batch_size], pad_id).to(device)
        ctx = MemoryContext(b.entity_ids) if memory_on else MemoryContext.off(b.entity_ids)
        with amp:
            logits = model(b.tokens, ctx)
        valid = b.labels != -100
        ce = F.cross_entropy(logits.float().transpose(1, 2), b.labels, ignore_index=-100, reduction="none").sum(1)
        correct = ((logits.argmax(-1) == b.labels) | ~valid).all(1)
        ce_sum += ce.sum().item()
        n_tok += valid.sum().item()
        n_correct += correct.sum().item()
        for meta, ce_i in zip(b.meta, ce.tolist()):
            if meta["kind"] == "qa_closed":
                info.append(prior_bits[meta["attr"]] - ce_i / math.log(2))

    n = len(examples)
    if examples[0]["meta"]["kind"] == "story":
        return {"n": n, "ppl": math.exp(ce_sum / max(n_tok, 1))}
    out = {"n": n, "em": n_correct / n, "ce_bits": ce_sum / n / math.log(2)}
    if info:
        out["info_bits"] = sum(info) / len(info)
    return out


def evaluate(model, suites, pad_id, device, amp, prior_bits):
    """Returns {suite: {"on": metrics, "off": metrics}}; "off" is skipped for a model with no memory."""
    was_training = model.training
    model.eval()
    modes = {"on": True, "off": False} if model.memory.layers else {"on": True}
    results = {
        name: {mode: score_suite(model, exs, pad_id, device, amp, on, prior_bits) for mode, on in modes.items()}
        for name, exs in sorted(suites.items())
    }
    model.train(was_training)
    return results


def format_results(results):
    lines = []
    for suite, modes in results.items():
        for mode, m in modes.items():
            stats = "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in m.items())
            lines.append(f"  {suite:<16} mem={mode:<3} {stats}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max-examples", type=int, default=None, help="cap per suite (default: all)")
    ap.add_argument("--out", default=None, help="metrics JSON path (default: next to the checkpoint)")
    args = ap.parse_args()

    device = pick_device()
    model, cfg, ckpt = load_model(args.ckpt, device)
    tok = load_tokenizer(cfg.data.dir)
    meta, _ = load_data_meta(cfg.data.dir)
    suites = load_eval_suites(cfg.data.dir, tok, cfg.data.max_len, args.max_examples, cfg.seed)

    results = evaluate(model, suites, tok.pad, device, autocast_for(device), meta["attr_prior_bits"])
    print(f"step {ckpt['step']}\n{format_results(results)}")

    out = Path(args.out) if args.out else Path(args.ckpt).with_name("metrics.json")
    out.write_text(json.dumps({"step": ckpt["step"], "data_sha256": meta["sha256"], "results": results}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
