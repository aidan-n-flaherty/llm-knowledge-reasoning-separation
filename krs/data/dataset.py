"""Example and batch format shared by training and evaluation.

An example on disk (one JSON object per line) is
    {"kind": ..., "entity_id": int, "attr": str|null, "suite": str (eval only),
     "segments": [[text, tag], ...]}
with tag one of
    "prompt"  context, no loss
    "text"    next-token loss
    "fact"    next-token loss, and counted as a fact token (the memory-masked objective acts here)
"""

import json
from dataclasses import dataclass
from typing import List

import torch

PROMPT, TEXT, FACT = "prompt", "text", "fact"


@dataclass
class Batch:
    tokens: torch.Tensor  # [B, T] long
    labels: torch.Tensor  # [B, T] long, -100 where no loss
    fact_mask: torch.Tensor  # [B, T] bool, True where the label is a fact token
    entity_ids: torch.Tensor  # [B] long, -1 = no entity
    meta: List[dict]  # per example: kind, attr, suite

    def to(self, device):
        return Batch(
            self.tokens.to(device),
            self.labels.to(device),
            self.fact_mask.to(device),
            self.entity_ids.to(device),
            self.meta,
        )


def encode_example(ex, tok, max_len):
    ids, loss, fact = [tok.bos], [False], [False]
    for text, tag in ex["segments"]:
        piece = tok.encode(text)
        ids += piece
        loss += [tag != PROMPT] * len(piece)
        fact += [tag == FACT] * len(piece)
    ids.append(tok.eos)
    loss.append(True)
    fact.append(False)
    n = max_len + 1
    return {
        "ids": ids[:n],
        "loss": loss[:n],
        "fact": fact[:n],
        "truncated": len(ids) > n,
        "entity_id": ex["entity_id"],
        "meta": {"kind": ex["kind"], "attr": ex.get("attr"), "suite": ex.get("suite")},
    }


def load_examples(path, tok, max_len, keep=None):
    """Read and tokenize a jsonl file. keep(example_dict) -> bool filters raw examples."""
    out, cut = [], 0
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            if keep is not None and not keep(ex):
                continue
            enc = encode_example(ex, tok, max_len)
            cut += enc["truncated"] and ex["kind"] != "story"
            out.append(enc)
    if cut:
        print(f"WARNING: {cut} non-story examples in {path} were truncated to {max_len} tokens")
    return out


def collate(examples, pad_id):
    T = max(len(e["ids"]) for e in examples) - 1
    B = len(examples)
    tokens = torch.full((B, T), pad_id, dtype=torch.long)
    labels = torch.full((B, T), -100, dtype=torch.long)
    fact_mask = torch.zeros((B, T), dtype=torch.bool)
    for i, e in enumerate(examples):
        ids = torch.tensor(e["ids"])
        n = len(ids) - 1
        tokens[i, :n] = ids[:-1]
        loss = torch.tensor(e["loss"][1:])
        labels[i, :n] = torch.where(loss, ids[1:], torch.full_like(ids[1:], -100))
        fact_mask[i, :n] = torch.tensor(e["fact"][1:])
    entity_ids = torch.tensor([e["entity_id"] for e in examples], dtype=torch.long)
    return Batch(tokens, labels, fact_mask, entity_ids, [e["meta"] for e in examples])
