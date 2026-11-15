"""Memory modules read in parallel with the FFN.

Contract (do not change without a PR to main):
    memory(layer, h, ctx) -> delta
where h is the post-norm residual [B, T, D], delta has the same shape, and delta must be
exactly zero for rows where ctx.masked is True or ctx.entity_ids is -1.

To add a variant: subclass Memory, set self.layers, and register it in MEMORIES.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class MemoryContext:
    entity_ids: torch.Tensor  # [B] long, -1 = example has no entity
    masked: Optional[torch.Tensor] = None  # [B] bool, True = memory off for that row

    def active(self):
        on = self.entity_ids >= 0
        return on if self.masked is None else on & ~self.masked

    @staticmethod
    def off(entity_ids):
        return MemoryContext(entity_ids, torch.ones_like(entity_ids, dtype=torch.bool))


class Memory(nn.Module):
    layers = ()  # block indices this memory is read in

    def forward(self, layer, h, ctx):
        raise NotImplementedError


class NoMemory(Memory):
    def __init__(self, cfg, n_entities):
        super().__init__()


class EntityBlockMemory(Memory):
    """M0, oracle routing: every entity owns `slots` key/value pairs per memory layer, fetched by ID."""

    def __init__(self, cfg, n_entities):
        super().__init__()
        m, d = cfg.memory, cfg.model.emb_dim
        self.layers = tuple(m.layers)
        self.key_dim = m.key_dim
        names = [str(layer) for layer in self.layers]
        self.keys = nn.ParameterDict(
            {n: nn.Parameter(torch.randn(n_entities, m.slots, m.key_dim) * 0.02) for n in names}
        )
        self.values = nn.ParameterDict(
            {n: nn.Parameter(torch.randn(n_entities, m.slots, m.value_dim) * 0.02) for n in names}
        )
        self.q_proj = nn.ModuleDict({n: nn.Linear(d, m.key_dim, bias=False) for n in names})
        self.out_proj = nn.ModuleDict({n: nn.Linear(m.value_dim, d, bias=False) for n in names})
        for proj in self.q_proj.values():
            nn.init.normal_(proj.weight, mean=0.0, std=0.02)
        # zero init: memory has no effect at the start of training
        for proj in self.out_proj.values():
            nn.init.zeros_(proj.weight)

    def forward(self, layer, h, ctx):
        n = str(layer)
        ids = ctx.entity_ids.clamp(min=0)
        keys, values = self.keys[n][ids], self.values[n][ids]  # [B, S, dk], [B, S, dv]
        q = self.q_proj[n](h)  # [B, T, dk]
        w = torch.softmax(q @ keys.transpose(1, 2).to(q.dtype) / math.sqrt(self.key_dim), dim=-1)  # [B, T, S]
        out = self.out_proj[n](w @ values.to(q.dtype))
        return out * ctx.active()[:, None, None].to(out.dtype)


MEMORIES = {
    "none": NoMemory,
    "entity_block": EntityBlockMemory,
}


def build_memory(cfg, n_entities):
    return MEMORIES[cfg.memory.kind](cfg, n_entities)
