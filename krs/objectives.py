"""Losses.

MASKED_FACT_LOSSES holds the loss applied to fact tokens in the memory-masked pass. Each entry is
    fn(logits_off, batch) -> scalar
where logits_off are the logits with memory masked and batch is a krs.data.dataset.Batch
(batch.labels, batch.fact_mask, batch.meta[i]["attr"] are what a variant usually needs).

To add a variant: write the function and decorate it with @register("name"), then select it with
objective.masked_fact_loss in a config.
"""

import torch.nn.functional as F

MASKED_FACT_LOSSES = {}


def register(name):
    def wrap(fn):
        MASKED_FACT_LOSSES[name] = fn
        return fn

    return wrap


def lm_loss(logits, labels):
    """Mean next-token cross-entropy over positions where labels != -100 (0 if there are none)."""
    total = F.cross_entropy(
        logits.float().view(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100, reduction="sum"
    )
    return total / (labels != -100).sum().clamp(min=1)


@register("none")
def no_fact_loss(logits_off, batch):
    return logits_off.new_zeros(())
