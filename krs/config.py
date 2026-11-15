"""Run configuration: dataclass defaults, overridden by a YAML file, then by --set key=value."""

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import List

import yaml


@dataclass
class ModelConfig:
    context_length: int = 512
    emb_dim: int = 512
    n_heads: int = 8
    n_kv_groups: int = 2
    n_layers: int = 8
    hidden_dim: int = 1408
    rope_base: float = 500_000.0
    norm_eps: float = 1e-5


@dataclass
class MemoryConfig:
    kind: str = "none"  # key into krs.model.memory.MEMORIES
    layers: List[int] = field(default_factory=lambda: [3, 4])
    slots: int = 4
    key_dim: int = 64
    value_dim: int = 64


@dataclass
class ObjectiveConfig:
    masked_pass: bool = False  # second forward pass with memory masked
    masked_fact_loss: str = "none"  # key into krs.objectives.MASKED_FACT_LOSSES
    lambda_nonfact: float = 1.0
    lambda_fact: float = 1.0


@dataclass
class DataConfig:
    dir: str = "data/generated/main"
    max_len: int = 256
    exclude_forget: bool = False  # oracle baseline: drop the forget set from training


@dataclass
class TrainConfig:
    max_steps: int = 20_000
    batch_size: int = 64
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    log_every: int = 50
    eval_every: int = 1000
    ckpt_every: int = 500
    eval_max_examples: int = 1000  # per suite, for in-training evals


@dataclass
class Config:
    run_name: str = "run"
    out_dir: str = "runs"
    seed: int = 0
    model: ModelConfig = field(default_factory=ModelConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self):
        return asdict(self)


def _update(obj, values, path=""):
    names = {f.name for f in fields(obj)}
    for key, value in values.items():
        if key not in names:
            raise KeyError(f"unknown config key: {path}{key}")
        current = getattr(obj, key)
        if is_dataclass(current):
            _update(current, value, f"{path}{key}.")
        elif isinstance(current, float):
            setattr(obj, key, float(value))  # YAML reads "1e-4" as a string
        elif type(value) is not type(current):
            raise TypeError(f"{path}{key}: expected {type(current).__name__}, got {value!r}")
        else:
            setattr(obj, key, value)


def load_config(path=None, overrides=()):
    cfg = Config()
    if path:
        with open(path) as f:
            _update(cfg, yaml.safe_load(f) or {})
    for item in overrides:
        key, _, raw = item.partition("=")
        nested = yaml.safe_load(raw)
        for part in reversed(key.split(".")):
            nested = {part: nested}
        _update(cfg, nested)
    return cfg


def config_from_dict(values):
    cfg = Config()
    _update(cfg, values)
    return cfg
