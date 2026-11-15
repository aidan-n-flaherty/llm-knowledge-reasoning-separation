import contextlib
import json
import os
from pathlib import Path

import torch

from krs.config import config_from_dict
from krs.data.tokenizer import TOKENIZER_FILE, Tokenizer
from krs.model.transformer import Llama3Model


def pick_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def autocast_for(device):
    """bf16 autocast where the GPU supports it (OSCAR nodes differ), otherwise a no-op."""
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def load_data_meta(data_dir):
    data_dir = Path(data_dir)
    meta = json.loads((data_dir / "meta.json").read_text())
    splits = json.loads((data_dir / "splits.json").read_text())
    return meta, splits


def load_tokenizer(data_dir):
    return Tokenizer(Path(data_dir) / TOKENIZER_FILE)


def save_checkpoint(path, model, opt, step, cfg, vocab_size, n_entities):
    state = {
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "step": step,
        "config": cfg.to_dict(),
        "vocab_size": vocab_size,
        "n_entities": n_entities,
    }
    tmp = f"{path}.tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic, so a job killed mid-save never leaves a broken checkpoint


def load_model(ckpt_path, device):
    """Rebuild the model from a checkpoint. Returns (model, cfg, checkpoint dict)."""
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = config_from_dict(ckpt["config"])
    model = Llama3Model(cfg, ckpt["vocab_size"], ckpt["n_entities"]).to(device)
    model.load_state_dict(ckpt["model"])
    return model, cfg, ckpt
