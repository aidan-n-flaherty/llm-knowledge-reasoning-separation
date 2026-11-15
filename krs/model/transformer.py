import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from krs.model.memory import build_memory

ROPE_FREQ = {
    "factor": 32.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_context_length": 8192,
}


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return out.type_as(x) * self.weight


def precompute_rope(head_dim, theta_base, context_length, freq_config):
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))

    low_wavelen = freq_config["original_context_length"] / freq_config["low_freq_factor"]
    high_wavelen = freq_config["original_context_length"] / freq_config["high_freq_factor"]
    wavelen = 2 * math.pi / inv_freq

    scaled = torch.where(wavelen > low_wavelen, inv_freq / freq_config["factor"], inv_freq)
    smooth = (freq_config["original_context_length"] / wavelen - freq_config["low_freq_factor"]) / (
        freq_config["high_freq_factor"] - freq_config["low_freq_factor"]
    )
    smoothed = (1 - smooth) * (inv_freq / freq_config["factor"]) + smooth * inv_freq
    is_medium = (wavelen <= low_wavelen) & (wavelen >= high_wavelen)
    inv_freq = torch.where(is_medium, smoothed, scaled)

    angles = torch.outer(torch.arange(context_length, dtype=torch.float32), inv_freq)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    xf = x.float().reshape(*x.shape[:-1], -1, 2)
    x0, x1 = xf[..., 0], xf[..., 1]
    c, s = cos[None, :, None, :], sin[None, :, None, :]
    out = torch.stack((x0 * c - x1 * s, x0 * s + x1 * c), dim=-1).flatten(-2)
    return out.type_as(x)


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.emb_dim, cfg.hidden_dim, bias=False)
        self.up_proj = nn.Linear(cfg.emb_dim, cfg.hidden_dim, bias=False)
        self.down_proj = nn.Linear(cfg.hidden_dim, cfg.emb_dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_groups = cfg.n_kv_groups
        self.head_dim = cfg.emb_dim // cfg.n_heads
        self.q_proj = nn.Linear(cfg.emb_dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.emb_dim, self.n_kv_groups * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.emb_dim, self.n_kv_groups * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.emb_dim, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_groups, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_groups, self.head_dim)

        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        rep = self.n_heads // self.n_kv_groups
        k = k.repeat_interleave(rep, dim=2)
        v = v.repeat_interleave(rep, dim=2)

        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, -1))


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.emb_dim, cfg.norm_eps)
        self.self_attn = GroupedQueryAttention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.emb_dim, cfg.norm_eps)
        self.mlp = FeedForward(cfg)

    def forward(self, x, cos, sin, read_memory=None):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        h = self.post_attention_layernorm(x)
        out = x + self.mlp(h)
        if read_memory is not None:
            out = out + read_memory(h)
        return out


class Llama3Model(nn.Module):
    def __init__(self, cfg, vocab_size, n_entities):
        super().__init__()
        m = cfg.model
        self.embed_tokens = nn.Embedding(vocab_size, m.emb_dim)
        self.layers = nn.ModuleList(TransformerBlock(m) for _ in range(m.n_layers))
        self.norm = RMSNorm(m.emb_dim, m.norm_eps)
        self.lm_head = nn.Linear(m.emb_dim, vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

        cos, sin = precompute_rope(m.emb_dim // m.n_heads, m.rope_base, m.context_length, ROPE_FREQ)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.apply(self._init_weights)
        # built after the base init so the memory keeps its own initialisation
        self.memory = build_memory(cfg, n_entities)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, tokens, ctx=None):
        """ctx: MemoryContext, or None to run the base model with no memory read."""
        T = tokens.shape[1]
        x = self.embed_tokens(tokens)
        cos, sin = self.cos[:T], self.sin[:T]
        for i, layer in enumerate(self.layers):
            read = None
            if ctx is not None and i in self.memory.layers:
                read = lambda h, i=i: self.memory(i, h, ctx)
            x = layer(x, cos, sin, read)
        return self.lm_head(self.norm(x))
