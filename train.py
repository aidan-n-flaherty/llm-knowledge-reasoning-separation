import math
import random
import re
import shutil
import string
from pathlib import Path

import tiktoken
import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tiktoken.load import load_tiktoken_bpe

from model import Llama3Model

TOKENIZER_PATH = "tokenizer.model"
MAX_LEN = 256
BATCH_SIZE = 8
GRAD_ACCUM = 4
EPOCHS = 100
LR = 3e-4
MIN_LR = 3e-5
WARMUP_STEPS = 200
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
LOG_EVERY = 10
EVAL_EXAMPLES = 100
EVAL_PRINT=10
EVAL_MAX_NEW_TOKENS = 32
SAVE_PATH = "llama3_1b_triviaqa.pt"
SEED = 0

SPECIAL_TOKENS = [
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|reserved_special_token_0|>",
    "<|reserved_special_token_1|>",
    "<|finetune_right_pad_id|>",
    "<|step_id|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eom_id|>",
    "<|eot_id|>",
    "<|python_tag|>",
]
SPECIAL_TOKENS += [f"<|reserved_special_token_{2 + i}|>" for i in range(256 - len(SPECIAL_TOKENS))]


def build_tokenizer(path):
    ranks = load_tiktoken_bpe(path)
    return tiktoken.Encoding(
        name="llama3",
        pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        mergeable_ranks=ranks,
        special_tokens={t: len(ranks) + i for i, t in enumerate(SPECIAL_TOKENS)},
    )


class ChatFormat:
    def __init__(self, enc):
        self.enc = enc
        self.bos = enc.encode_single_token("<|begin_of_text|>")
        self.start_header = enc.encode_single_token("<|start_header_id|>")
        self.end_header = enc.encode_single_token("<|end_header_id|>")
        self.eot = enc.encode_single_token("<|eot_id|>")
        self.pad = enc.encode_single_token("<|finetune_right_pad_id|>")

    def header(self, role):
        return [self.start_header] + self.enc.encode_ordinary(role) + [self.end_header] + self.enc.encode_ordinary("\n\n")

    def encode_prompt(self, query):
        return [self.bos] + self.header("user") + self.enc.encode_ordinary(query) + [self.eot] + self.header("assistant")

    def encode_example(self, query, answer):
        prompt = self.encode_prompt(query)
        target = self.enc.encode_ordinary(answer) + [self.eot]
        ids = (prompt + target)[: MAX_LEN + 1]
        labels = ([-100] * len(prompt) + target)[: MAX_LEN + 1]
        return ids, labels


def collate(batch, pad_id):
    n = max(len(ids) for ids, _ in batch)
    x = torch.full((len(batch), n), pad_id, dtype=torch.long)
    y = torch.full((len(batch), n), -100, dtype=torch.long)
    for i, (ids, labels) in enumerate(batch):
        x[i, : len(ids)] = torch.tensor(ids)
        y[i, : len(labels)] = torch.tensor(labels)
    return x[:, :-1], y[:, 1:]


def get_lr(step, total_steps):
    if step < WARMUP_STEPS:
        return LR * (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
    return MIN_LR + 0.5 * (1 + math.cos(math.pi * progress)) * (LR - MIN_LR)


def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


@torch.no_grad()
def generate(model, chat, query, device, amp):
    ids = torch.tensor([chat.encode_prompt(query)], device=device)
    out = []
    for _ in range(EVAL_MAX_NEW_TOKENS):
        with amp:
            logits = model(ids)
        next_id = logits[0, -1].argmax().item()
        if next_id == chat.eot:
            break
        out.append(next_id)
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)
    return chat.enc.decode(out)


@torch.no_grad()
def evaluate(model, chat, val_examples, device, amp):
    model.eval()
    correct = 0
    for i, ex in enumerate(val_examples):
        pred = normalize_answer(generate(model, chat, ex["query"], device, amp))
        hit = pred in ex["answer"]
        correct += hit
        if i < EVAL_PRINT:
            print(f"  Q: {ex['query']}\n  pred: {pred!r} | target: {ex['answer']!r} | {'ok' if hit else 'miss'}")
    model.train()
    return correct / len(val_examples)


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not Path(TOKENIZER_PATH).exists():
        downloaded = hf_hub_download("meta-llama/Llama-3.2-1B", "original/tokenizer.model")
        shutil.copy(downloaded, TOKENIZER_PATH)

    chat = ChatFormat(build_tokenizer(TOKENIZER_PATH))

    train_ds = load_dataset("sentence-transformers/natural-questions", split="train")
    #val_ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    examples = [chat.encode_example(ex["query"], ex["answer"]) for ex in train_ds]
    val_examples = [train_ds[i] for i in range(EVAL_EXAMPLES)]

    model = Llama3Model().to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": WEIGHT_DECAY}, {"params": no_decay, "weight_decay": 0.0}],
        lr=LR,
        betas=(0.9, 0.95),
    )

    steps_per_epoch = len(examples) // (BATCH_SIZE * GRAD_ACCUM)
    total_steps = steps_per_epoch * EPOCHS
    amp = torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda")

    print(f"exact match before training: {evaluate(model, chat, val_examples, device, amp):.3f}")

    model.train()
    step = 0
    for epoch in range(EPOCHS):
        random.shuffle(examples)
        for s in range(steps_per_epoch):
            for g in opt.param_groups:
                g["lr"] = get_lr(step, total_steps)

            loss_sum = 0.0
            for a in range(GRAD_ACCUM):
                start = (s * GRAD_ACCUM + a) * BATCH_SIZE
                x, y = collate(examples[start : start + BATCH_SIZE], chat.pad)
                x, y = x.to(device), y.to(device)
                with amp:
                    logits = model(x)
                loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)
                (loss / GRAD_ACCUM).backward()
                loss_sum += loss.item() / GRAD_ACCUM

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            if step % LOG_EVERY == 0:
                print(f"epoch {epoch} step {step}/{total_steps} loss {loss_sum:.4f} lr {get_lr(step, total_steps):.2e}")

        print(f"epoch {epoch} exact match: {evaluate(model, chat, val_examples, device, amp):.3f}")

    torch.save(model.state_dict(), SAVE_PATH)


if __name__ == "__main__":
    main()