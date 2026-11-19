import math
import random
import re
import shutil
import string
from collections import Counter
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
BATCH_SIZE = 16
GRAD_ACCUM = 4
EPOCHS = 100
LR = 3e-4
MIN_LR = 3e-5
WARMUP_STEPS = 200
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
LOG_EVERY = 10
EVAL_EXAMPLES = 10
EVAL_PRINT=10
EVAL_MAX_NEW_TOKENS = 128
COQA_STORY_CHARS = 500
SAVE_PATH = "llm_weights.pt"
SEED = 0

REFUSALS = [
    "I don't know.",
    "I don't have that information.",
    "I can't answer that without more information.",
]

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

def gpu_stats():
    if not torch.cuda.is_available():
        return ""
    total = torch.cuda.get_device_properties(0).total_memory
    used, peak = torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated()
    try:
        util = f" | util {torch.cuda.utilization()}%"  # needs pynvml: pip install nvidia-ml-py
    except Exception:
        util = ""
    return f" | mem {used / 2**30:.1f}G (peak {peak / 2**30:.1f}G) / {total / 2**30:.0f}G{util}"

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


def f1_score(pred, gold):
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    n = sum((Counter(p) & Counter(g)).values())
    if n == 0:
        return 0.0
    precision, recall = n / len(p), n / len(g)
    return 2 * precision * recall / (precision + recall)


REFUSALS_NORM = {normalize_answer(r) for r in REFUSALS}


def coqa_examples(ds):
    out = []
    for ex in ds:
        story = ex["story"][:COQA_STORY_CHARS]
        for q, a in zip(ex["questions"], ex["answers"]["input_text"]):
            out.append({"query": f"{story}\n\nQuestion: {q}", "answer": a})
    return out


@torch.no_grad()
def generate(model, chat, query, device, amp, use_memory=True):
    ids = torch.tensor([chat.encode_prompt(query)], device=device)
    out = []
    for _ in range(EVAL_MAX_NEW_TOKENS):
        with amp:
            logits = model(ids, use_memory=use_memory)
        next_id = logits[0, -1].argmax().item()
        if next_id == chat.eot:
            break
        out.append(next_id)
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)
    return chat.enc.decode(out)


@torch.no_grad()
def evaluate(model, chat, val_examples, device, amp, use_memory=True, refusal=False):
    model.eval()
    scores, preds = [], []
    for i, ex in enumerate(val_examples):
        pred = generate(model, chat, ex["query"], device, amp, use_memory)
        preds.append(normalize_answer(pred))
        score = float(preds[-1] in REFUSALS_NORM) if refusal else f1_score(pred, ex["answer"])
        scores.append(score)
        if i < EVAL_PRINT:
            target = "<refusal>" if refusal else ex["answer"]
            print(f"  Q: {ex['query'][-120:]!r}\n  pred: {preds[-1]!r} | target: {target!r} | score {score:.2f}")
    model.train()
    return sum(scores) / len(scores), preds


def report(prefix, model, chat, nq_val, coqa_val, device, amp):
    nq_f1, _ = evaluate(model, chat, nq_val, device, amp, True)
    nq_ref, _ = evaluate(model, chat, nq_val, device, amp, False, refusal=True)
    co_mem, p_mem = evaluate(model, chat, coqa_val, device, amp, True)
    co_no, p_no = evaluate(model, chat, coqa_val, device, amp, False)
    agree = sum(a == b for a, b in zip(p_mem, p_no)) / len(p_mem)
    print(
        f"{prefix} NQ F1 (memory): {nq_f1:.3f} | NQ refusal (no memory): {nq_ref:.3f} | "
        f"CoQA F1 memory/no memory: {co_mem:.3f}/{co_no:.3f} | CoQA agreement: {agree:.3f}"
    )


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
    examples = [
        (chat.encode_example(ex["query"], ex["answer"]), chat.encode_example(ex["query"], random.choice(REFUSALS)))
        for ex in train_ds
    ]
    val_examples = [train_ds[i] for i in range(EVAL_EXAMPLES)]

    coqa_train = coqa_examples(load_dataset("stanfordnlp/coqa", split="train"))
    logic_examples = [chat.encode_example(ex["query"], ex["answer"]) for ex in coqa_train]
    coqa_val = coqa_examples(load_dataset("stanfordnlp/coqa", split="validation"))[:EVAL_EXAMPLES]

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

    report("before training:", model, chat, val_examples, coqa_val, device, amp)

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
                batch = examples[start : start + BATCH_SIZE]
                passes = [
                    ([b[0] for b in batch], True),
                    ([b[1] for b in batch], False),
                    (random.sample(logic_examples, BATCH_SIZE), random.random() < 0.5),
                ]
                scale = GRAD_ACCUM * len(passes)
                for pass_batch, use_memory in passes:
                    x, y = collate(pass_batch, chat.pad)
                    x, y = x.to(device), y.to(device)
                    with amp:
                        logits = model(x, use_memory=use_memory)
                    loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)
                    (loss / scale).backward()
                    loss_sum += loss.item() / scale

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            if step % LOG_EVERY == 0:
                print(f"epoch {epoch} step {step}/{total_steps} loss {loss_sum:.4f} lr {get_lr(step, total_steps):.2e}{gpu_stats()}")

        report(f"epoch {epoch}", model, chat, val_examples, coqa_val, device, amp)

    torch.save(model.state_dict(), SAVE_PATH)


if __name__ == "__main__":
    main()