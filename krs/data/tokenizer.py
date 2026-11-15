"""Byte-level BPE trained on our own corpus.

    python -m krs.data.tokenizer --data data/generated/main --vocab-size 8192
"""

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer as HFTokenizer
from tokenizers import decoders, models, pre_tokenizers, trainers

PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"
TOKENIZER_FILE = "tokenizer.json"


class Tokenizer:
    def __init__(self, path):
        self.tok = HFTokenizer.from_file(str(path))
        self.pad = self.tok.token_to_id(PAD)
        self.bos = self.tok.token_to_id(BOS)
        self.eos = self.tok.token_to_id(EOS)
        self.vocab_size = self.tok.get_vocab_size()

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self.tok.decode(ids)


def train_tokenizer(texts, vocab_size, out_path):
    tok = HFTokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[PAD, BOS, EOS],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer)
    tok.save(str(out_path))


def _train_texts(data_dir):
    with open(Path(data_dir) / "train.jsonl") as f:
        for line in f:
            yield "".join(text for text, _ in json.loads(line)["segments"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--vocab-size", type=int, default=8192)
    args = ap.parse_args()

    out = Path(args.data) / TOKENIZER_FILE
    train_tokenizer(_train_texts(args.data), args.vocab_size, out)
    print(f"wrote {out} (vocab {Tokenizer(out).vocab_size})")


if __name__ == "__main__":
    main()
