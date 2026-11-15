# llm-knowledge-reasoning-separation

Shared trunk for the CSCI 2951X project. The research plan is in [plan.md](plan.md).

## Quickstart

```bash
pip install -r requirements.txt
./scripts/smoke.sh        # generate -> tokenizer -> train -> resume -> evaluate, about a minute on CPU
pytest -q tests
```

Full pipeline:

```bash
python -m krs.data.generate --out data/generated/main --n-people 20000 --stories 200000
python -m krs.data.tokenizer --data data/generated/main --vocab-size 8192
python -m krs.train --config configs/vanilla.yaml
python -m krs.evaluate --ckpt runs/vanilla/last.pt
```

On OSCAR: `mkdir -p logs && sbatch --array=0-2 scripts/train.sbatch configs/vanilla.yaml` (the array index is the seed).

## Layout

| Path | What | Owner |
|---|---|---|
| `krs/data/generate.py` | Synthetic biographies, QA, splits, fact-span tags | data |
| `krs/data/tokenizer.py` | BPE tokenizer trained on our corpus | data |
| `krs/data/dataset.py` | Example and `Batch` format | data |
| `krs/model/transformer.py` | Llama-3-style backbone | model |
| `krs/model/memory.py` | Memory interface and the fetch-by-ID memory | model |
| `krs/objectives.py` | Loss registry for the memory-masked pass | training |
| `krs/train.py` | Training loop, checkpoint and resume | training |
| `krs/evaluate.py` | Evaluation harness | eval |
| `configs/` | One YAML per experiment | everyone |
| `scripts/train.sbatch` | Slurm launcher | training |

## Shared contracts

Changing any of these changes everyone's numbers, so they go through a PR to `main`.

**Batch** (`krs/data/dataset.py`): `tokens`, `labels` (-100 where no loss), `fact_mask` (true where the label is a fact token), `entity_ids` (-1 for no entity), `meta` (kind, attr, suite).

**Memory** (`krs/model/memory.py`): `memory(layer, h, ctx) -> delta`, added to the block output beside the FFN. `delta` must be exactly zero for rows that are masked or have no entity; `tests/test_memory.py` checks this.

**Masked fact loss** (`krs/objectives.py`): `fn(logits_off, batch) -> scalar`, applied in the memory-masked pass.

**Dataset**: the generator prints a hash and stores it in `meta.json`; `train.py` logs it at startup. Two runs are comparable only if the hashes match.

## Extending on a branch

- **New objective variant:** add a function with `@register("name")` in `krs/objectives.py` and a config that sets `objective.masked_fact_loss: name`.
- **New memory module:** subclass `Memory`, add it to `MEMORIES`, select it with `memory.kind`.
- **New experiment:** add a YAML in `configs/`. Any config field can also be overridden on the command line with `--set a.b=value`.

Variants that need more than a loss function (gradient routing, memory swap) will need a hook in `compute_loss` in `krs/train.py`; propose that as a PR to `main` so the hook is shared.

## Evaluation suites

| Suite | People | Measures |
|---|---|---|
| `closed_train` | QA seen in training | Memorization |
| `closed_heldout` | Biography trained, QA never seen | Knowledge extraction (the headline recall number) |
| `closed_forget` | Reserved forget set | Forgetting experiments; trained on unless `data.exclude_forget` |
| `open_unseen` | Never seen; biography given in the prompt | Reasoning over context |
| `stories` | TinyStories validation | Language modelling |

Every suite is scored with memory on and memory masked. `info_bits` with memory masked is the leakage estimate.

## Not built yet

- Objective variants A-D from the plan (only `none` is registered).
- Learned-retrieval and hierarchical memory.
- Relearning attack, probing, paraphrase and editing evaluations.
- More biography templates and paraphrased eval questions (marked `TODO` in `generate.py`).
- Environment setup in `scripts/train.sbatch` (marked `TODO`).
