#!/bin/bash
# End-to-end check on CPU in about a minute: generate -> tokenizer -> train -> resume -> evaluate.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}

rm -rf data/generated/smoke runs/smoke
$PY -m krs.data.generate --out data/generated/smoke --n-people 200 --bio-variants 2
$PY -m krs.data.tokenizer --data data/generated/smoke --vocab-size 1000
$PY -m krs.train --config configs/smoke.yaml --set train.max_steps=30
$PY -m krs.train --config configs/smoke.yaml   # resumes at step 30
$PY -m krs.evaluate --ckpt runs/smoke/last.pt
