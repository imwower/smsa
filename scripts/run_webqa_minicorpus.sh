#!/usr/bin/env bash
set -euo pipefail

LOG_PATH="runs/minicorpus_webqa.log"
SUMMARY_PATH="runs/minicorpus_webqa.summary.json"

# Create runs directory if missing
mkdir -p runs

echo "[run] Starting WebQA MiniCorpus training..." | tee "$LOG_PATH"

# Train Self-Model on WebQA with moderate budget
python scripts/train_minicorpus.py \
  --dataset data/webqa/train.jsonl \
  --episodes 12 \
  --train-limit 2200 \
  --val-limit 600 \
  --vocab-size 384 \
  --seed 0 \
  --meta-window 8 \
  --meta-delta 0.01 \
  --log-interval 200 \
  >> "$LOG_PATH" 2>&1

echo "[run] Training finished. Parsing tail metrics..." | tee -a "$LOG_PATH"

python - <<'PY'
import json, re
from pathlib import Path
log = Path("runs/minicorpus_webqa.log").read_text(encoding="utf-8", errors="ignore")
m = re.findall(r"MiniCorpus tail metrics: NLL=([0-9.]+) cause=([0-9.]+) energy=([0-9.]+) spikes=([0-9.]+) positives=([0-9.]+)", log)
out = {}
if m:
    nll, cause, energy, spikes, pos = m[-1]
    out = {
        "tail_nll": float(nll),
        "cause_acc": float(cause),
        "energy_mse": float(energy),
        "spike_norm": float(spikes),
        "positive_ratio": float(pos),
    }
Path("runs/minicorpus_webqa.summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False))
PY

echo "[run] Summary written to $SUMMARY_PATH" | tee -a "$LOG_PATH"

