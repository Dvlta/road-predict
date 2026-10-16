#!/usr/bin/env bash
# Train several model configs one after another, resuming any run that was interrupted.
#
#     bash scripts/train_all.sh configs/interaction_history.yaml configs/interaction_neighbors.yaml
#
# Each config trains as outputs/runs/<config>_<tier>. Rerun the same command after a disconnect
# or on a new pod: runs with a last.pt resume from it (a finished run trains nothing and is
# skipped quickly), the others start fresh. Settings: DATA_CONFIG (default configs/data_exp.yaml)
# and WORKERS (loader processes, default 8; results do not depend on it).
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_CONFIG="${DATA_CONFIG:-configs/data_exp.yaml}"
WORKERS="${WORKERS:-8}"
tier="$(basename "$DATA_CONFIG" .yaml)"
tier="${tier#data_}"

for config in "$@"; do
    run="outputs/runs/$(basename "$config" .yaml)_$tier"
    if [ -f "$run/last.pt" ]; then
        uv run python scripts/train.py --resume "$run" --num-workers "$WORKERS"
    else
        uv run python scripts/train.py --config "$config" --data-config "$DATA_CONFIG" \
            --name "$(basename "$run")" --num-workers "$WORKERS"
    fi
done
