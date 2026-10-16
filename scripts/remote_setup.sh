#!/usr/bin/env bash
# Set up a rented Linux GPU machine (e.g. a RunPod pod) for training (design doc §30).
#
#     bash scripts/remote_setup.sh
#
# Run from the repo root on the persistent volume. Safe to rerun, and needed again on every new
# pod: it reinstalls uv (which lives on the pod's own disk) and reuses everything on the volume
# (the virtualenv, uv's cache, the Google Cloud CLI and its login).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/remote_env.sh

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv sync --all-extras
uv run python - <<'EOF'
import torch

if not torch.cuda.is_available():
    raise SystemExit(
        f"CUDA is not available to torch {torch.__version__} (built for CUDA {torch.version.cuda}):"
        " this machine's NVIDIA driver is too old. Pick a machine whose driver supports that CUDA."
    )
print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.get_device_name()}")
EOF

if [ ! -x "$REMOTE_VOLUME/google-cloud-sdk/bin/gcloud" ]; then
    curl -sSL https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz \
        | tar -xz -C "$REMOTE_VOLUME"
fi
gcloud --version | sed -n 1p

account="$(gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null || true)"
if [ -n "$account" ]; then
    echo "gcloud signed in as: $account"
else
    echo
    echo "Next: sign in to Google Cloud with the account registered for WOMD:"
    echo "    gcloud auth login --no-launch-browser"
fi
