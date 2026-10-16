# shellcheck shell=bash
# Environment for a rented Linux GPU machine (e.g. a RunPod pod; design doc §30).
# Source it in every new shell, from the repo root on the persistent volume:
#
#     source scripts/remote_env.sh
#
# Keeps uv's package cache and the Google Cloud CLI (with its login) on the volume, next to the
# repo, so a new pod attached to the same volume needs no reinstall and no new gcloud login.

REMOTE_VOLUME="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export REMOTE_VOLUME
export UV_CACHE_DIR="$REMOTE_VOLUME/.uv-cache"
export CLOUDSDK_CONFIG="$REMOTE_VOLUME/.gcloud"
export PATH="$HOME/.local/bin:$REMOTE_VOLUME/google-cloud-sdk/bin:$PATH"
