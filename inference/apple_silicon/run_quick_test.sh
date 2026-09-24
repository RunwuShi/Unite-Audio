#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

# A short MPS smoke test using output_0 from the TangoFlux/AudioCaps list.
"$ROOT/.venv/bin/python" "$ROOT/generate_mps.py" \
  --tangoflux-index 0 \
  --model s3_rl14 \
  --seconds 2 \
  --prior-steps 4 \
  --wave-steps 1
