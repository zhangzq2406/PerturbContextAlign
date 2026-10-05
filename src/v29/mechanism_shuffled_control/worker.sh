#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo "usage: worker.sh RUN_ROOT DEEP_OVERLAY PYTHON" >&2
  exit 2
fi
RUN_ROOT="$1"
DEEP="$2"
PY="$3"
SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/mechanism_shuffled_control_v1.py"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
"$PY" "$SCRIPT" preflight --deep-root "$DEEP" --run-root "$RUN_ROOT"
"$PY" "$SCRIPT" encode --deep-root "$DEEP" --run-root "$RUN_ROOT"
"$PY" "$SCRIPT" geometry --deep-root "$DEEP" --run-root "$RUN_ROOT"
"$PY" "$SCRIPT" predict --deep-root "$DEEP" --run-root "$RUN_ROOT"
"$PY" "$SCRIPT" score --deep-root "$DEEP" --run-root "$RUN_ROOT"
"$PY" "$SCRIPT" summarize --deep-root "$DEEP" --run-root "$RUN_ROOT"
cp "$RUN_ROOT/summarize/COMPACT.txt" "$RUN_ROOT/COMPACT.txt"
