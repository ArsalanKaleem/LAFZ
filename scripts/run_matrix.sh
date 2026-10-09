#!/usr/bin/env bash
# Run the full experiment matrix for one language.
# Usage: scripts/run_matrix.sh ur [configs/qwen2.5-1.5b.yaml] [conditions...]
set -euo pipefail
LANG_CODE=${1:?usage: run_matrix.sh <ur|sd|ur+sd> [config] [conditions...]}
CONFIG=${2:-configs/qwen2.5-1.5b.yaml}
shift $(( $# >= 2 ? 2 : 1 ))
CONDS=${*:-"baseline A B C D D_mean E F baseline_lora"}

harf prepare --config "$CONFIG" --lang "$LANG_CODE"
harf audit   --config "$CONFIG" --lang "$LANG_CODE" || true   # frozen after first run
harf select  --config "$CONFIG" --lang "$LANG_CODE"
for C in $CONDS; do harf extend --config "$CONFIG" --lang "$LANG_CODE" --condition "$C"; done
harf eval-compression --config "$CONFIG" --lang "$LANG_CODE"   # CHECKPOINT: verify compression before GPU time
for C in $CONDS; do harf run --config "$CONFIG" --lang "$LANG_CODE" --condition "$C"; done
harf report --config "$CONFIG" --lang "$LANG_CODE"
