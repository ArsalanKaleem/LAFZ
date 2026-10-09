#!/usr/bin/env bash
# Fast pipeline shakedown on a real GPU (minutes). Expect to fix a few things here — that's the point.
set -euxo pipefail
L=${1:-ur}
CFG="--config configs/debug.yaml --lang $L"
harf env $CFG
harf smoke $CFG
harf prepare $CFG
harf audit $CFG || true
harf select $CFG
harf run $CFG --condition baseline --steps eval-ppl
for C in A B C; do harf run $CFG --condition $C --steps extend,inject,eval-ppl; done
harf run $CFG --condition D --steps extend,inject,train-lora,eval-ppl
harf eval-compression $CFG
harf report $CFG
