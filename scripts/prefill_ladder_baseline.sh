#!/usr/bin/env bash
# Prefill-ladder baseline for issue-1. One model, three context rungs, JSON out.
#
# `mtplx bench prefill-ladder` loads the model IN-PROCESS (prefill_bench.py ->
# mtplx.runtime.load(model, mtp=True)), so it must not overlap with a server holding the
# same weights. Run it before starting scripts/serve-flash-next-*.sh.
#
#   usage: prefill-ladder-baseline.sh <9002|9001> [extra mtplx bench flags...]
#
# `--prefill-layout profile` keeps the selected profile's own default so this run is a
# baseline; the sweep in slice 2 overrides it per run.
set -euo pipefail

MTPLX_HOME_DIR="${MTPLX_HOME:-$HOME/.mtplx}"
MTPLX="${MTPLX_BIN:-$MTPLX_HOME_DIR/bin/mtplx}"
OUT_DIR="${MTPLX_BENCH_DIR:-$MTPLX_HOME_DIR/bench}"
PORT="${1:?usage: prefill-ladder-baseline.sh <9002|9001> [bench flags...]}"
shift || true

case "$PORT" in
  9002)
    MODEL_DIR="${MTPLX_MODEL_DIR_SPEED:-$MTPLX_HOME_DIR/models/Youssofal--Qwen3.8-Flash-Next-MTPLX-Optimized-Speed}"
    PROFILE="${MTPLX_PROFILE:-turbo}"
    DEPTH="${MTPLX_DEPTH:-3}"
    ;;
  9001)
    MODEL_DIR="${MTPLX_MODEL_DIR_UNCENSORED:-$MTPLX_HOME_DIR/models/grant-ai--Qwen3.8-Flash-Next-Abliterated-MTPLX-4bit}"
    PROFILE="${MTPLX_PROFILE:-stable}"
    DEPTH="${MTPLX_DEPTH:-1}"
    ;;
  *) echo "unknown port: $PORT (expected 9002 or 9001)" >&2; exit 2 ;;
esac

export MTPLX_NGRAM_RESIDENT="${MTPLX_NGRAM_RESIDENT:-1}"
mkdir -p "$OUT_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT_FILE="${OUT_DIR}/prefill-ladder-baseline-${PORT}-${STAMP}.json"

exec "$MTPLX" bench prefill-ladder \
  --model "$MODEL_DIR" \
  --profile "$PROFILE" \
  --depth "$DEPTH" \
  --contexts 8k,32k,128k \
  --prompt-style coding-agent \
  --prompt-format chat \
  --prefill-layout profile \
  --temperature 1.0 --top-p 0.95 --top-k 20 \
  --json --output "$OUT_FILE" \
  "$@"
