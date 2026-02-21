#!/usr/bin/env bash
set -euo pipefail

VIDEO="${1:-}"
OUTDIR="${2:-results/reports}"
FRAMES="${3:-8}"
MODE="${4:-stub}"

if [[ -z "$VIDEO" ]]; then
echo "Usage: bash scripts/40_generate_report.sh <video.mp4> [outdir] [frames] [stub|vllm_openai|vllm_inprocess]"
  exit 1
fi

mkdir -p "$OUTDIR"
BASENAME="$(basename "$VIDEO")"
ID="${BASENAME%.*}"

trie report --video "$VIDEO" --out "$OUTDIR/${ID}.json" --frames "$FRAMES" --backend "$MODE"
