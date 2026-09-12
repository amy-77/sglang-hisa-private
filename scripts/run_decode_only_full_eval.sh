#!/usr/bin/env bash
# Evaluate the trained M=8 router with official shared DSA during prefill and
# per-head Top-720 retrieval during decode. The run is append/resume safe.
set -euo pipefail

ROOT=/DATA/disk0/qyl
REPO="${ROOT}/code/dpskv32/sglang-hisa"
EXP="${ROOT}/data/misa_assignment_router_pilot_v2"
OUT="${ROOT}/data/decode_only_full_20260909"
CONTAINER=dsv32-decode-only-full-eval
IMAGE=qyl/sglang-hisa:eval
MODEL=/workspace/qyl/models/deepseek-v3.2
ROUTER=/workspace/qyl/data/misa_assignment_router_pilot_v2/training/runtime.json
PORT=31720

mkdir -p "${OUT}"
log() { echo "[$(date -Iseconds)] $*" | tee -a "${OUT}/run.log"; }

cleanup() {
    docker logs "${CONTAINER}" >"${OUT}/server.log" 2>&1 || true
    docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
if curl --fail --silent --max-time 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    log "FATAL: port ${PORT} is already occupied"
    exit 1
fi

# This router-enabled server currently profiles a ~101 GB/GPU model footprint,
# versus ~81 GB/GPU for the teacher-only probe. At 0.75 its KV pool held only
# 25,280 tokens; 0.82 restores enough capacity for the 32K/128K evaluation.
log "starting decode-only router server"
docker run -d --name "${CONTAINER}" \
    --gpus all --ipc host --network host \
    -v "${ROOT}:/workspace/qyl" \
    -e PYTHONPATH=/workspace/qyl/code/dpskv32/sglang-hisa/python \
    -e SGLANG_NSA_FUSE_TOPK=0 \
    -e SGLANG_NSA_PER_HEAD_INDEX=0 \
    -e SGLANG_NSA_EXPERIMENTAL_MIN_SEQ_LEN=4096 \
    -e SGLANG_NSA_EXPERIMENTAL_PREFILL_PER_HEAD=0 \
    -e SGLANG_NSA_OFFLINE_ROUTER_CONFIG="${ROUTER}" \
    -e SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
    "${IMAGE}" \
    python -m sglang.launch_server \
    --model-path "${MODEL}" \
    --served-model-name deepseek-v3.2 \
    --tp-size 8 --host 0.0.0.0 --port "${PORT}" \
    --trust-remote-code --reasoning-parser deepseek-v3 \
    --mem-fraction-static 0.82 --max-running-requests 1 \
    --disable-cuda-graph \
    --json-model-override-args '{"use_hisa":false}' >/dev/null

for i in $(seq 1 180); do
    if curl --fail --silent --max-time 5 \
        "http://127.0.0.1:${PORT}/health_generate" >/dev/null 2>&1; then
        log "server healthy after ${i} polls"
        break
    fi
    if [[ "$(docker inspect --format '{{.State.Running}}' "${CONTAINER}" 2>/dev/null || true)" != true ]]; then
        log "FATAL: server exited during startup"
        docker logs "${CONTAINER}" | tail -80
        exit 1
    fi
    if (( i == 180 )); then
        log "FATAL: server did not become healthy"
        exit 1
    fi
    sleep 10
done

log "evaluating 401 held-out LongBench-v2 examples"
docker run --rm --network host -v "${ROOT}:/workspace/qyl" "${IMAGE}" \
    python /workspace/qyl/code/dpskv32/evaluate_longbench_v2_e2e.py \
    --data /workspace/qyl/data/misa_assignment_router_pilot_v2/longbench_heldout_full.json \
    --model "${MODEL}" \
    --server "http://127.0.0.1:${PORT}" \
    --output /workspace/qyl/data/decode_only_full_20260909/longbench_predictions.jsonl \
    --summary /workspace/qyl/data/decode_only_full_20260909/longbench_summary.json \
    --max-context-tokens 131072 --max-new-tokens 128 \
    --concurrency 1 --resume \
    2>&1 | tee -a "${OUT}/longbench.log"

log "evaluating the disjoint fixed RULER test split"
docker run --rm --network host -v "${ROOT}:/workspace/qyl" "${IMAGE}" \
    python /workspace/qyl/code/dpskv32/evaluate_ruler_e2e.py \
    --data-root /workspace/qyl/data/ruler_deepseek_v3_2 \
    --model "${MODEL}" \
    --server "http://127.0.0.1:${PORT}" \
    --output /workspace/qyl/data/decode_only_full_20260909/ruler_predictions.jsonl \
    --summary /workspace/qyl/data/decode_only_full_20260909/ruler_summary.json \
    --lengths 32k 128k --resume \
    2>&1 | tee -a "${OUT}/ruler.log"

log "evaluating 520 independently generated RULER examples"
docker run --rm --network host -v "${ROOT}:/workspace/qyl" "${IMAGE}" \
    python /workspace/qyl/code/dpskv32/evaluate_ruler_e2e.py \
    --data-root /workspace/qyl/data/ruler_deepseek_v3_2_full20 \
    --model "${MODEL}" \
    --server "http://127.0.0.1:${PORT}" \
    --output /workspace/qyl/data/decode_only_full_20260909/ruler_expanded_predictions.jsonl \
    --summary /workspace/qyl/data/decode_only_full_20260909/ruler_expanded_summary.json \
    --lengths 32k 128k --all-records --resume \
    2>&1 | tee -a "${OUT}/ruler_expanded.log"

touch "${OUT}/COMPLETED"
log "decode-only full evaluation completed"
