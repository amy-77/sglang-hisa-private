#!/usr/bin/env bash
# Does unified prefill (per-head Top-720 + coarse pruning + dense boundary) run?
#
# Launches one router-enabled server and drives prompts that exercise each
# prefill regime: entirely below the sparse threshold, straddling it, and long
# enough to need several chunks.  Any regime that raises leaves its traceback in
# the server log, so the exit status alone tells us whether the path is usable.
set -uo pipefail

ROOT=/DATA/disk0/qyl
EXP="${ROOT}/data/unified_prefill_smoke"
CONTAINER=dsv32-unified-prefill-smoke
IMAGE=qyl/sglang-hisa:eval
MODEL=/workspace/qyl/models/deepseek-v3.2
ROUTER=/workspace/qyl/data/misa_assignment_router_pilot_v2/training/runtime.json
PORT=31730

mkdir -p "${EXP}"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "${EXP}/smoke.log"; }

docker rm -f "${CONTAINER}" >/dev/null 2>&1
log "launching router-enabled server on :${PORT}"
docker run -d --name "${CONTAINER}" \
    --gpus all --ipc host --network host \
    -v "${ROOT}:/workspace/qyl" \
    -e PYTHONPATH=/workspace/qyl/code/dpskv32/sglang-hisa/python \
    -e SGLANG_NSA_FUSE_TOPK=0 \
    -e SGLANG_NSA_PER_HEAD_INDEX=0 \
    -e SGLANG_NSA_EXPERIMENTAL_MIN_SEQ_LEN=4096 \
    -e SGLANG_NSA_EXPERIMENTAL_PREFILL_PER_HEAD=1 \
    -e SGLANG_NSA_EXPERIMENTAL_TRACE_PREFILL=1 \
    -e SGLANG_NSA_OFFLINE_ROUTER_CONFIG="${ROUTER}" \
    -e SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
    "${IMAGE}" \
    python -m sglang.launch_server \
    --model-path "${MODEL}" \
    --served-model-name deepseek-v3.2 \
    --tp-size 8 --host 0.0.0.0 --port "${PORT}" \
    --trust-remote-code --reasoning-parser deepseek-v3 \
    --mem-fraction-static 0.75 --max-running-requests 2 \
    --disable-cuda-graph \
    --json-model-override-args '{"use_hisa":false}' >/dev/null

for i in $(seq 1 150); do
    if curl --fail --silent --max-time 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        log "server healthy after ${i} polls"
        break
    fi
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
        log "FATAL: container exited during startup"
        docker logs "${CONTAINER}" >"${EXP}/server.log" 2>&1
        tail -60 "${EXP}/server.log" | tee -a "${EXP}/smoke.log"
        exit 1
    fi
    sleep 20
done

docker run --rm --network host -v "${ROOT}:/workspace/qyl" \
    -e PYTHONPATH=/workspace/qyl/code/dpskv32/sglang-hisa/python \
    "${IMAGE}" python /workspace/qyl/code/dpskv32/sglang-hisa/scripts/smoke_unified_prefill.py \
    --server "http://127.0.0.1:${PORT}" \
    --model "${MODEL}" \
    --out "/workspace/qyl/data/unified_prefill_smoke/results.json" \
    2>&1 | tee -a "${EXP}/smoke.log"
STATUS=${PIPESTATUS[0]}

docker logs "${CONTAINER}" >"${EXP}/server.log" 2>&1
log "driver exit status ${STATUS}; server log at ${EXP}/server.log"
if grep -nE 'Traceback|NotImplementedError|AssertionError|CUDA error|device-side assert' "${EXP}/server.log" >"${EXP}/server_errors.txt" 2>&1; then
    log "server log contains errors:"
    head -40 "${EXP}/server_errors.txt" | tee -a "${EXP}/smoke.log"
fi
docker rm -f "${CONTAINER}" >/dev/null 2>&1
exit "${STATUS}"
