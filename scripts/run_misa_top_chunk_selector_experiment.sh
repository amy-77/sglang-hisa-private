#!/usr/bin/env bash
# Replay frozen RULER queries and compare mean-pooled Top-K/Top-fraction
# aggregation for MISA's 64-to-8 Indexer-head selection.
set -euo pipefail

ROOT=/DATA/disk0/qyl
REPO="${ROOT}/code/dpskv32/sglang-hisa"
SOURCE="${ROOT}/data/ruler_router_decomposition_20260909"
EXP="${ROOT}/data/ruler_top_chunk_selectors_mean_20260910"
CONTAINER_EXP=/workspace/qyl/data/ruler_top_chunk_selectors_mean_20260910
IMAGE=qyl/sglang-hisa:eval
MODEL=/workspace/qyl/models/deepseek-v3.2
PORT=31721
CONTAINER=dsv32-ruler-top-chunk-selectors
LAYERS="$(seq -s, 0 60)"

mkdir -p "${EXP}/probe" "${EXP}/reports"
log() { echo "[$(date -Iseconds)] $*" | tee -a "${EXP}/run.log"; }
stop_server() { docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true; }
trap stop_server EXIT

EXPECTED_SAMPLES=$(python - "${SOURCE}/trajectories.jsonl" <<'PY'
import json, sys
count = 0
for line in open(sys.argv[1]):
    output_length = len(json.loads(line).get("output_ids") or [])
    count += sum(position <= output_length for position in (0, 4, 16, 64))
print(count)
PY
)

if [[ -f "${EXP}/samples.jsonl" ]] && python - "${EXP}/samples.jsonl" "${EXPECTED_SAMPLES}" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
raise SystemExit(0 if sum(row.get("status") == "ok" for row in rows) >= int(sys.argv[2]) else 1)
PY
then
    log "probe replay already complete (${EXPECTED_SAMPLES} samples)"
else
    log "starting mean Top-K/Top-fraction probe (${EXPECTED_SAMPLES} samples, 61 layers)"
    stop_server
    docker run -d \
        --name "${CONTAINER}" \
        --gpus all --ipc host --network host \
        -v "${ROOT}:/workspace/qyl" \
        -e PYTHONPATH=/workspace/qyl/code/dpskv32/sglang-hisa/python \
        -e SGLANG_NSA_FUSE_TOPK=0 \
        -e SGLANG_NSA_PER_HEAD_INDEX=0 \
        -e SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
        -e SGLANG_NSA_HEADMAP_PROBE_DIR="${CONTAINER_EXP}/probe" \
        -e SGLANG_NSA_HEADMAP_PROBE_LAYERS="${LAYERS}" \
        -e SGLANG_NSA_HEADMAP_PROBE_MIN_LEN=4096 \
        -e SGLANG_NSA_HEADMAP_PROBE_MAX_SAMPLES="${EXPECTED_SAMPLES}" \
        -e SGLANG_NSA_HEADMAP_PROBE_MISA_BUDGET=64 \
        -e SGLANG_NSA_HEADMAP_PROBE_MISA_BLOCK_SIZE=512 \
        -e SGLANG_NSA_HEADMAP_PROBE_MISA_KEEP=0.60 \
        -e SGLANG_NSA_HEADMAP_PROBE_SHARD_SIZE=16 \
        "${IMAGE}" \
        python -m sglang.launch_server \
        --model-path "${MODEL}" \
        --served-model-name deepseek-v3.2 \
        --tp-size 8 --host 0.0.0.0 --port "${PORT}" \
        --trust-remote-code --reasoning-parser deepseek-v3 \
        --mem-fraction-static 0.75 --max-running-requests 1 \
        --disable-cuda-graph \
        --json-model-override-args '{"use_hisa":false}' >/dev/null

    deadline=$((SECONDS + 1800))
    until curl --fail --silent --max-time 5 \
        "http://127.0.0.1:${PORT}/health_generate" >/dev/null 2>&1
    do
        if [[ "$(docker inspect --format '{{.State.Running}}' "${CONTAINER}" 2>/dev/null || true)" != true ]]; then
            docker logs "${CONTAINER}" >&2 || true
            exit 1
        fi
        if ((SECONDS >= deadline)); then
            docker logs "${CONTAINER}" >&2 || true
            exit 1
        fi
        sleep 10
    done

    docker run --rm --network host -v "${ROOT}:/workspace/qyl" "${IMAGE}" \
        python /workspace/qyl/code/dpskv32/sglang-hisa/scripts/collect_misa_router_samples.py \
        replay \
        --trajectories /workspace/qyl/data/ruler_router_decomposition_20260909/trajectories.jsonl \
        --server "http://127.0.0.1:${PORT}" \
        --resume \
        --out "${CONTAINER_EXP}/samples.jsonl" \
        2>&1 | tee "${EXP}/probe_driver.log"
    docker logs "${CONTAINER}" >"${EXP}/probe_server.log" 2>&1
    stop_server
fi

log "analyzing selector variants"
docker run --rm -v "${ROOT}:/workspace/qyl" \
    -e PYTHONPATH=/workspace/qyl/code/dpskv32/sglang-hisa/python \
    "${IMAGE}" \
    python /workspace/qyl/code/dpskv32/sglang-hisa/scripts/analyze_misa_top_chunk_selectors.py \
    --probe-dir "${CONTAINER_EXP}/probe" \
    --sample-manifest "${CONTAINER_EXP}/samples.jsonl" \
    --budget 8 \
    --out "${CONTAINER_EXP}/reports/top_chunk_selectors.json" \
    2>&1 | tee "${EXP}/reports/analyze.log"

touch "${EXP}/COMPLETED"
log "COMPLETED"
