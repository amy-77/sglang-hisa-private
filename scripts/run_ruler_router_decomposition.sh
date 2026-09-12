#!/usr/bin/env bash
# Pause-owned GPU diagnostic: all-64 candidate probe on fixed RULER test queries,
# then set-selection vs assignment regret decomposition.
set -euo pipefail

ROOT=/DATA/disk0/qyl
REPO="${ROOT}/code/dpskv32/sglang-hisa"
EXP="${ROOT}/data/ruler_router_decomposition_20260909"
CONTAINER_EXP="/workspace/qyl/data/ruler_router_decomposition_20260909"
IMAGE=qyl/sglang-hisa:eval
MODEL=/workspace/qyl/models/deepseek-v3.2
PORT=31720
LAYERS="$(seq -s, 0 60)"
TRAJECTORY_CONTAINER=dsv32-ruler-decomp-trajectories
PROBE_CONTAINER=dsv32-ruler-decomp-probe
CHECKPOINT=/workspace/qyl/data/misa_assignment_router_pilot_v2/training/router.pt

mkdir -p "${EXP}/probe" "${EXP}/reports"
log() { echo "[$(date -Iseconds)] $*" | tee -a "${EXP}/run.log"; }

stop_container() {
    docker rm -f "$1" >/dev/null 2>&1 || true
}

wait_for_server() {
    local container="$1"
    local deadline=$((SECONDS + 1800))
    while ((SECONDS < deadline)); do
        if curl --fail --silent --max-time 5 \
            "http://127.0.0.1:${PORT}/health_generate" >/dev/null 2>&1; then
            return
        fi
        if [[ "$(docker inspect --format '{{.State.Running}}' "${container}" 2>/dev/null || true)" != true ]]; then
            docker logs "${container}" >&2 || true
            return 1
        fi
        sleep 10
    done
    return 1
}

start_server() {
    local container="$1"
    shift
    stop_container "${container}"
    if curl --fail --silent --max-time 5 \
        "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "port ${PORT} already occupied" >&2
        return 1
    fi
    docker run -d \
        --name "${container}" \
        --gpus all --ipc host --network host \
        -v "${ROOT}:/workspace/qyl" \
        -e PYTHONPATH=/workspace/qyl/code/dpskv32/sglang-hisa/python \
        -e SGLANG_NSA_FUSE_TOPK=1 \
        -e SGLANG_NSA_PER_HEAD_INDEX=0 \
        -e SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
        "$@" \
        "${IMAGE}" \
        python -m sglang.launch_server \
        --model-path "${MODEL}" \
        --served-model-name deepseek-v3.2 \
        --tp-size 8 --host 0.0.0.0 --port "${PORT}" \
        --trust-remote-code --reasoning-parser deepseek-v3 \
        --mem-fraction-static 0.75 --max-running-requests 1 \
        --disable-cuda-graph \
        --json-model-override-args '{"use_hisa":false}' >/dev/null
    wait_for_server "${container}"
}

# ---- Stage A: fixed 26 RULER test prompts ---------------------------------
if [[ ! -f "${EXP}/prompts.jsonl" ]]; then
    log "building fixed RULER test prompts"
    docker run --rm -v "${ROOT}:/workspace/qyl" "${IMAGE}" \
        python /workspace/qyl/code/dpskv32/sglang-hisa/scripts/build_ruler_test_prompts.py \
        --data-root /workspace/qyl/data/ruler_deepseek_v3_2 \
        --out "${CONTAINER_EXP}/prompts.jsonl"
fi

# ---- Stage B: Official-DSA trajectories ----------------------------------
if [[ ! -f "${EXP}/trajectories.jsonl" ]] || \
   ! python - "${EXP}/prompts.jsonl" "${EXP}/trajectories.jsonl" <<'PY'
import json, sys
prompts = sum(1 for l in open(sys.argv[1]) if l.strip())
try:
    rows = [json.loads(l) for l in open(sys.argv[2]) if l.strip()]
except FileNotFoundError:
    rows = []
ok = [r for r in rows if r.get("output_ids")]
raise SystemExit(0 if len(ok) >= prompts else 1)
PY
then
    log "generating Official-DSA trajectories for fixed RULER test"
    start_server "${TRAJECTORY_CONTAINER}"
    docker run --rm --network host -v "${ROOT}:/workspace/qyl" "${IMAGE}" \
        python /workspace/qyl/code/dpskv32/sglang-hisa/scripts/collect_misa_router_samples.py \
        generate \
        --prompts "${CONTAINER_EXP}/prompts.jsonl" \
        --model "${MODEL}" \
        --server "http://127.0.0.1:${PORT}" \
        --long-max-new-tokens 96 \
        --reasoning-max-new-tokens 96 \
        --concurrency 1 \
        --resume \
        --out "${CONTAINER_EXP}/trajectories.jsonl" \
        2>&1 | tee "${EXP}/trajectory_driver.log"
    stop_container "${TRAJECTORY_CONTAINER}"
else
    log "trajectories already complete"
fi

# ---- Stage C: all-64 probe -----------------------------------------------
EXPECTED_SAMPLES=$(python - "${EXP}/trajectories.jsonl" <<'PY'
import json, sys
from pathlib import Path
# Mirror collect_misa_router_samples.replay_positions for long-context rows.
count = 0
for line in open(sys.argv[1]):
    row = json.loads(line)
    out_len = len(row.get("output_ids") or [])
    for pos in (0, 4, 16, 64):
        if pos <= out_len:
            count += 1
print(count)
PY
)
log "expected probe samples: ${EXPECTED_SAMPLES}"

COMPLETE_PROBE=0
if [[ -f "${EXP}/samples.jsonl" ]]; then
    if python - "${EXP}/samples.jsonl" "${EXPECTED_SAMPLES}" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
ok = sum(1 for row in rows if row.get("status") == "ok")
raise SystemExit(0 if ok >= int(sys.argv[2]) else 1)
PY
    then
        COMPLETE_PROBE=1
    fi
fi

if [[ "${COMPLETE_PROBE}" != 1 ]]; then
    log "starting all-64 MISA teacher probe"
    rm -rf "${EXP}/probe"
    mkdir -p "${EXP}/probe"
    start_server "${PROBE_CONTAINER}" \
        -e SGLANG_NSA_FUSE_TOPK=0 \
        -e SGLANG_NSA_HEADMAP_PROBE_DIR="${CONTAINER_EXP}/probe" \
        -e SGLANG_NSA_HEADMAP_PROBE_LAYERS="${LAYERS}" \
        -e SGLANG_NSA_HEADMAP_PROBE_MIN_LEN=4096 \
        -e SGLANG_NSA_HEADMAP_PROBE_MAX_SAMPLES="${EXPECTED_SAMPLES}" \
        -e SGLANG_NSA_HEADMAP_PROBE_MISA_BUDGET=64 \
        -e SGLANG_NSA_HEADMAP_PROBE_MISA_BLOCK_SIZE=512 \
        -e SGLANG_NSA_HEADMAP_PROBE_MISA_KEEP=0.60 \
        -e SGLANG_NSA_HEADMAP_PROBE_SHARD_SIZE=16
    docker run --rm --network host -v "${ROOT}:/workspace/qyl" "${IMAGE}" \
        python /workspace/qyl/code/dpskv32/sglang-hisa/scripts/collect_misa_router_samples.py \
        replay \
        --trajectories "${CONTAINER_EXP}/trajectories.jsonl" \
        --server "http://127.0.0.1:${PORT}" \
        --resume \
        --out "${CONTAINER_EXP}/samples.jsonl" \
        2>&1 | tee "${EXP}/probe_driver.log"
    docker logs "${PROBE_CONTAINER}" >"${EXP}/probe_server.log" 2>&1
    stop_container "${PROBE_CONTAINER}"
else
    log "all-64 probe already complete"
fi

# ---- Stage D: regret decomposition ---------------------------------------
log "computing set-selection and assignment regret"
docker run --rm --gpus all --ipc host \
    -v "${ROOT}:/workspace/qyl" \
    -e PYTHONPATH=/workspace/qyl/code/dpskv32/sglang-hisa/python \
    "${IMAGE}" \
    python /workspace/qyl/code/dpskv32/sglang-hisa/scripts/validate_misa_router.py \
    --probe-dir "${CONTAINER_EXP}/probe" \
    --sample-manifest "${CONTAINER_EXP}/samples.jsonl" \
    --checkpoint "${CHECKPOINT}" \
    --budget 8 \
    --dataset ruler \
    --split test \
    --out "${CONTAINER_EXP}/reports/decomposition.json" \
    2>&1 | tee "${EXP}/reports/validate.log"

touch "${EXP}/DECOMPOSITION_COMPLETED"
log "DECOMPOSITION_COMPLETED"
