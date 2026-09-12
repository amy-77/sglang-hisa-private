#!/usr/bin/env bash
# Group16 all-64 router: collect [8,64] teacher labels, then train four ablations.
#
# Fixed probe contract:
#   budget=64 / block=512 / keep=60% / save group-router features
#
# Usage:
#   STAGE=collect EXP_NAME=... TRAJECTORIES=... bash scripts/run_group16_all64_router.sh
#   STAGE=train   TRAIN_EXP=... TEST_EXP=... OUT_DIR=... bash scripts/run_group16_all64_router.sh
#   STAGE=all     ... both stages
set -euo pipefail

ROOT=/DATA/disk0/qyl
REPO="${ROOT}/code/dpskv32/sglang-hisa"
IMAGE=qyl/sglang-hisa:eval
MODEL=/workspace/qyl/models/deepseek-v3.2
PORT="${GROUP16_ROUTER_PORT:-31720}"
LAYERS="$(seq -s, 0 60)"
STAGE="${STAGE:-all}"
EXP_NAME="${EXP_NAME:-group16_indexer_router_all64}"
EXP="${ROOT}/data/${EXP_NAME}"
CONTAINER_EXP="/workspace/qyl/data/${EXP_NAME}"
TRAJECTORIES="${TRAJECTORIES:-${EXP}/trajectories.jsonl}"
CONTAINER_TRAJECTORIES="${CONTAINER_TRAJECTORIES:-${CONTAINER_EXP}/trajectories.jsonl}"
TRAIN_EXP="${TRAIN_EXP:-${EXP_NAME}}"
TEST_EXP="${TEST_EXP:-group16_indexer_router_test_all64}"
OUT_DIR="${OUT_DIR:-${ROOT}/data/${EXP_NAME}/training}"
CONTAINER_TAG="${EXP_NAME//_/-}"
PROBE_CONTAINER="dsv32-${CONTAINER_TAG}-probe"

# Frozen all-64 teacher contract.
PROBE_BUDGET=64
PROBE_BLOCK_SIZE=512
PROBE_KEEP=0.60

log() { echo "[$(date -Iseconds)] $*"; }

stop_container() {
    docker rm -f "$1" >/dev/null 2>&1 || true
}
trap 'stop_container "${PROBE_CONTAINER}"' EXIT

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

start_probe_server() {
    stop_container "${PROBE_CONTAINER}"
    if curl --fail --silent --max-time 5 \
        "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "port ${PORT} is already served by another container" >&2
        return 1
    fi
    docker run -d \
        --name "${PROBE_CONTAINER}" \
        --gpus all \
        --ipc host \
        --network host \
        -v "${ROOT}:/workspace/qyl" \
        -v "${ROOT}/cache/misa_router_pilot_v1:/root/.cache" \
        -e PYTHONPATH=/workspace/qyl/code/dpskv32/sglang-hisa/python \
        -e SGLANG_NSA_FUSE_TOPK=0 \
        -e SGLANG_NSA_PER_HEAD_INDEX=0 \
        -e SGLANG_NSA_EXPERIMENTAL_PREFILL_PER_HEAD=0 \
        -e SGLANG_NSA_EXPERIMENTAL_MIN_SEQ_LEN=4096 \
        -e SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
        -e SGLANG_NSA_HEADMAP_PROBE_DIR="${CONTAINER_EXP}/probe" \
        -e SGLANG_NSA_HEADMAP_PROBE_LAYERS="${LAYERS}" \
        -e SGLANG_NSA_HEADMAP_PROBE_MIN_LEN=4096 \
        -e SGLANG_NSA_HEADMAP_PROBE_MAX_SAMPLES="${SAMPLES}" \
        -e SGLANG_NSA_HEADMAP_PROBE_MISA_BUDGET="${PROBE_BUDGET}" \
        -e SGLANG_NSA_HEADMAP_PROBE_MISA_BLOCK_SIZE="${PROBE_BLOCK_SIZE}" \
        -e SGLANG_NSA_HEADMAP_PROBE_MISA_KEEP="${PROBE_KEEP}" \
        -e SGLANG_NSA_HEADMAP_PROBE_SAVE_GROUP_ROUTER_FEATURES=1 \
        -e SGLANG_NSA_HEADMAP_PROBE_SHARD_SIZE=16 \
        "${IMAGE}" \
        python -m sglang.launch_server \
        --model-path "${MODEL}" \
        --served-model-name deepseek-v3.2 \
        --tp-size 8 \
        --host 0.0.0.0 \
        --port "${PORT}" \
        --trust-remote-code \
        --reasoning-parser deepseek-v3 \
        --mem-fraction-static 0.75 \
        --max-running-requests 1 \
        --disable-cuda-graph \
        --json-model-override-args '{"use_hisa":false}' >/dev/null
    wait_for_server "${PROBE_CONTAINER}"
}

collect_stage() {
    log "collect all-64 Group16 teacher labels -> ${EXP}"
    if [[ ! -f "${TRAJECTORIES}" ]]; then
        echo "missing trajectories: ${TRAJECTORIES}" >&2
        exit 1
    fi
    python - "${TRAJECTORIES}" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
empty = sum(not r.get("output_ids") for r in rows)
print(f"trajectories: {len(rows)} rows, {empty} without output_ids")
raise SystemExit(1 if empty else 0)
PY
    SAMPLES="$(
        docker run --rm -v "${ROOT}:/workspace/qyl" "${IMAGE}" \
            python /workspace/qyl/code/dpskv32/sglang-hisa/scripts/collect_misa_router_samples.py \
            count \
            --trajectories "${CONTAINER_TRAJECTORIES}" \
            --min-query-seq-len 4096 \
            | awk 'NF { value=$0 } END { print value }'
    )"
    if [[ ! "${SAMPLES}" =~ ^[1-9][0-9]*$ ]]; then
        echo "no valid decode queries in ${TRAJECTORIES}" >&2
        exit 1
    fi
    log "expecting exactly ${SAMPLES} identified decode queries"
    rm -rf "${EXP}/probe"
    rm -f "${EXP}/samples.jsonl"
    mkdir -p "${EXP}/probe" "${ROOT}/cache/misa_router_pilot_v1"
    start_probe_server
    docker run --rm --network host \
        -v "${ROOT}:/workspace/qyl" \
        "${IMAGE}" \
        python /workspace/qyl/code/dpskv32/sglang-hisa/scripts/collect_misa_router_samples.py \
        replay \
        --trajectories "${CONTAINER_TRAJECTORIES}" \
        --server "http://127.0.0.1:${PORT}" \
        --min-query-seq-len 4096 \
        --out "${CONTAINER_EXP}/samples.jsonl" \
        2>&1 | tee "${EXP}/probe_driver.log"
    docker logs "${PROBE_CONTAINER}" >"${EXP}/probe_server.log" 2>&1
    stop_container "${PROBE_CONTAINER}"
    python - "${EXP}" "${SAMPLES}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
expected = int(sys.argv[2])
manifest = [
    json.loads(line)
    for line in (root / "samples.jsonl").read_text().splitlines()
    if line.strip()
]
request_ids = [row["request_id"] for row in manifest if row["status"] == "ok"]
if len(request_ids) != expected or len(set(request_ids)) != expected:
    raise SystemExit(
        f"manifest completeness failed: expected={expected}, "
        f"rows={len(request_ids)}, unique={len(set(request_ids))}"
    )
markers = {}
for path in (root / "probe").glob("complete_L*_rank*.json"):
    row = json.loads(path.read_text())
    markers[(int(row["layer"]), int(row["rank"]))] = int(row["records"])
expected_keys = {(layer, rank) for layer in range(61) for rank in range(8)}
if set(markers) != expected_keys:
    missing = sorted(expected_keys - set(markers))
    raise SystemExit(f"probe completion markers missing: {missing[:8]}")
bad = {key: count for key, count in markers.items() if count != expected}
if bad:
    raise SystemExit(f"probe completion counts differ: {list(bad.items())[:8]}")
print(f"validated {expected} requests x 61 layers x 8 TP ranks")
PY
    log "collect done: ${EXP}/probe + ${EXP}/samples.jsonl"
}

train_stage() {
    local train_root="${ROOT}/data/${TRAIN_EXP}"
    local test_root="${ROOT}/data/${TEST_EXP}"
    mkdir -p "${OUT_DIR}"
    local rel_out="${OUT_DIR#${ROOT}/}"
    if [[ "${rel_out}" == "${OUT_DIR}" ]]; then
        echo "OUT_DIR must be under ${ROOT}: ${OUT_DIR}" >&2
        exit 1
    fi
    log "train four Group16 routers"
    log "  train=${train_root}"
    log "  test=${test_root}"
    log "  out=${OUT_DIR}"
    docker run --rm --gpus all --ipc host \
        -v "${ROOT}:/workspace/qyl" \
        -e PYTHONPATH=/workspace/qyl/code/dpskv32/sglang-hisa/python \
        "${IMAGE}" \
        python /workspace/qyl/code/dpskv32/sglang-hisa/scripts/train_group16_indexer_router_ablation.py \
        --train-probe-dir "/workspace/qyl/data/${TRAIN_EXP}/probe" \
        --train-manifest "/workspace/qyl/data/${TRAIN_EXP}/samples.jsonl" \
        --test-probe-dir "/workspace/qyl/data/${TEST_EXP}/probe" \
        --test-manifest "/workspace/qyl/data/${TEST_EXP}/samples.jsonl" \
        --steps "${STEPS:-10000}" \
        --batch-size 32 \
        --rank 64 \
        --seed 20260912 \
        --out-dir "/workspace/qyl/${rel_out}" \
        2>&1 | tee "${OUT_DIR}/train.log"
}

case "${STAGE}" in
    collect) collect_stage ;;
    train) train_stage ;;
    all)
        collect_stage
        train_stage
        ;;
    *)
        echo "STAGE must be collect|train|all, got ${STAGE}" >&2
        exit 1
        ;;
esac
