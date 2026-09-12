#!/usr/bin/env bash
set -euo pipefail

ROOT=/DATA/disk0/qyl
REPO="${ROOT}/code/dpskv32/sglang-hisa"
EXP_NAME="${MISA_ROUTER_EXP_NAME:-misa_assignment_router_group16}"
EXP="${ROOT}/data/${EXP_NAME}"
CONTAINER_EXP="/workspace/qyl/data/${EXP_NAME}"
IMAGE=qyl/sglang-hisa:eval
MODEL=/workspace/qyl/models/deepseek-v3.2
PORT=31720
LAYERS="$(seq -s, 0 60)"
SAMPLES="${MISA_ROUTER_MAX_SAMPLES:-5000}"
MAX_RUNNING_REQUESTS="${MISA_ROUTER_MAX_RUNNING_REQUESTS:-1}"
TRAJECTORY_CONCURRENCY="${MISA_ROUTER_TRAJECTORY_CONCURRENCY:-1}"
PROBE_SAMPLE_STRIDE="${MISA_ROUTER_PROBE_SAMPLE_STRIDE:-1}"
PROBE_BUDGET="${MISA_ROUTER_PROBE_BUDGET:-8}"
PROBE_BLOCK_SIZE="${MISA_ROUTER_PROBE_BLOCK_SIZE:-256}"
PROBE_KEEP="${MISA_ROUTER_PROBE_KEEP:-}"
PROBE_TOPK="${MISA_ROUTER_PROBE_TOPK:-8}"
CONTAINER_TAG="${EXP_NAME//_/-}"
TRAJECTORY_CONTAINER="dsv32-${CONTAINER_TAG}-trajectories"
PROBE_CONTAINER="dsv32-${CONTAINER_TAG}-probe"
EVAL_CONTAINER="dsv32-${CONTAINER_TAG}-eval"
PROMPTS="${EXP}/prompts.jsonl"
TRAJECTORIES="${EXP}/trajectories.jsonl"
CONTAINER_PROMPTS="${CONTAINER_EXP}/prompts.jsonl"
CONTAINER_TRAJECTORIES="${CONTAINER_EXP}/trajectories.jsonl"
# A new candidate policy must replay the exact same token trajectories to make
# its oracle gate comparable, but should write probes/checkpoints to a fresh
# experiment directory. These overrides avoid copying a large corpus.
PROMPTS="${MISA_ROUTER_PROMPTS:-${PROMPTS}}"
TRAJECTORIES="${MISA_ROUTER_TRAJECTORIES:-${TRAJECTORIES}}"
CONTAINER_PROMPTS="${MISA_ROUTER_CONTAINER_PROMPTS:-${CONTAINER_PROMPTS}}"
CONTAINER_TRAJECTORIES="${MISA_ROUTER_CONTAINER_TRAJECTORIES:-${CONTAINER_TRAJECTORIES}}"
# Comma-separated subset of: trajectories,probe,train,eval.  Lets a queue
# orchestrator share one Official-DSA server across experiments and then run
# the GPU-exclusive stages back to back.
STAGES="${MISA_ROUTER_STAGES:-trajectories,probe,train,eval}"

stage_enabled() {
    [[ ",${STAGES}," == *",$1,"* ]]
}

mkdir -p "${EXP}/probe" "${EXP}/training" "${EXP}/evaluation" \
    "${ROOT}/cache/misa_router_pilot_v1"

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
    local router_config="$2"
    local fuse_topk="${SERVER_FUSE_TOPK:-0}"
    local graph_args=(--disable-cuda-graph)
    if [[ "${SERVER_DISABLE_CUDA_GRAPH:-1}" == 0 ]]; then
        graph_args=()
    fi
    shift 2
    stop_container "${container}"
    # A foreign server on our port would answer the health check and make a
    # misconfigured stage (e.g. probe without probe env) look healthy.
    if curl --fail --silent --max-time 5 \
        "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "port ${PORT} is already served by another container; refusing to start ${container}" >&2
        docker ps --format '{{.Names}} {{.Status}}' | grep dsv32 >&2 || true
        return 1
    fi
    docker run -d \
        --name "${container}" \
        --gpus all \
        --ipc host \
        --network host \
        -v "${ROOT}:/workspace/qyl" \
        -v "${ROOT}/cache/misa_router_pilot_v1:/root/.cache" \
        -e PYTHONPATH=/workspace/qyl/code/dpskv32/sglang-hisa/python \
        -e SGLANG_NSA_FUSE_TOPK="${fuse_topk}" \
        -e SGLANG_NSA_PER_HEAD_INDEX=0 \
        -e SGLANG_NSA_EXPERIMENTAL_MIN_SEQ_LEN=4096 \
        -e SGLANG_NSA_OFFLINE_ROUTER_CONFIG="${router_config}" \
        -e SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
        "$@" \
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
        --max-running-requests "${MAX_RUNNING_REQUESTS}" \
        "${graph_args[@]}" \
        --json-model-override-args '{"use_hisa":false}' >/dev/null
    wait_for_server "${container}"
}

if stage_enabled trajectories; then
echo "[$(date -Iseconds)] generating official-DSA trajectories"
if [[ ! -f "${PROMPTS}" ]]; then
    if [[ -f "${TRAJECTORIES}" ]] && python - "${TRAJECTORIES}" <<'PY'
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
raise SystemExit(0 if rows and all(not row.get("output_ids") for row in rows) else 1)
PY
    then
        mv "${TRAJECTORIES}" "${PROMPTS}"
    else
        echo "missing ${PROMPTS}; run build_misa_router_pilot.py --out ${PROMPTS}" >&2
        exit 1
    fi
fi
stop_container "${TRAJECTORY_CONTAINER}"
SERVER_FUSE_TOPK=1 SERVER_DISABLE_CUDA_GRAPH=0 \
    start_server "${TRAJECTORY_CONTAINER}" ""
docker run --rm --network host \
    -v "${ROOT}:/workspace/qyl" \
    "${IMAGE}" \
    python /workspace/qyl/code/dpskv32/sglang-hisa/scripts/collect_misa_router_samples.py \
    generate \
    --prompts "${CONTAINER_PROMPTS}" \
    --model "${MODEL}" \
    --server "http://127.0.0.1:${PORT}" \
    --long-max-new-tokens 96 \
    --reasoning-max-new-tokens 512 \
    --concurrency "${TRAJECTORY_CONCURRENCY}" \
    --resume \
    --out "${CONTAINER_TRAJECTORIES}" \
    2>&1 | tee "${EXP}/trajectory_driver.log"
stop_container "${TRAJECTORY_CONTAINER}"
fi

if stage_enabled probe; then
echo "[$(date -Iseconds)] starting MISA teacher collection"
python - "${TRAJECTORIES}" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
empty = sum(not r.get("output_ids") for r in rows)
print(f"trajectories: {len(rows)} rows, {empty} without output_ids")
raise SystemExit(1 if empty else 0)
PY
rm -rf "${EXP}/probe"
rm -f "${EXP}/samples.jsonl"
mkdir -p "${EXP}/probe"
PROBE_PRUNE_ENV=()
if [[ -n "${PROBE_KEEP}" ]]; then
    PROBE_PRUNE_ENV+=(
        -e "SGLANG_NSA_HEADMAP_PROBE_MISA_KEEP=${PROBE_KEEP}"
    )
else
    PROBE_PRUNE_ENV+=(
        -e "SGLANG_NSA_HEADMAP_PROBE_MISA_TOPK=${PROBE_TOPK}"
    )
fi
start_server "${PROBE_CONTAINER}" "" \
    -e SGLANG_NSA_HEADMAP_PROBE_DIR="${CONTAINER_EXP}/probe" \
    -e SGLANG_NSA_HEADMAP_PROBE_LAYERS="${LAYERS}" \
    -e SGLANG_NSA_HEADMAP_PROBE_MIN_LEN=4096 \
    -e SGLANG_NSA_HEADMAP_PROBE_MAX_SAMPLES="${SAMPLES}" \
    -e SGLANG_NSA_HEADMAP_PROBE_MISA_BUDGET="${PROBE_BUDGET}" \
    -e SGLANG_NSA_HEADMAP_PROBE_MISA_BLOCK_SIZE="${PROBE_BLOCK_SIZE}" \
    -e SGLANG_NSA_HEADMAP_PROBE_SAVE_EXACT_TOPK_IDS="${MISA_ROUTER_SAVE_EXACT_TOPK_IDS:-0}" \
    -e SGLANG_NSA_HEADMAP_PROBE_SAVE_GROUP_ROUTER_FEATURES="${MISA_ROUTER_SAVE_GROUP_ROUTER_FEATURES:-0}" \
    -e SGLANG_NSA_HEADMAP_PROBE_SHARD_SIZE=16 \
    "${PROBE_PRUNE_ENV[@]}"

docker run --rm --network host \
    -v "${ROOT}:/workspace/qyl" \
    "${IMAGE}" \
    python /workspace/qyl/code/dpskv32/sglang-hisa/scripts/collect_misa_router_samples.py \
    replay \
    --trajectories "${CONTAINER_TRAJECTORIES}" \
    --server "http://127.0.0.1:${PORT}" \
    --out "${CONTAINER_EXP}/samples.jsonl" \
    2>&1 | tee "${EXP}/probe_driver.log"
docker logs "${PROBE_CONTAINER}" >"${EXP}/probe_server.log" 2>&1
stop_container "${PROBE_CONTAINER}"
fi

if stage_enabled train; then
echo "[$(date -Iseconds)] training Assignment Router"
docker run --rm --gpus all --ipc host \
    -v "${ROOT}:/workspace/qyl" \
    -e PYTHONPATH=/workspace/qyl/code/dpskv32/sglang-hisa/python \
    "${IMAGE}" \
    python /workspace/qyl/code/dpskv32/sglang-hisa/scripts/train_misa_assignment_router.py \
    --probe-dir "${CONTAINER_EXP}/probe" \
    --sample-manifest "${CONTAINER_EXP}/samples.jsonl" \
    --probe-sample-stride "${PROBE_SAMPLE_STRIDE}" \
    --min-seq-len 4096 \
    --steps 10000 \
    --batch-size 32 \
    --rank 64 \
    --temperature 0.02 \
    --regret-weight 1.0 \
    --top1-weight 0.1 \
    --checkpoint "${CONTAINER_EXP}/training/router.pt" \
    --out "${CONTAINER_EXP}/training/report.json" \
    2>&1 | tee "${EXP}/training/train.log"

docker run --rm \
    -v "${ROOT}:/workspace/qyl" \
    "${IMAGE}" \
    python /workspace/qyl/code/dpskv32/sglang-hisa/scripts/build_offline_router_runtime_config.py \
    --checkpoint "${CONTAINER_EXP}/training/router.pt" \
    --budget 8 \
    --misa-chunk-size 256 \
    --misa-prune-topk 8 \
    --min-seq-len 4096 \
    --out "${CONTAINER_EXP}/training/runtime.json"
fi

if stage_enabled eval; then
echo "[$(date -Iseconds)] evaluating held-out LongBench-v2 and RULER"
start_server "${EVAL_CONTAINER}" \
    "${CONTAINER_EXP}/training/runtime.json"

docker run --rm --network host \
    -v "${ROOT}:/workspace/qyl" \
    "${IMAGE}" \
    python /workspace/qyl/code/dpskv32/evaluate_longbench_v2_e2e.py \
    --data "${CONTAINER_EXP}/longbench_test.json" \
    --model "${MODEL}" \
    --server "http://127.0.0.1:${PORT}" \
    --output "${CONTAINER_EXP}/evaluation/longbench_predictions.jsonl" \
    --summary "${CONTAINER_EXP}/evaluation/longbench_summary.json" \
    --max-context-tokens 131072 \
    --max-new-tokens 128 \
    --concurrency 1 \
    2>&1 | tee "${EXP}/evaluation/longbench.log"

docker run --rm --network host \
    -v "${ROOT}:/workspace/qyl" \
    "${IMAGE}" \
    python /workspace/qyl/code/dpskv32/evaluate_ruler_e2e.py \
    --data-root /workspace/qyl/data/ruler_deepseek_v3_2 \
    --model "${MODEL}" \
    --server "http://127.0.0.1:${PORT}" \
    --output "${CONTAINER_EXP}/evaluation/ruler_predictions.jsonl" \
    --summary "${CONTAINER_EXP}/evaluation/ruler_summary.json" \
    --lengths 32k 128k \
    2>&1 | tee "${EXP}/evaluation/ruler.log"

docker logs "${EVAL_CONTAINER}" >"${EXP}/evaluation/server.log" 2>&1
stop_container "${EVAL_CONTAINER}"
touch "${EXP}/PIPELINE_COMPLETED"
echo "[$(date -Iseconds)] PIPELINE_COMPLETED"
fi
