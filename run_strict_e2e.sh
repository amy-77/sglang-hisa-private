#!/usr/bin/env bash
set -euo pipefail

ROOT=/DATA/disk0/qyl
OUT="${ROOT}/data/static_group16_e2e_20260913/strict_heldout"
IMAGE=qyl/sglang-hisa:eval
MODEL=/workspace/qyl/models/deepseek-v3.2
PORT=31730
STATIC_CONTAINER=dsv32-static-group16-smoke
OFFICIAL_CONTAINER=dsv32-official-dsa-strict-e2e
AIME_INDICES=14,20,22
MATH_INDICES="$(cat "${OUT}/math500_heldout_indices.txt")"

run_longbench() {
    local arm="$1"
    docker run --rm --network host -v "${ROOT}:/workspace/qyl" "${IMAGE}" \
        python /workspace/qyl/code/dpskv32/evaluate_longbench_v2_e2e.py \
        --data /workspace/qyl/data/static_group16_e2e_20260913/strict_heldout/longbench_heldout.json \
        --model "${MODEL}" --server "http://127.0.0.1:${PORT}" \
        --output "/workspace/qyl/data/static_group16_e2e_20260913/strict_heldout/${arm}/longbench_predictions.jsonl" \
        --summary "/workspace/qyl/data/static_group16_e2e_20260913/strict_heldout/${arm}/longbench_summary.json" \
        --max-context-tokens 131072 --max-new-tokens 128 --concurrency 1 --resume \
        2>&1 | tee -a "${OUT}/${arm}/longbench.log"
}

run_ruler() {
    local arm="$1"
    docker run --rm --network host -v "${ROOT}:/workspace/qyl" "${IMAGE}" \
        python /workspace/qyl/code/dpskv32/evaluate_ruler_e2e.py \
        --data-root /workspace/qyl/data/static_group16_e2e_20260913/strict_heldout/ruler \
        --model "${MODEL}" --server "http://127.0.0.1:${PORT}" \
        --output "/workspace/qyl/data/static_group16_e2e_20260913/strict_heldout/${arm}/ruler_predictions.jsonl" \
        --summary "/workspace/qyl/data/static_group16_e2e_20260913/strict_heldout/${arm}/ruler_summary.json" \
        --lengths 32k 128k --all-records --resume \
        2>&1 | tee -a "${OUT}/${arm}/ruler.log"
}

run_aime() {
    local arm="$1"
    docker run --rm --network host -v "${ROOT}:/workspace/qyl" "${IMAGE}" \
        python /workspace/qyl/code/dpskv32/evaluate_aime_e2e.py \
        --data /workspace/qyl/code/dpskv32/aime2025_data.json \
        --model "${MODEL}" --server "http://127.0.0.1:${PORT}" \
        --output "/workspace/qyl/data/static_group16_e2e_20260913/strict_heldout/${arm}/aime_results.jsonl" \
        --summary "/workspace/qyl/data/static_group16_e2e_20260913/strict_heldout/${arm}/aime_summary.json" \
        --only-indices "${AIME_INDICES}" --max-new-tokens 65536 \
        --temperature 1.0 --top-p 0.95 --seed 2025 --num-samples 1 \
        --concurrency 1 --timeout-seconds 43200 --resume \
        2>&1 | tee -a "${OUT}/${arm}/aime.log"
}

run_math500() {
    local arm="$1"
    docker run --rm --network host -v "${ROOT}:/workspace/qyl" \
        -e PYTHONPATH=/workspace/qyl/cache/math500_grader_py:/workspace/qyl/code/SeerAttention/eval/reasoning_tasks \
        "${IMAGE}" \
        python /workspace/qyl/code/dpskv32/evaluate_math500_e2e.py \
        --data /workspace/qyl/code/SeerAttention/eval/reasoning_tasks/data/math/test.jsonl \
        --model "${MODEL}" --server "http://127.0.0.1:${PORT}" \
        --output "/workspace/qyl/data/static_group16_e2e_20260913/strict_heldout/${arm}/math500_results.jsonl" \
        --summary "/workspace/qyl/data/static_group16_e2e_20260913/strict_heldout/${arm}/math500_summary.json" \
        --only-indices "${MATH_INDICES}" --max-new-tokens 32768 --resume \
        2>&1 | tee -a "${OUT}/${arm}/math500.log"
}

wait_for_server() {
    local container="$1"
    for _ in $(seq 1 180); do
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

start_static_server() {
    if curl --fail --silent --max-time 5 \
        "http://127.0.0.1:${PORT}/health_generate" >/dev/null 2>&1; then
        return
    fi
    docker rm -f "${STATIC_CONTAINER}" >/dev/null 2>&1 || true
    docker run -d --name "${STATIC_CONTAINER}" \
        --gpus all --ipc host --network host -v "${ROOT}:/workspace/qyl" \
        -e PYTHONPATH=/workspace/qyl/code/dpskv32/sglang-hisa/python \
        -e SGLANG_NSA_FUSE_TOPK=0 \
        -e SGLANG_NSA_PER_HEAD_INDEX=0 \
        -e SGLANG_NSA_EXPERIMENTAL_MIN_SEQ_LEN=4096 \
        -e SGLANG_NSA_EXPERIMENTAL_PREFILL_PER_HEAD=0 \
        -e SGLANG_NSA_OFFLINE_ROUTER_CONFIG=/workspace/qyl/data/group16_static_calibration_all64_full_20260912/static_group16_runtime.json \
        -e SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
        "${IMAGE}" python -m sglang.launch_server \
        --model-path "${MODEL}" --served-model-name deepseek-v3.2 \
        --tp-size 8 --host 0.0.0.0 --port "${PORT}" \
        --trust-remote-code --reasoning-parser deepseek-v3 \
        --mem-fraction-static 0.82 --max-running-requests 1 \
        --disable-cuda-graph --json-model-override-args '{"use_hisa":false}' >/dev/null
    wait_for_server "${STATIC_CONTAINER}"
}

mkdir -p "${OUT}/static_group16" "${OUT}/official_dsa"
start_static_server

run_longbench static_group16
run_ruler static_group16
run_aime static_group16
run_math500 static_group16
docker logs "${STATIC_CONTAINER}" >"${OUT}/static_group16/server.log" 2>&1
docker rm -f "${STATIC_CONTAINER}" >/dev/null 2>&1

docker rm -f "${OFFICIAL_CONTAINER}" >/dev/null 2>&1 || true
docker run -d --name "${OFFICIAL_CONTAINER}" \
    --gpus all --ipc host --network host -v "${ROOT}:/workspace/qyl" \
    -e PYTHONPATH=/workspace/qyl/code/dpskv32/sglang-hisa/python \
    -e SGLANG_NSA_FUSE_TOPK=0 \
    -e SGLANG_NSA_PER_HEAD_INDEX=0 \
    -e SGLANG_NSA_EXPERIMENTAL_PREFILL_PER_HEAD=0 \
    -e SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
    "${IMAGE}" python -m sglang.launch_server \
    --model-path "${MODEL}" --served-model-name deepseek-v3.2 \
    --tp-size 8 --host 0.0.0.0 --port "${PORT}" \
    --trust-remote-code --reasoning-parser deepseek-v3 \
    --mem-fraction-static 0.82 --max-running-requests 1 \
    --disable-cuda-graph --json-model-override-args '{"use_hisa":false}' >/dev/null
wait_for_server "${OFFICIAL_CONTAINER}"

run_longbench official_dsa
run_ruler official_dsa
run_aime official_dsa
run_math500 official_dsa
docker logs "${OFFICIAL_CONTAINER}" >"${OUT}/official_dsa/server.log" 2>&1
docker rm -f "${OFFICIAL_CONTAINER}" >/dev/null 2>&1

python - "${OUT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = json.loads((root / "manifest.json").read_text())
report = {"manifest": manifest, "datasets": {}}
for dataset, metric in (
    ("longbench", "accuracy"),
    ("ruler", "accuracy"),
    ("aime", "pass_at_1"),
    ("math500", "accuracy"),
):
    static = json.loads(
        (root / "static_group16" / f"{dataset}_summary.json").read_text()
    )
    official = json.loads(
        (root / "official_dsa" / f"{dataset}_summary.json").read_text()
    )
    report["datasets"][dataset] = {
        "metric": metric,
        "static_group16": static,
        "official_dsa": official,
        "delta": float(static[metric]) - float(official[metric]),
    }
(root / "comparison_report.json").write_text(json.dumps(report, indent=2))
PY

touch "${OUT}/COMPLETED"
