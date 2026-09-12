#!/usr/bin/env bash
# Serialize the GPU-exclusive stages of several Router experiments.
#
# One 671B server occupies all eight GPUs, so only one server can exist at a
# time.  The Official-DSA trajectory server is identical for every experiment,
# so trajectory generation is shared: all experiments' generate clients run
# against one server, then probe/train/eval run experiment by experiment.
#
# Usage: run_misa_router_queue.sh EXP_NAME [EXP_NAME ...]
set -euo pipefail

REPO=/DATA/disk0/qyl/code/dpskv32/sglang-hisa
ROOT=/DATA/disk0/qyl
PORT=31720
IMAGE=qyl/sglang-hisa:eval
MODEL=/workspace/qyl/models/deepseek-v3.2
PIPELINE="${REPO}/scripts/legacy/run_misa_router_pilot_v1.sh"
EXPS=("$@")
(( ${#EXPS[@]} )) || { echo "usage: $0 EXP_NAME..." >&2; exit 1; }

log() { echo "[$(date -Iseconds)] $*"; }

server_up() {
    curl --fail --silent --max-time 5 "http://127.0.0.1:${PORT}/health_generate" >/dev/null 2>&1
}

trajectories_complete() {
    python - "${ROOT}/data/$1/prompts.jsonl" "${ROOT}/data/$1/trajectories.jsonl" <<'PY'
import json, sys
prompts = sum(1 for l in open(sys.argv[1]) if l.strip())
try:
    rows = [json.loads(l) for l in open(sys.argv[2]) if l.strip()]
except FileNotFoundError:
    rows = []
ok = [r for r in rows if r.get("output_ids")]
print(f"  {len(ok)}/{prompts} valid trajectories", file=sys.stderr)
raise SystemExit(0 if len(ok) >= prompts else 1)
PY
}

# Stage 1: shared trajectory generation.  If a trajectory server is already
# serving the port (e.g. left over from an interrupted run), reuse it.  Any
# generate clients left over from an interrupted pipeline keep writing their
# own trajectories.jsonl; let them finish before starting a second writer.
while pgrep -f 'collect_misa_router_samples.py generate' >/dev/null; do
    sleep 30
done
pending=()
for exp in "${EXPS[@]}"; do
    if trajectories_complete "${exp}"; then
        log "${exp}: trajectories already complete"
    else
        pending+=("${exp}")
    fi
done
if (( ${#pending[@]} )); then
    if server_up; then
        log "reusing running server on :${PORT} for trajectories: ${pending[*]}"
    else
        docker ps --format '{{.Names}}' | grep -E '^dsv32-' | xargs -r docker rm -f >/dev/null 2>&1 || true
        # Bring up a trajectory server via the first pending experiment's
        # pipeline, but without its generate client (STAGES empty except a
        # no-op) -- simpler: run its trajectories stage; other experiments join.
        log "starting trajectory server via ${pending[0]}"
        MISA_ROUTER_EXP_NAME="${pending[0]}" MISA_ROUTER_STAGES=trajectories \
            MISA_ROUTER_MAX_RUNNING_REQUESTS=4 MISA_ROUTER_TRAJECTORY_CONCURRENCY=4 \
            bash "${PIPELINE}" &
        # Wait until the server is healthy, then the remaining experiments
        # generate concurrently against it.
        until server_up; do sleep 15; done
        pending=("${pending[@]:1}")
    fi
    pids=()
    for exp in "${pending[@]}"; do
        log "${exp}: generating trajectories on shared server"
        docker run --rm --network host -v "${ROOT}:/workspace/qyl" "${IMAGE}" \
            python /workspace/qyl/code/dpskv32/sglang-hisa/scripts/collect_misa_router_samples.py \
            generate --prompts "/workspace/qyl/data/${exp}/prompts.jsonl" --model "${MODEL}" \
            --server "http://127.0.0.1:${PORT}" --long-max-new-tokens 96 \
            --reasoning-max-new-tokens 512 --concurrency 4 --resume \
            --out "/workspace/qyl/data/${exp}/trajectories.jsonl" \
            >"${ROOT}/data/${exp}/trajectory_driver.log" 2>&1 &
        pids+=($!)
    done
    wait
    docker ps --format '{{.Names}}' | grep -E '^dsv32-.*-trajectories$' | xargs -r docker rm -f >/dev/null 2>&1 || true
fi
for exp in "${EXPS[@]}"; do
    trajectories_complete "${exp}" || { log "${exp}: trajectories incomplete, aborting"; exit 1; }
done

# Stage 2: GPU-exclusive probe -> train -> eval, one experiment at a time.
for exp in "${EXPS[@]}"; do
    if [[ -f "${ROOT}/data/${exp}/PIPELINE_COMPLETED" ]]; then
        log "${exp}: already completed"; continue
    fi
    log "${exp}: probe/train/eval"
    MISA_ROUTER_EXP_NAME="${exp}" MISA_ROUTER_STAGES=probe,train,eval \
        MISA_ROUTER_MAX_SAMPLES=8000 bash "${PIPELINE}"
done
log "QUEUE_COMPLETED"
