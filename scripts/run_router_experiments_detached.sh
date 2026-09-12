#!/usr/bin/env bash
# Durable experiment sequence. Safe to leave running after the Cursor session
# disconnects; launch with setsid/nohup as shown at the bottom of this file.
set -euo pipefail

ROOT=/DATA/disk0/qyl
REPO="${ROOT}/code/dpskv32/sglang-hisa"
STATE="${ROOT}/data/misa_router_detached"
PIPELINE="${REPO}/scripts/legacy/run_misa_router_pilot_v1.sh"
PILOT=misa_assignment_router_pilot_v2
ZERO=misa_assignment_router_zero_shot

mkdir -p "${STATE}"
exec 9>"${STATE}/pipeline.lock"
flock -n 9 || {
    echo "another detached Router pipeline already holds ${STATE}/pipeline.lock"
    exit 1
}

log() {
    echo "[$(date -Iseconds)] $*" | tee -a "${STATE}/status.log"
}

cleanup() {
    # Remove only this experiment's transient servers. Data already written by
    # generate/replay is append-only and survives a process interruption.
    docker ps --format '{{.Names}}' \
        | grep -E '^dsv32-misa-assignment-router-(pilot-v2|zero-shot)-' \
        | xargs -r docker rm -f >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

log "pilot_v2: starting probe -> train -> eval"
MISA_ROUTER_EXP_NAME="${PILOT}" \
MISA_ROUTER_STAGES="${PILOT_STAGES:-probe,train,eval}" \
MISA_ROUTER_MAX_SAMPLES=8000 \
MISA_ROUTER_PROBE_SAMPLE_STRIDE=2 \
bash "${PIPELINE}" 2>&1 | tee -a "${STATE}/pilot_v2.log"
touch "${STATE}/PILOT_V2_COMPLETED"
log "pilot_v2: completed"

log "zero_shot: resuming trajectories only; unified prefill probe is pending"
MISA_ROUTER_EXP_NAME="${ZERO}" \
MISA_ROUTER_STAGES="${ZERO_STAGES:-trajectories}" \
MISA_ROUTER_MAX_SAMPLES=8000 \
MISA_ROUTER_MAX_RUNNING_REQUESTS=4 \
MISA_ROUTER_TRAJECTORY_CONCURRENCY=4 \
bash "${PIPELINE}" 2>&1 | tee -a "${STATE}/zero_shot.log"
if [[ ",${ZERO_STAGES:-trajectories}," == *,probe,* ]]; then
    touch "${STATE}/ZERO_SHOT_COMPLETED"
    touch "${STATE}/ALL_COMPLETED"
    log "zero_shot: full pipeline completed"
else
    touch "${STATE}/ZERO_SHOT_TRAJECTORIES_COMPLETED"
    log "zero_shot: trajectories completed; probe intentionally paused"
fi

# Durable launch command:
# setsid nohup bash scripts/run_router_experiments_detached.sh \
#   > /DATA/disk0/qyl/data/misa_router_detached/master.log 2>&1 < /dev/null &
