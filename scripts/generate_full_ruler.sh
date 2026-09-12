#!/usr/bin/env bash
set -euo pipefail

GENERATOR=/DATA/disk0/qyl/data/RULER-main/scripts/data
MODEL=/DATA/disk0/qyl/models/deepseek-v3.2
OUT=/DATA/disk0/qyl/data/ruler_deepseek_v3_2_full20
SEED=20260823
SAMPLES=20
PARALLELISM="${PARALLELISM:-4}"
TASKS=(
    niah_single_1 niah_single_2 niah_single_3
    niah_multikey_1 niah_multikey_2 niah_multikey_3
    niah_multivalue niah_multiquery vt cwe fwe qa_1 qa_2
)
LENGTHS=(4096 8192 16384 32768 65536 131072)

generate_one() {
    local length="$1"
    local task="$2"
    local label="$((length / 1024))k"
    local file="${OUT}/${label}/${task}/validation.jsonl"
    if [[ -f "${file}" && "$(wc -l <"${file}")" -eq "${SAMPLES}" ]]; then
        echo "skip ${label}/${task}"
        return
    fi
    cd "${GENERATOR}"
    python3 prepare.py \
        --save_dir "${OUT}/${label}" \
        --benchmark synthetic \
        --task "${task}" \
        --subset validation \
        --tokenizer_path "${MODEL}" \
        --tokenizer_type hf \
        --max_seq_length "${length}" \
        --model_template_type base \
        --num_samples "${SAMPLES}" \
        --random_seed "${SEED}"
    [[ -f "${file}" && "$(wc -l <"${file}")" -eq "${SAMPLES}" ]]
    echo "done ${label}/${task}"
}
export -f generate_one
export GENERATOR MODEL OUT SEED SAMPLES

mkdir -p "${OUT}"
for length in "${LENGTHS[@]}"; do
    for task in "${TASKS[@]}"; do
        printf '%s %s\n' "${length}" "${task}"
    done
done | xargs -n2 -P"${PARALLELISM}" bash -c 'generate_one "$0" "$1"'

python3 - "${OUT}" "${SEED}" "${SAMPLES}" <<'PY'
import json
import sys
from pathlib import Path

out, seed, samples = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
payload = {
    "generator": "NVIDIA RULER",
    "generator_path": "/DATA/disk0/qyl/data/RULER-main",
    "tokenizer_path": "/DATA/disk0/qyl/models/deepseek-v3.2",
    "lengths": ["4k", "8k", "16k", "32k", "64k", "128k"],
    "tasks": 13,
    "samples_per_task": samples,
    "random_seed": seed,
    "total_records": 6 * 13 * samples,
}
(out / "generation_metadata.json").write_text(json.dumps(payload, indent=2) + "\n")
PY
echo "FULL_RULER_COMPLETED"
