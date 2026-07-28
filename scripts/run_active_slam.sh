#!/usr/bin/env bash
set -euo pipefail

rounds="${1:-10}"
episodes="${2:-}"
max_steps="${3:-}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"

python_command=(python)
if ! python -c "import tensorflow_quantum, cirq, gymnasium" >/dev/null 2>&1; then
    if command -v conda >/dev/null 2>&1 \
        && conda run -n eQMARL_SLAM python -c "import tensorflow_quantum, cirq, gymnasium" >/dev/null 2>&1; then
        python_command=(conda run -n eQMARL_SLAM python)
    else
        echo "No compatible Python environment found. Activate eQMARL_SLAM or build Dockerfile.slam." >&2
        exit 1
    fi
fi

overrides=()
if [[ -n "$episodes" ]]; then
    overrides+=(--n-episodes "$episodes")
fi
if [[ -n "$max_steps" ]]; then
    overrides+=(--max-steps-per-episode "$max_steps")
fi

echo "Using: ${python_command[*]}"
"${python_command[@]}" scripts/experiment_runner.py experiments/active_slam_maa2c_eqmarl_psi+.yml -r "$rounds" "${overrides[@]}"
"${python_command[@]}" scripts/experiment_runner.py experiments/active_slam_maa2c_qfctde.yml -r "$rounds" "${overrides[@]}"
"${python_command[@]}" scripts/experiment_runner.py experiments/active_slam_maa2c_fctde.yml -r "$rounds" "${overrides[@]}"
"${python_command[@]}" scripts/experiment_runner.py experiments/active_slam_maa2c_sctde.yml -r "$rounds" "${overrides[@]}"
