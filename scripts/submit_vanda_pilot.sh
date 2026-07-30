#!/usr/bin/env bash
# Submit four frameworks times three seeds as independent PBS pilot jobs.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
episodes="${SLAM_N_EPISODES:-400}"
max_steps="${SLAM_MAX_STEPS:-250}"
read -r -a seeds <<< "${SLAM_PILOT_SEEDS:-0 1 2}"

for seed in "${seeds[@]}"; do
    if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
        echo "Pilot seeds must be non-negative integers, got: $seed" >&2
        exit 2
    fi
    for framework in eqmarl_psi+ qfctde fctde sctde; do
        tag="${framework%%_*}"
        variables="SLAM_FRAMEWORK=$framework,SLAM_MODE=pilot,SLAM_SEED=$seed"
        variables+=",SLAM_N_EPISODES=$episodes,SLAM_MAX_STEPS=$max_steps"
        qsub -N "slam-${tag}-s${seed}" -v "$variables" "$repo_root/train_vanda.pbs"
    done
done
