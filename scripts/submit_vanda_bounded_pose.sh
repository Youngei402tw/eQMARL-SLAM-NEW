#!/usr/bin/env bash
# Submit all four methods for the isolated bounded-pose protocol.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:-pilot}"

case "$mode" in
    pilot)
        episodes="${SLAM_N_EPISODES:-400}"
        read -r -a seeds <<< "${SLAM_BOUNDED_PILOT_SEEDS:-0 1 2}"
        ;;
    full)
        episodes="${SLAM_N_EPISODES:-1000}"
        read -r -a seeds <<< "${SLAM_BOUNDED_FULL_SEEDS:-8 9 10 11 12}"
        ;;
    *)
        echo "Usage: bash scripts/submit_vanda_bounded_pose.sh [pilot|full]" >&2
        exit 2
        ;;
esac
max_steps="${SLAM_MAX_STEPS:-250}"

for seed in "${seeds[@]}"; do
    if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
        echo "Seeds must be non-negative integers, got: $seed" >&2
        exit 2
    fi
    for framework in eqmarl_psi+ qfctde fctde sctde; do
        tag="${framework%%_*}"
        variables="SLAM_FRAMEWORK=$framework,SLAM_MODE=$mode,SLAM_SEED=$seed"
        variables+=",SLAM_N_EPISODES=$episodes,SLAM_MAX_STEPS=$max_steps"
        qsub -N "slam-bp-${tag}-s${seed}" -v "$variables" \
            "$repo_root/train_vanda_bounded_pose.pbs"
    done
done
