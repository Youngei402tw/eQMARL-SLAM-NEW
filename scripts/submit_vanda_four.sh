#!/usr/bin/env bash
# Submit one one-seed PBS job for each active-SLAM framework.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:-fast}"
if [[ "$mode" != "fast" && "$mode" != "full" ]]; then
    echo "Usage: bash scripts/submit_vanda_four.sh [fast|full]" >&2
    exit 2
fi

for framework in eqmarl_psi+ qfctde fctde sctde; do
    variables="SLAM_FRAMEWORK=$framework,SLAM_MODE=$mode"
    for name in SLAM_N_EPISODES SLAM_MAX_STEPS; do
        value="${!name-}"
        if [[ -n "$value" ]]; then
            variables+=",$name=$value"
        fi
    done
    qsub -v "$variables" "$repo_root/train_vanda.pbs"
done
