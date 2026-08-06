#!/usr/bin/env bash
# Submit all four methods for the isolated bounded-pose protocol.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:-pilot}"

bounded_job_ids() {
    local state job_ids job_id job_name
    for state in Q R H; do
        job_ids="$(qselect -u "$USER" -s "$state" 2>/dev/null || true)"
        for job_id in $job_ids; do
            job_name="$(qstat -f "$job_id" 2>/dev/null \
                | awk -F ' = ' '$1 ~ /Job_Name/ {print $2; exit}' || true)"
            case "$job_name" in
                slam-bp-*) printf '%s\n' "$job_id" ;;
            esac
        done
    done
}

release_held_bounded_jobs() {
    local held_jobs job_id job_name
    held_jobs="$(qselect -u "$USER" -s H 2>/dev/null || true)"
    for job_id in $held_jobs; do
        job_name="$(qstat -f "$job_id" 2>/dev/null \
            | awk -F ' = ' '$1 ~ /Job_Name/ {print $2; exit}')"
        case "$job_name" in
            slam-bp-*)
                if qrls -h u "$job_id"; then
                    echo "Released user hold on bounded-pose job $job_id ($job_name)"
                else
                    echo "Could not release bounded-pose job $job_id ($job_name);" \
                        "it may have a system/admin hold." >&2
                fi
                ;;
        esac
    done
}

monitor_bounded_jobs() {
    local poll_seconds="${SLAM_HOLD_POLL_SECONDS:-60}"
    local job_ids job_count
    if [[ ! "$poll_seconds" =~ ^[1-9][0-9]*$ ]]; then
        echo "SLAM_HOLD_POLL_SECONDS must be a positive integer" >&2
        return 2
    fi
    while true; do
        release_held_bounded_jobs
        job_ids="$(bounded_job_ids)"
        if [[ -z "$job_ids" ]]; then
            echo "No queued, running, or held bounded-pose jobs remain."
            return 0
        fi
        job_count="$(printf '%s\n' "$job_ids" | sort -u | sed '/^$/d' | wc -l)"
        echo "$(date): monitoring $job_count bounded-pose jobs; polling again in ${poll_seconds}s"
        sleep "$poll_seconds"
    done
}

case "$mode" in
    monitor)
        monitor_bounded_jobs
        exit $?
        ;;
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

release_held_bounded_jobs

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

if [[ "${SLAM_MONITOR_HELD:-1}" == "1" ]]; then
    monitor_log="$repo_root/bounded-pose-monitor-${mode}-$(date +%Y%m%dT%H%M%S).log"
    script_path="$repo_root/scripts/$(basename "${BASH_SOURCE[0]}")"
    nohup bash "$script_path" monitor >"$monitor_log" 2>&1 < /dev/null &
    echo "Started held-job monitor (PID $!): $monitor_log"
else
    echo "Held-job monitor disabled (SLAM_MONITOR_HELD=$SLAM_MONITOR_HELD)"
fi
