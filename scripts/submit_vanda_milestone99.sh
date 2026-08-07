#!/usr/bin/env bash
# Submit all four methods for the isolated 99%-coverage milestone protocol.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:-pilot}"

milestone99_job_ids() {
    local state job_ids job_id job_name
    for state in Q R H; do
        job_ids="$(qselect -u "$USER" -s "$state" 2>/dev/null || true)"
        for job_id in $job_ids; do
            job_name="$(qstat -f "$job_id" 2>/dev/null \
                | awk -F ' = ' '$1 ~ /Job_Name/ {print $2; exit}' || true)"
            case "$job_name" in
                slam-m99-*) printf '%s\n' "$job_id" ;;
            esac
        done
    done
}

release_held_milestone99_jobs() {
    local held_jobs job_id job_name
    held_jobs="$(qselect -u "$USER" -s H 2>/dev/null || true)"
    for job_id in $held_jobs; do
        job_name="$(qstat -f "$job_id" 2>/dev/null \
            | awk -F ' = ' '$1 ~ /Job_Name/ {print $2; exit}')"
        case "$job_name" in
            slam-m99-*)
                if qrls -h u "$job_id"; then
                    echo "Released user hold on milestone99 job $job_id ($job_name)"
                else
                    echo "Could not release milestone99 job $job_id ($job_name);" \
                        "it may have a system/admin hold." >&2
                fi
                ;;
        esac
    done
}

monitor_milestone99_jobs() {
    local poll_seconds="${SLAM_HOLD_POLL_SECONDS:-60}"
    local job_ids job_count
    if [[ ! "$poll_seconds" =~ ^[1-9][0-9]*$ ]]; then
        echo "SLAM_HOLD_POLL_SECONDS must be a positive integer" >&2
        return 2
    fi
    while true; do
        release_held_milestone99_jobs
        job_ids="$(milestone99_job_ids)"
        if [[ -z "$job_ids" ]]; then
            echo "No queued, running, or held milestone99 jobs remain."
            return 0
        fi
        job_count="$(printf '%s\n' "$job_ids" | sort -u | sed '/^$/d' | wc -l)"
        echo "$(date): monitoring $job_count milestone99 jobs; polling again in ${poll_seconds}s"
        sleep "$poll_seconds"
    done
}

case "$mode" in
    monitor)
        monitor_milestone99_jobs
        exit $?
        ;;
    pilot)
        episodes="${SLAM_N_EPISODES:-400}"
        read -r -a seeds <<< "${SLAM_MILESTONE99_PILOT_SEEDS:-13000 14000 15000}"
        ;;
    full)
        episodes="${SLAM_N_EPISODES:-1000}"
        read -r -a seeds <<< "${SLAM_MILESTONE99_FULL_SEEDS:-16000 17000 18000 19000 20000}"
        ;;
    *)
        echo "Usage: bash scripts/submit_vanda_milestone99.sh [pilot|full|monitor]" >&2
        exit 2
        ;;
esac
max_steps="${SLAM_MAX_STEPS:-250}"
if [[ ! "$episodes" =~ ^[1-9][0-9]*$ ]]; then
    echo "SLAM_N_EPISODES must be a positive integer, got: $episodes" >&2
    exit 2
fi
if [[ ! "$max_steps" =~ ^[1-9][0-9]*$ ]]; then
    echo "SLAM_MAX_STEPS must be a positive integer, got: $max_steps" >&2
    exit 2
fi

for seed in "${seeds[@]}"; do
    if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
        echo "Seeds must be non-negative integers, got: $seed" >&2
        exit 2
    fi
done
for ((left = 0; left < ${#seeds[@]}; left++)); do
    for ((right = left + 1; right < ${#seeds[@]}; right++)); do
        difference=$((seeds[left] - seeds[right]))
        if ((difference < 0)); then
            difference=$((-difference))
        fi
        if ((difference < episodes)); then
            echo "Seed ranges overlap: ${seeds[left]} and ${seeds[right]} are less" \
                "than $episodes apart." >&2
            exit 2
        fi
    done
done

release_held_milestone99_jobs

for seed in "${seeds[@]}"; do
    for framework in eqmarl_psi+ qfctde fctde sctde; do
        tag="${framework%%_*}"
        variables="SLAM_FRAMEWORK=$framework,SLAM_MODE=$mode,SLAM_SEED=$seed"
        variables+=",SLAM_N_EPISODES=$episodes,SLAM_MAX_STEPS=$max_steps"
        qsub -N "slam-m99-${tag}-s${seed}" -v "$variables" \
            "$repo_root/train_vanda_milestone99.pbs"
    done
done

if [[ "${SLAM_MONITOR_HELD:-1}" == "1" ]]; then
    monitor_log="$repo_root/milestone99-monitor-${mode}-$(date +%Y%m%dT%H%M%S).log"
    script_path="$repo_root/scripts/$(basename "${BASH_SOURCE[0]}")"
    nohup bash "$script_path" monitor >"$monitor_log" 2>&1 < /dev/null &
    echo "Started held-job monitor (PID $!): $monitor_log"
else
    echo "Held-job monitor disabled (SLAM_MONITOR_HELD=$SLAM_MONITOR_HELD)"
fi
