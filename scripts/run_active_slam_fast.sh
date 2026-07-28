#!/usr/bin/env bash
# Fast active-SLAM iteration preset: rounds, episodes, and steps are overrideable.
set -euo pipefail

rounds="${1:-1}"
episodes="${2:-50}"
max_steps="${3:-50}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/run_active_slam.sh" "$rounds" "$episodes" "$max_steps" --fast
