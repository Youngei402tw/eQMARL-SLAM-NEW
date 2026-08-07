"""Milestone-specific statistics for the Active-SLAM full-run audit."""

from __future__ import annotations

import numpy as np


def calculate_milestone_timing(
    runs, frameworks, seeds, metrics, start, stop, estimate
):
    """Summarize reach rate and 250-step-censored threshold time by seed."""
    timing = {}
    for framework in frameworks:
        timing[framework] = {}
        for metric in metrics:
            series_by_seed = {
                seed: runs[(framework, seed)].series[metric][start:stop]
                for seed in seeds
            }
            timing[framework][metric] = {
                "reach_rate": estimate(
                    {
                        seed: float(np.mean(values >= 0.0))
                        for seed, values in series_by_seed.items()
                    }
                ),
                "restricted_steps": estimate(
                    {
                        seed: float(np.mean(np.where(values >= 0.0, values, 251.0)))
                        for seed, values in series_by_seed.items()
                    }
                ),
            }
    return timing


def print_milestone_summary(
    report, frameworks, reward_metrics, step_metrics, format_estimate
):
    """Print reward decomposition and threshold timing for milestone99 runs."""
    print("\nFinal reward components")
    print("method\tcoverage\tuncertainty\tmilestone\tcollision\tstep")
    for framework in frameworks:
        summary = report["window_summary"]["final"][framework]
        values = tuple(format_estimate(summary[metric]) for metric in reward_metrics)
        print("\t".join((framework,) + values))

    print("\nFinal milestone reach rate / restricted steps")
    print("method\t90%\t95%\t98%\t99%")
    for framework in frameworks:
        timing = report["final_milestone_timing"][framework]
        values = []
        for metric in step_metrics:
            reach = timing[metric]["reach_rate"]["mean"]
            steps = timing[metric]["restricted_steps"]["mean"]
            values.append(f"{100.0 * reach:.1f}% / {steps:.1f}")
        print("\t".join((framework,) + tuple(values)))
