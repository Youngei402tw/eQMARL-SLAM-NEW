"""Audit and summarize a 1,000-episode Active-SLAM comparison."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Iterable

import numpy as np
from scipy.stats import t as student_t
import yaml

try:
    from .active_slam_analysis_support import (
        calculate_milestone_timing,
        print_milestone_summary,
    )
    from .active_slam_protocol import expected_config
except ImportError:
    from active_slam_analysis_support import (
        calculate_milestone_timing,
        print_milestone_summary,
    )
    from active_slam_protocol import expected_config


FRAMEWORKS = ("eqmarl_psi+", "qfctde", "fctde", "sctde")
DEFAULT_SEEDS = (3, 4, 5, 6, 7)
BOUNDED_POSE_DEFAULT_SEEDS = (8, 9, 10, 11, 12)
MILESTONE99_DEFAULT_SEEDS = (16000, 17000, 18000, 19000, 20000)
EXPECTED_EPISODES = 1000
WINDOWS = {
    "pilot_end": (300, 400),
    "post_decay": (500, 600),
    "final": (900, 1000),
}
REQUIRED_METRICS = (
    "coverage",
    "success",
    "forward_action_fraction",
    "left_action_fraction",
    "right_action_fraction",
    "policy_entropy",
    "actor_loss",
    "critic_loss",
    "actor_gradient_norm",
    "critic_gradient_norm",
    "episode_steps",
    "occupancy_iou",
    "pose_rmse",
)
REPORT_METRICS = ("team_reward",) + REQUIRED_METRICS
MILESTONE_REWARD_METRICS = (
    "coverage_reward",
    "uncertainty_reward",
    "milestone_reward",
    "collision_penalty",
    "step_penalty",
)
MILESTONE_STEP_METRICS = (
    "steps_to_coverage_90",
    "steps_to_coverage_95",
    "steps_to_coverage_98",
    "steps_to_coverage_99",
)


class ProtocolError(ValueError):
    """Raised when saved results do not match the faithful full protocol."""


@dataclass(frozen=True)
class FullRun:
    framework: str
    seed: int
    session_dir: Path
    series: dict[str, np.ndarray]


def _get(config: dict, dotted_path: str):
    value = config
    for key in dotted_path.split("."):
        value = value[key]
    return value


def _protocol_prefix(protocol: str) -> str:
    if protocol not in {"faithful", "bounded_pose", "milestone99"}:
        raise ProtocolError(f"unsupported protocol: {protocol}")
    return f"active_slam_{protocol}_full"


def _expected_config(
    framework: str, seed: int, protocol: str = "faithful"
) -> dict[str, object]:
    return expected_config(framework, seed, protocol, _protocol_prefix(protocol))


def _validate_config(
    config: dict,
    framework: str,
    seed: int,
    path: Path,
    protocol: str = "faithful",
) -> None:
    violations = []
    for dotted_path, expected in _expected_config(framework, seed, protocol).items():
        try:
            actual = _get(config, dotted_path)
        except (KeyError, TypeError):
            violations.append(f"missing {dotted_path}")
            continue
        if actual != expected:
            violations.append(f"{dotted_path}={actual!r}, expected {expected!r}")
    try:
        callbacks = _get(config, "experiment.train.callbacks")
        callback_funcs = [callback["func"] for callback in callbacks]
        save_frequencies = [callback["params"]["save_freq"] for callback in callbacks]
        expected_callbacks = [
            "eqmarl.AlgorithmResultCheckpoint",
            "eqmarl.AlgorithmModelCheckpoint",
            "eqmarl.AlgorithmModelCheckpoint",
        ]
        if callback_funcs != expected_callbacks or save_frequencies != [100, 100, 100]:
            violations.append(
                "callbacks must be one result and two model checkpoints at frequency 100"
            )
    except (KeyError, TypeError):
        violations.append("missing faithful checkpoint callbacks")
    if violations:
        raise ProtocolError(f"{path}: " + "; ".join(violations))


def _load_run(
    framework: str, session_dir: Path, protocol: str = "faithful"
) -> FullRun:
    match = re.search(r"-seed([0-9]+)$", session_dir.name)
    if not match:
        raise ProtocolError(f"invalid full-run directory name: {session_dir}")
    seed = int(match.group(1))
    config_path = session_dir / "config.yml"
    metrics_paths = sorted(session_dir.glob("metrics-*.json"))
    if not config_path.is_file():
        raise ProtocolError(f"missing resolved config: {config_path}")
    if len(metrics_paths) != 1:
        raise ProtocolError(
            f"{session_dir}: expected one metrics file, found {len(metrics_paths)}"
        )

    with config_path.open() as config_file:
        config = yaml.safe_load(config_file)
    _validate_config(config, framework, seed, config_path, protocol)

    with metrics_paths[0].open() as metrics_file:
        payload = json.load(metrics_file)
    reward = np.asarray(payload.get("reward", []), dtype=float)
    if reward.ndim == 0 or reward.shape[0] != EXPECTED_EPISODES:
        episode_count = 0 if reward.ndim == 0 else reward.shape[0]
        raise ProtocolError(
            f"{metrics_paths[0]}: reward has {episode_count} episodes, "
            f"expected {EXPECTED_EPISODES}"
        )
    if not np.isfinite(reward).all():
        raise ProtocolError(f"{metrics_paths[0]}: reward contains NaN or infinity")
    metrics = payload.get("metrics", {})
    required_metrics = set(REQUIRED_METRICS)
    if protocol == "milestone99":
        required_metrics.update(MILESTONE_REWARD_METRICS)
        required_metrics.update(MILESTONE_STEP_METRICS)
    missing = sorted(required_metrics - set(metrics))
    if missing:
        raise ProtocolError(f"{metrics_paths[0]}: missing metrics {missing}")

    series = {
        name: np.asarray(values, dtype=float) for name, values in metrics.items()
    }
    bad_shapes = {
        name: list(values.shape)
        for name, values in series.items()
        if values.shape != (EXPECTED_EPISODES,)
    }
    if bad_shapes:
        raise ProtocolError(
            f"{metrics_paths[0]}: metric shapes do not match: {bad_shapes}"
        )
    nonfinite = sorted(
        name for name, values in series.items() if not np.isfinite(values).all()
    )
    if nonfinite:
        raise ProtocolError(f"{metrics_paths[0]}: non-finite metrics {nonfinite}")
    series["team_reward"] = reward.mean(axis=1) if reward.ndim > 1 else reward
    return FullRun(framework, seed, session_dir, series)


def load_faithful_full_runs(
    root: Path | str = "experiment_output",
    seeds: Iterable[int] = DEFAULT_SEEDS,
    protocol: str = "faithful",
) -> dict[tuple[str, int], FullRun]:
    """Load exactly one complete full run per method and expected seed."""
    root = Path(root)
    prefix = _protocol_prefix(protocol)
    protocol_label = protocol.replace("_", " ")
    expected_seeds = tuple(int(seed) for seed in seeds)
    if len(set(expected_seeds)) != len(expected_seeds) or any(
        seed < 0 for seed in expected_seeds
    ):
        raise ProtocolError(f"seeds must be unique non-negative integers: {expected_seeds}")
    if protocol == "milestone99":
        ordered = sorted(expected_seeds)
        overlapping = [
            (left, right)
            for left, right in zip(ordered, ordered[1:])
            if right - left < EXPECTED_EPISODES
        ]
        if overlapping:
            raise ProtocolError(
                f"milestone99 seed map ranges overlap: {overlapping}; "
                f"seeds must be at least {EXPECTED_EPISODES} apart"
            )

    runs = {}
    for framework in FRAMEWORKS:
        framework_root = root / f"{prefix}_{framework}"
        if not framework_root.is_dir():
            raise ProtocolError(
                f"missing {protocol_label} full directory: {framework_root}"
            )
        for session_dir in sorted(path for path in framework_root.iterdir() if path.is_dir()):
            run = _load_run(framework, session_dir, protocol)
            if run.seed not in expected_seeds:
                raise ProtocolError(
                    f"{session_dir}: unexpected seed {run.seed}; expected {expected_seeds}"
                )
            key = (framework, run.seed)
            if key in runs:
                raise ProtocolError(
                    f"duplicate {protocol_label} full run for {framework} seed {run.seed}: "
                    f"{runs[key].session_dir} and {session_dir}"
                )
            runs[key] = run

    missing = [
        f"{framework}/seed{seed}"
        for framework in FRAMEWORKS
        for seed in expected_seeds
        if (framework, seed) not in runs
    ]
    if missing:
        raise ProtocolError(
            f"missing {protocol_label} full runs: {', '.join(missing)}"
        )
    return runs


def _estimate(values_by_seed: dict[int, float]) -> dict:
    values = np.asarray(list(values_by_seed.values()), dtype=float)
    mean = float(values.mean())
    if len(values) > 1:
        std = float(values.std(ddof=1))
        ci95 = float(student_t.ppf(0.975, len(values) - 1) * std / np.sqrt(len(values)))
    else:
        std = ci95 = 0.0
    return {
        "n": len(values),
        "mean": mean,
        "std": std,
        "ci95": ci95,
        "per_seed": {str(seed): value for seed, value in values_by_seed.items()},
    }


def analyze_active_slam_full(
    root: Path | str = "experiment_output",
    seeds: Iterable[int] = DEFAULT_SEEDS,
    protocol: str = "faithful",
) -> dict:
    """Audit full runs and calculate paired window and framework statistics."""
    seeds = tuple(int(seed) for seed in seeds)
    runs = load_faithful_full_runs(root, seeds, protocol)
    report_metrics = REPORT_METRICS
    if protocol == "milestone99":
        report_metrics += MILESTONE_REWARD_METRICS
    per_window = {}
    for window, (start, stop) in WINDOWS.items():
        per_window[window] = {
            framework: {
                metric: _estimate(
                    {
                        seed: float(runs[(framework, seed)].series[metric][start:stop].mean())
                        for seed in seeds
                    }
                )
                for metric in report_metrics
            }
            for framework in FRAMEWORKS
        }

    stability = {
        framework: {
            metric: _estimate(
                {
                    seed: (
                        per_window["final"][framework][metric]["per_seed"][str(seed)]
                        - per_window["pilot_end"][framework][metric]["per_seed"][str(seed)]
                    )
                    for seed in seeds
                }
            )
            for metric in report_metrics
        }
        for framework in FRAMEWORKS
    }
    eqmarl_minus_qfctde = {
        metric: _estimate(
            {
                seed: (
                    per_window["final"]["eqmarl_psi+"][metric]["per_seed"][str(seed)]
                    - per_window["final"]["qfctde"][metric]["per_seed"][str(seed)]
                )
                for seed in seeds
            }
        )
        for metric in report_metrics
    }
    milestone_timing = None
    if protocol == "milestone99":
        start, stop = WINDOWS["final"]
        milestone_timing = calculate_milestone_timing(
            runs,
            FRAMEWORKS,
            seeds,
            MILESTONE_STEP_METRICS,
            start,
            stop,
            _estimate,
        )
    return {
        "protocol": _protocol_prefix(protocol),
        "episodes": EXPECTED_EPISODES,
        "seeds": list(seeds),
        "frameworks": list(FRAMEWORKS),
        "windows": {name: list(bounds) for name, bounds in WINDOWS.items()},
        "window_summary": per_window,
        "stability_final_minus_pilot_end": stability,
        "final_eqmarl_minus_qfctde": eqmarl_minus_qfctde,
        "final_milestone_timing": milestone_timing,
    }


def analyze_faithful_full(
    root: Path | str = "experiment_output",
    seeds: Iterable[int] = DEFAULT_SEEDS,
    protocol: str = "faithful",
) -> dict:
    """Backward-compatible alias for :func:`analyze_active_slam_full`."""
    return analyze_active_slam_full(root, seeds, protocol)


def _format_estimate(estimate: dict, scale: float = 1.0) -> str:
    return f"{scale * estimate['mean']:.4f} +/- {scale * estimate['ci95']:.4f}"


def print_summary(report: dict) -> None:
    print(
        f"Validated {len(report['frameworks']) * len(report['seeds'])} "
        f"{report['protocol']} runs "
        f"for seeds {report['seeds']}."
    )
    print("\nFinal 100 episodes (mean +/- 95% t-CI across seeds)")
    print("method\tcoverage\tsuccess_pct\tforward\tIoU\tpose_RMSE")
    for framework in FRAMEWORKS:
        summary = report["window_summary"]["final"][framework]
        values = (
            _format_estimate(summary["coverage"]),
            _format_estimate(summary["success"], 100.0),
            _format_estimate(summary["forward_action_fraction"]),
            _format_estimate(summary["occupancy_iou"]),
            _format_estimate(summary["pose_rmse"]),
        )
        print("\t".join((framework,) + values))

    print("\nFinal minus episodes 300-399 (paired by seed)")
    print("method\tcoverage\tsuccess_pct\tforward\tentropy")
    for framework in FRAMEWORKS:
        summary = report["stability_final_minus_pilot_end"][framework]
        values = (
            _format_estimate(summary["coverage"]),
            _format_estimate(summary["success"], 100.0),
            _format_estimate(summary["forward_action_fraction"]),
            _format_estimate(summary["policy_entropy"]),
        )
        print("\t".join((framework,) + values))

    print("\nFinal eQMARL minus qfCTDE (paired by seed)")
    print("metric\tdifference")
    comparison = report["final_eqmarl_minus_qfctde"]
    for metric, scale in (
        ("coverage", 1.0),
        ("success", 100.0),
        ("team_reward", 1.0),
        ("occupancy_iou", 1.0),
        ("pose_rmse", 1.0),
    ):
        print(f"{metric}\t{_format_estimate(comparison[metric], scale)}")

    if report["protocol"] == "active_slam_milestone99_full":
        print_milestone_summary(
            report,
            FRAMEWORKS,
            MILESTONE_REWARD_METRICS,
            MILESTONE_STEP_METRICS,
            _format_estimate,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("experiment_output"))
    parser.add_argument(
        "--protocol",
        choices=("faithful", "bounded_pose", "milestone99"),
        default="faithful",
    )
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    seeds = args.seeds
    if seeds is None:
        default_seeds = {
            "faithful": DEFAULT_SEEDS,
            "bounded_pose": BOUNDED_POSE_DEFAULT_SEEDS,
            "milestone99": MILESTONE99_DEFAULT_SEEDS,
        }
        seeds = default_seeds[args.protocol]
    try:
        report = analyze_active_slam_full(args.root, seeds, args.protocol)
    except ProtocolError as error:
        parser.error(str(error))
    print_summary(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
