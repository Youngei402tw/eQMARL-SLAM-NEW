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


FRAMEWORKS = ("eqmarl_psi+", "qfctde", "fctde", "sctde")
DEFAULT_SEEDS = (3, 4, 5, 6, 7)
BOUNDED_POSE_DEFAULT_SEEDS = (8, 9, 10, 11, 12)
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
    if protocol not in {"faithful", "bounded_pose"}:
        raise ProtocolError(f"unsupported protocol: {protocol}")
    return f"active_slam_{protocol}_full"


def _expected_config(
    framework: str, seed: int, protocol: str = "faithful"
) -> dict[str, object]:
    critic_name = "eqmarl" if framework == "eqmarl_psi+" else framework
    prefix = _protocol_prefix(protocol)
    expected = {
        "experiment.roots.root_dir": (
            f"experiment_output/{prefix}_{framework}"
        ),
        "experiment.train.n_episodes": EXPECTED_EPISODES,
        "experiment.train.max_steps_per_episode": 250,
        "experiment.algorithm.init_func": "eqmarl.algorithms.MAA2C",
        "experiment.algorithm.init_params.gamma": 0.99,
        "experiment.algorithm.init_params.alpha": 0.01,
        "experiment.algorithm.init_params.alpha_final": 0.001,
        "experiment.algorithm.init_params.alpha_decay_episodes": 500,
        "experiment.algorithm.init_params.seed": seed,
        "experiment.algorithm.init_params.reward_aggregation": "mean",
        "experiment.algorithm.init_params.normalize_advantages": True,
        "experiment.algorithm.init_params.gradient_clip_norm": 1.0,
        "experiment.algorithm.init_params.episode_metrics_callback": (
            "eqmarl.environments.active_slam.episode_metrics_callback"
        ),
        "experiment.algorithm.init_params.env.func": (
            "eqmarl.environments.active_slam.active_slam_make"
        ),
        "experiment.algorithm.init_params.env.params.map_size": 24,
        "experiment.algorithm.init_params.env.params.n_agents": 2,
        "experiment.algorithm.init_params.env.params.n_beams": 36,
        "experiment.algorithm.init_params.env.params.patch_size": 7,
        "experiment.algorithm.init_params.env.params.time_limit": 250,
        "experiment.algorithm.init_params.env.params.seed": seed,
        "experiment.algorithm.init_params.model_actor.init_func": (
            "eqmarl.active_slam_models.generate_actor_classical"
        ),
        "experiment.algorithm.init_params.model_actor.init_params.observation_dim": 147,
        "experiment.algorithm.init_params.model_actor.init_params.n_actions": 3,
        "experiment.algorithm.init_params.model_actor.init_params.units": [100],
        "experiment.algorithm.init_params.model_actor.build_shape": [None, 147],
        "experiment.algorithm.init_params.optimizer_actor.params.learning_rate": 0.0001,
        "experiment.algorithm.init_params.model_critic.init_func": (
            f"eqmarl.active_slam_models.generate_critic_{critic_name}"
        ),
        "experiment.algorithm.init_params.model_critic.init_params.observation_dim": 147,
        "experiment.algorithm.init_params.model_critic.init_params.n_agents": 2,
        "experiment.algorithm.init_params.model_critic.init_params.n_actions": 3,
        "experiment.algorithm.init_params.model_critic.build_shape": [None, 2, 147],
    }
    if protocol == "bounded_pose":
        expected[
            "experiment.algorithm.init_params.env.params.bounded_slam_pose"
        ] = True
    critic = "experiment.algorithm.init_params.model_critic.init_params"
    optimizer = "experiment.algorithm.init_params.optimizer_critic"
    if framework in {"eqmarl_psi+", "qfctde"}:
        expected.update(
            {
                f"{critic}.d_qubits": 4,
                f"{critic}.n_layers": 5,
                f"{critic}.nn_activation": "linear",
                f"{critic}.trainable_w_enc": False,
                optimizer: [
                    {
                        "func": "tensorflow.keras.optimizers.Adam",
                        "params": {"learning_rate": rate},
                    }
                    for rate in (0.001, 0.001, 0.01, 0.01)
                ],
            }
        )
        if framework == "eqmarl_psi+":
            expected[f"{critic}.input_entanglement_type"] = "psi+"
    else:
        expected[f"{critic}.units"] = [100]
        expected[f"{optimizer}.params.learning_rate"] = 0.0001
    return expected


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
    missing = sorted(set(REQUIRED_METRICS) - set(metrics))
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
                for metric in REPORT_METRICS
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
            for metric in REPORT_METRICS
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
        for metric in REPORT_METRICS
    }
    return {
        "protocol": _protocol_prefix(protocol),
        "episodes": EXPECTED_EPISODES,
        "seeds": list(seeds),
        "frameworks": list(FRAMEWORKS),
        "windows": {name: list(bounds) for name, bounds in WINDOWS.items()},
        "window_summary": per_window,
        "stability_final_minus_pilot_end": stability,
        "final_eqmarl_minus_qfctde": eqmarl_minus_qfctde,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("experiment_output"))
    parser.add_argument(
        "--protocol", choices=("faithful", "bounded_pose"), default="faithful"
    )
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    seeds = args.seeds
    if seeds is None:
        seeds = (
            BOUNDED_POSE_DEFAULT_SEEDS
            if args.protocol == "bounded_pose"
            else DEFAULT_SEEDS
        )
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
