"""Plots and rollout animation helpers for active-SLAM experiments."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_METRICS = (
    "team_reward",
    "coverage",
    "occupancy_iou",
    "pose_rmse",
    "success",
    "collisions",
    "path_length",
    "pose_uncertainty",
    "redundant_coverage",
    "forward_action_fraction",
    "policy_entropy",
    "critic_loss",
    "actor_gradient_norm",
    "critic_gradient_norm",
    "coverage_reward",
    "uncertainty_reward",
    "milestone_reward",
    "collision_penalty",
    "step_penalty",
)


def discover_metric_files(root: str | Path = "experiment_output") -> dict[str, list[Path]]:
    """Find the newest compatible SLAM protocol without mixing result families."""
    root = Path(root)
    groups = {}
    protocols = (
        "active_slam_milestone99_full_",
        "active_slam_bounded_pose_full_",
        "active_slam_milestone99_pilot_",
        "active_slam_bounded_pose_pilot_",
        "active_slam_faithful_full_",
        "active_slam_faithful_pilot_",
        "active_slam_faithful_fast_",
        "active_slam_stable_full_",
        "active_slam_stable_pilot_",
        "active_slam_stable_",
        "active_slam_minigrid_",
    )
    filepaths = []
    for protocol in protocols:
        filepaths = sorted(root.glob(f"{protocol}*/*/metrics-*.json"))
        if filepaths:
            break
    for filepath in filepaths:
        framework = filepath.parents[1].name.removeprefix(protocol)
        groups.setdefault(framework, []).append(filepath)
    return groups


def load_metrics(groups: dict[str, list[Path]]) -> pd.DataFrame:
    """Convert saved training histories into one row per framework/run/episode."""
    records = []
    for framework, filepaths in groups.items():
        for run, filepath in enumerate(filepaths):
            with filepath.open() as metrics_file:
                payload = json.load(metrics_file)
            rewards = payload["reward"]
            metrics = payload["metrics"]
            for episode, reward in enumerate(rewards):
                record = {
                    "framework": framework,
                    "run": run,
                    "episode": episode,
                    "team_reward": float(np.mean(reward)),
                }
                record.update({key: float(values[episode]) for key, values in metrics.items()})
                records.append(record)
    if not records:
        raise FileNotFoundError("no active-SLAM metrics files were found")
    return pd.DataFrame.from_records(records)


def summarize_final_metrics(data: pd.DataFrame, window: int = 100) -> pd.DataFrame:
    """Compute mean and 95% confidence intervals over final per-run values."""
    metric_columns = [column for column in DEFAULT_METRICS if column in data]
    per_run = (
        data.sort_values("episode")
        .groupby(["framework", "run"], as_index=False)
        .tail(window)
        .groupby(["framework", "run"], as_index=False)[metric_columns]
        .mean()
    )
    rows = []
    for framework, group in per_run.groupby("framework"):
        for metric in metric_columns:
            values = group[metric].to_numpy()
            ci95 = 0.0 if len(values) < 2 else 1.96 * values.std(ddof=1) / np.sqrt(len(values))
            rows.append(
                {
                    "framework": framework,
                    "metric": metric,
                    "mean": values.mean(),
                    "ci95": ci95,
                }
            )
    return pd.DataFrame(rows)


def plot_learning_curves(
    data: pd.DataFrame,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    smoothing: int = 50,
):
    """Plot framework learning curves with 95% confidence bands across runs."""
    metrics = [metric for metric in metrics if metric in data]
    columns = 2
    rows = int(np.ceil(len(metrics) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(14, 3.6 * rows), squeeze=False)
    for axis, metric in zip(axes.flat, metrics):
        for framework, subset in data.groupby("framework"):
            aggregate = subset.groupby("episode")[metric].agg(["mean", "count", "std"])
            aggregate["mean"] = aggregate["mean"].rolling(smoothing, min_periods=1).mean()
            aggregate["ci95"] = (
                1.96 * aggregate["std"].fillna(0.0) / np.sqrt(aggregate["count"])
            ).rolling(smoothing, min_periods=1).mean()
            x = aggregate.index.to_numpy()
            axis.plot(x, aggregate["mean"], label=framework)
            axis.fill_between(
                x,
                aggregate["mean"] - aggregate["ci95"],
                aggregate["mean"] + aggregate["ci95"],
                alpha=0.18,
            )
        axis.set(title=metric.replace("_", " "), xlabel="episode")
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(metrics) :]:
        axis.set_visible(False)
    axes[0, 0].legend(title="framework")
    figure.tight_layout()
    return figure


def rollout(env, policy, seed: int = 10000, max_steps: int | None = None) -> list[dict]:
    """Run a policy and retain map, pose, and metric state for each frame."""
    observation, _ = env.reset(seed=seed)
    frames = [_snapshot(env, observation, 0.0)]
    max_steps = max_steps or env.time_limit
    terminated = truncated = False
    for _ in range(max_steps):
        action = policy.action(env, observation)
        observation, rewards, terminated, truncated, _ = env.step(action)
        frames.append(_snapshot(env, observation, float(np.mean(rewards))))
        if terminated or truncated:
            break
    return frames


def _snapshot(env, observation, reward: float) -> dict:
    beliefs = [backend.belief for backend in env.backends]
    log_odds, observed = env._fuse_beliefs(beliefs)
    return {
        "ground_truth": env.ground_truth.copy(),
        "log_odds": log_odds,
        "observed": observed,
        "frontiers": env._frontiers(log_odds, observed),
        "true_poses": env.true_poses.copy(),
        "estimated_poses": np.stack([belief.pose for belief in beliefs]),
        "metrics": env.episode_metrics(),
        "reward": reward,
        "observation": observation,
    }


def plot_slam_frame(frame: dict, history: list[dict] | None = None):
    """Plot ground truth beside the current fused occupancy-grid belief."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 5))
    _draw_frame(axes[0], frame, history, ground_truth=True)
    _draw_frame(axes[1], frame, history, ground_truth=False)
    figure.tight_layout()
    return figure


def _draw_frame(axis, frame: dict, history, ground_truth: bool):
    if ground_truth:
        image = np.where(frame["ground_truth"], 0.0, 1.0)
        axis.imshow(image, cmap="gray", vmin=0, vmax=1, origin="lower")
        axis.set_title("Ground truth and true poses")
    else:
        probability = 1.0 / (1.0 + np.exp(-frame["log_odds"]))
        image = np.where(frame["observed"], 1.0 - probability, 0.5)
        axis.imshow(image, cmap="gray", vmin=0, vmax=1, origin="lower")
        frontiers = np.argwhere(frame["frontiers"])
        if len(frontiers):
            axis.scatter(frontiers[:, 1], frontiers[:, 0], s=5, c="tab:red", label="frontier")
        axis.set_title("Fused occupancy belief and estimates")
    poses = frame["true_poses"] if ground_truth else frame["estimated_poses"]
    colors = ("tab:blue", "tab:orange")
    for index, pose in enumerate(poses):
        if history:
            path = np.asarray(
                [
                    item["true_poses" if ground_truth else "estimated_poses"][index, :2]
                    for item in history
                ]
            )
            axis.plot(path[:, 0], path[:, 1], color=colors[index], alpha=0.65)
        axis.quiver(
            pose[0], pose[1], np.cos(pose[2]), np.sin(pose[2]),
            color=colors[index], scale=14, width=0.008,
        )
    metrics = frame["metrics"]
    height, width = image.shape
    axis.set(
        xlabel="x cell",
        ylabel="y cell",
        xlim=(-0.5, width - 0.5),
        ylim=(-0.5, height - 0.5),
    )
    axis.set_aspect("equal", adjustable="box")
    axis.text(
        0.02, 0.02,
        f"coverage={metrics['coverage']:.2f}\nIoU={metrics['occupancy_iou']:.2f}\nRMSE={metrics['pose_rmse']:.2f}",
        transform=axis.transAxes, fontsize=9, va="bottom",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )


def animate_rollout(frames: list[dict], interval: int = 250):
    """Create a Matplotlib animation of a recorded active-SLAM rollout."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 5))

    def update(index):
        for axis in axes:
            axis.clear()
        history = frames[: index + 1]
        _draw_frame(axes[0], frames[index], history, ground_truth=True)
        _draw_frame(axes[1], frames[index], history, ground_truth=False)
        figure.suptitle(f"step {index} | reward {frames[index]['reward']:.3f}")
        figure.tight_layout()

    return animation.FuncAnimation(figure, update, frames=len(frames), interval=interval)


def animate_rollout_comparison(
    rollouts: dict[str, list[dict]], interval: int = 250, frame_stride: int = 1
):
    """Synchronize true-path and belief animations for multiple policies."""
    if not rollouts or any(not frames for frames in rollouts.values()):
        raise ValueError("comparison requires at least one non-empty rollout")
    if frame_stride < 1:
        raise ValueError("frame_stride must be at least one")

    columns = len(rollouts)
    figure, axes = plt.subplots(
        2, columns, figsize=(4.5 * columns, 8), squeeze=False
    )
    figure.subplots_adjust(
        left=0.04, right=0.99, bottom=0.07, top=0.90, wspace=0.30, hspace=0.35
    )
    figure_title = figure.suptitle("Same-map policy comparison | synchronized step 0")
    frame_count = max(len(frames) for frames in rollouts.values())
    frame_indices = list(range(0, frame_count, frame_stride))
    if frame_indices[-1] != frame_count - 1:
        frame_indices.append(frame_count - 1)

    def update(frame_number):
        index = frame_indices[frame_number]
        for column, (method, frames) in enumerate(rollouts.items()):
            frame_index = min(index, len(frames) - 1)
            history = frames[: frame_index + 1]
            for axis in axes[:, column]:
                axis.clear()
            _draw_frame(
                axes[0, column], frames[frame_index], history, ground_truth=True
            )
            _draw_frame(
                axes[1, column], frames[frame_index], history, ground_truth=False
            )
            status = (
                f"finished at step {frame_index}"
                if index >= len(frames)
                else f"step {frame_index}"
            )
            axes[0, column].set_title(f"{method}: true path\n{status}")
            axes[1, column].set_title(f"{method}: SLAM belief")
        figure_title.set_text(
            f"Same-map policy comparison | synchronized step {index}"
        )

    return animation.FuncAnimation(
        figure, update, frames=len(frame_indices), interval=interval
    )
