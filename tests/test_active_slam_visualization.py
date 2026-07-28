"""Regression coverage for active-SLAM plotting and metric readers."""

import json

import matplotlib.pyplot as plt
import numpy as np

from eqmarl.active_slam_visualization import (
    discover_metric_files,
    load_metrics,
    plot_learning_curves,
    plot_slam_frame,
    rollout,
    summarize_final_metrics,
)
from eqmarl.environments.active_slam import MultiAgentSLAMEnv
from eqmarl.policies import FrontierJointPolicy


def test_metric_loading_summary_and_plots(tmp_path):
    output = tmp_path / "active_slam_maa2c_eqmarl_psi+" / "run"
    output.mkdir(parents=True)
    (output / "metrics-0.json").write_text(
        json.dumps(
            {
                "reward": [[1.0, 1.0], [2.0, 2.0]],
                "metrics": {
                    "coverage": [0.2, 0.4],
                    "occupancy_iou": [0.1, 0.3],
                    "pose_rmse": [2.0, 1.0],
                    "success": [0.0, 1.0],
                },
            }
        )
    )
    data = load_metrics(discover_metric_files(tmp_path))
    assert data["team_reward"].tolist() == [1.0, 2.0]
    summary = summarize_final_metrics(data, window=1)
    assert set(summary["framework"]) == {"eqmarl_psi+"}
    figure = plot_learning_curves(data, smoothing=1)
    assert figure.axes
    plt.close(figure)


def test_frontier_rollout_can_be_rendered():
    env = MultiAgentSLAMEnv(map_size=16, time_limit=3)
    frames = rollout(env, FrontierJointPolicy(), seed=2)
    assert len(frames) >= 2
    assert frames[-1]["ground_truth"].shape == (16, 16)
    figure = plot_slam_frame(frames[-1], history=frames)
    assert len(figure.axes) == 2
    plt.close(figure)


def test_belief_panel_uses_black_for_occupied_cells():
    env = MultiAgentSLAMEnv(map_size=16, time_limit=1)
    frames = rollout(env, FrontierJointPolicy(), seed=5)
    frame = frames[-1]
    frame["observed"][:] = True
    frame["log_odds"][:] = 0.0
    frame["log_odds"][1, 1] = 8.0
    frame["log_odds"][1, 2] = -8.0
    figure = plot_slam_frame(frame)
    image = figure.axes[1].images[0].get_array()
    assert image[1, 1] < 0.01
    assert image[1, 2] > 0.99
    plt.close(figure)
