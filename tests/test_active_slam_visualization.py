"""Regression coverage for active-SLAM plotting and metric readers."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from eqmarl.active_slam_visualization import (
    animate_rollout_comparison,
    discover_metric_files,
    load_metrics,
    plot_learning_curves,
    plot_slam_frame,
    rollout,
    summarize_final_metrics,
)
from eqmarl.environments.active_slam import MultiAgentSLAMEnv
from eqmarl.policies import FrontierJointPolicy, LearnedJointPolicy


def test_notebook_learned_rollout_uses_bounded_slam_pose():
    notebook_path = Path(__file__).parents[1] / "experiments" / "active_slam_visualization.ipynb"
    notebook = json.loads(notebook_path.read_text())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "LearnedJointPolicy(actor, mode='sample', seed=rollout_seed)" in source
    assert "active_slam_milestone99_full_" in source
    assert "training_seed = audit_seeds[0]" in source
    assert "coverage_target = 0.99" in source
    assert "rollout_step_limit = 250" in source
    assert "importlib.reload(active_slam_visualization)" in source
    assert "animate_rollout_comparison(" in source
    assert "rollouts_by_method, interval=180, frame_stride=3" in source
    assert "time_limit=rollout_step_limit" in source
    assert "coverage_milestones=coverage_milestones" in source
    assert "analyze_active_slam_full" in source
    assert "'milestone99': (16000, 17000, 18000, 19000, 20000)" in source


def test_learned_policy_can_reproduce_sampled_training_actions():
    class ConstantActor:
        def __call__(self, observations, training=False):
            return np.tile([0.45, 0.45, 0.10], (len(observations), 1))

    observation = {"local": np.zeros((2, 147), dtype=np.float32)}
    first = LearnedJointPolicy(ConstantActor(), mode="sample", seed=7)
    second = LearnedJointPolicy(ConstantActor(), mode="sample", seed=7)
    first_actions = [first.action(None, observation) for _ in range(10)]
    second_actions = [second.action(None, observation) for _ in range(10)]

    assert first_actions == second_actions
    assert any(actions != [0, 0] for actions in first_actions)


def test_metric_loading_summary_and_plots(tmp_path):
    legacy = tmp_path / "active_slam_maa2c_eqmarl_psi+" / "old-run"
    legacy.mkdir(parents=True)
    (legacy / "metrics-0.json").write_text('{"legacy": true}')
    output = tmp_path / "active_slam_minigrid_eqmarl_psi+" / "run"
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


def test_faithful_metrics_take_precedence_over_previous_protocols(tmp_path):
    for protocol, reward in (
        ("minigrid", 1.0),
        ("stable_full", 2.0),
        ("faithful_pilot", 3.0),
    ):
        output = tmp_path / f"active_slam_{protocol}_fctde" / "run"
        output.mkdir(parents=True)
        (output / "metrics-0.json").write_text(json.dumps({
            "reward": [[reward, reward]], "metrics": {"coverage": [0.5]}
        }))
    groups = discover_metric_files(tmp_path)
    assert list(groups) == ["fctde"]
    assert load_metrics(groups)["team_reward"].tolist() == [3.0]


def test_faithful_full_metrics_take_precedence_over_faithful_pilot(tmp_path):
    for protocol, reward in (("faithful_pilot", 1.0), ("faithful_full", 2.0)):
        output = tmp_path / f"active_slam_{protocol}_fctde" / "run"
        output.mkdir(parents=True)
        (output / "metrics-0.json").write_text(json.dumps({
            "reward": [[reward, reward]], "metrics": {"coverage": [0.5]}
        }))
    groups = discover_metric_files(tmp_path)
    assert load_metrics(groups)["team_reward"].tolist() == [2.0]


def test_bounded_pose_metrics_take_precedence_after_reevaluation(tmp_path):
    for protocol, reward in (("faithful_full", 1.0), ("bounded_pose_full", 2.0)):
        output = tmp_path / f"active_slam_{protocol}_fctde" / "run"
        output.mkdir(parents=True)
        (output / "metrics-0.json").write_text(json.dumps({
            "reward": [[reward, reward]], "metrics": {"coverage": [0.5]}
        }))
    groups = discover_metric_files(tmp_path)
    assert load_metrics(groups)["team_reward"].tolist() == [2.0]


def test_milestone99_metrics_take_precedence_after_retraining(tmp_path):
    for protocol, reward in (("bounded_pose_full", 1.0), ("milestone99_full", 2.0)):
        output = tmp_path / f"active_slam_{protocol}_fctde" / "run"
        output.mkdir(parents=True)
        (output / "metrics-0.json").write_text(json.dumps({
            "reward": [[reward, reward]], "metrics": {"coverage": [0.5]}
        }))
    groups = discover_metric_files(tmp_path)
    assert load_metrics(groups)["team_reward"].tolist() == [2.0]


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


def test_comparison_animation_synchronizes_multiple_rollouts():
    short_env = MultiAgentSLAMEnv(map_size=16, time_limit=1)
    long_env = MultiAgentSLAMEnv(map_size=16, time_limit=2)
    rollouts = {
        "short": rollout(short_env, FrontierJointPolicy(), seed=5),
        "long": rollout(long_env, FrontierJointPolicy(), seed=5),
    }

    comparison = animate_rollout_comparison(rollouts, interval=10, frame_stride=2)
    comparison._func(0)
    initial_positions = [axis.get_position().bounds for axis in comparison._fig.axes]
    comparison._func(1)

    assert len(comparison._fig.axes) == 4
    assert "finished at step 1" in comparison._fig.axes[0].get_title()
    assert all(
        np.allclose(initial, axis.get_position().bounds)
        for initial, axis in zip(initial_positions, comparison._fig.axes)
    )
    assert comparison._fig.axes[0].get_xlim() == (-0.5, 15.5)
    assert comparison._fig.axes[0].get_ylim() == (-0.5, 15.5)
    plt.close(comparison._fig)
