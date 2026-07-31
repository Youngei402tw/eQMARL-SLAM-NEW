"""Tests for the faithful full Active-SLAM result audit."""

import json

import pytest
import yaml

from scripts.analyze_active_slam_full import (
    EXPECTED_EPISODES,
    FRAMEWORKS,
    ProtocolError,
    analyze_faithful_full,
    load_faithful_full_runs,
)


REQUIRED_SERIES = (
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


def _config(framework, seed):
    critic = {
        "observation_dim": 147,
        "n_agents": 2,
        "n_actions": 3,
    }
    if framework in {"eqmarl_psi+", "qfctde"}:
        critic.update(
            d_qubits=4,
            n_layers=5,
            nn_activation="linear",
            trainable_w_enc=False,
        )
        if framework == "eqmarl_psi+":
            critic["input_entanglement_type"] = "psi+"
        critic_optimizer = [
            {
                "func": "tensorflow.keras.optimizers.Adam",
                "params": {"learning_rate": rate},
            }
            for rate in (0.001, 0.001, 0.01, 0.01)
        ]
    else:
        critic["units"] = [100]
        critic_optimizer = {
            "func": "tensorflow.keras.optimizers.Adam",
            "params": {"learning_rate": 0.0001},
        }
    return {
        "experiment": {
            "roots": {
                "root_dir": f"experiment_output/active_slam_faithful_full_{framework}"
            },
            "train": {
                "n_episodes": 1000,
                "max_steps_per_episode": 250,
                "callbacks": [
                    {
                        "func": "eqmarl.AlgorithmResultCheckpoint",
                        "params": {"save_freq": 100},
                    },
                    {
                        "func": "eqmarl.AlgorithmModelCheckpoint",
                        "params": {"save_freq": 100},
                    },
                    {
                        "func": "eqmarl.AlgorithmModelCheckpoint",
                        "params": {"save_freq": 100},
                    },
                ],
            },
            "algorithm": {
                "init_func": "eqmarl.algorithms.MAA2C",
                "init_params": {
                    "gamma": 0.99,
                    "alpha": 0.01,
                    "alpha_final": 0.001,
                    "alpha_decay_episodes": 500,
                    "seed": seed,
                    "reward_aggregation": "mean",
                    "normalize_advantages": True,
                    "gradient_clip_norm": 1.0,
                    "episode_metrics_callback": (
                        "eqmarl.environments.active_slam.episode_metrics_callback"
                    ),
                    "env": {
                        "func": "eqmarl.environments.active_slam.active_slam_make",
                        "params": {
                            "map_size": 24,
                            "n_agents": 2,
                            "n_beams": 36,
                            "patch_size": 7,
                            "time_limit": 250,
                            "seed": seed,
                        }
                    },
                    "model_actor": {
                        "init_func": "eqmarl.active_slam_models.generate_actor_classical",
                        "init_params": {
                            "observation_dim": 147,
                            "n_actions": 3,
                            "units": [100],
                        },
                        "build_shape": [None, 147],
                    },
                    "optimizer_actor": {"params": {"learning_rate": 0.0001}},
                    "model_critic": {
                        "init_func": (
                            "eqmarl.active_slam_models.generate_critic_eqmarl"
                            if framework == "eqmarl_psi+"
                            else f"eqmarl.active_slam_models.generate_critic_{framework}"
                        ),
                        "init_params": critic,
                        "build_shape": [None, 2, 147],
                    },
                    "optimizer_critic": critic_optimizer,
                }
            },
        }
    }


def _write_run(root, framework, seed, episodes=EXPECTED_EPISODES):
    session = root / f"active_slam_faithful_full_{framework}" / f"run-seed{seed}"
    session.mkdir(parents=True)
    (session / "config.yml").write_text(yaml.safe_dump(_config(framework, seed)))
    offset = FRAMEWORKS.index(framework) * 0.1 + seed * 0.01
    values = [offset + episode * 0.001 for episode in range(episodes)]
    metrics = {name: values for name in REQUIRED_SERIES}
    payload = {
        "reward": [[value, value] for value in values],
        "metrics": metrics,
    }
    (session / "metrics-0.json").write_text(json.dumps(payload))
    return session


def test_audit_computes_paired_full_horizon_statistics(tmp_path):
    for framework in FRAMEWORKS:
        for seed in (3, 4):
            _write_run(tmp_path, framework, seed)

    report = analyze_faithful_full(tmp_path, seeds=(3, 4))

    assert report["seeds"] == [3, 4]
    final = report["window_summary"]["final"]["eqmarl_psi+"]["coverage"]
    assert final["n"] == 2
    assert final["mean"] == pytest.approx(0.9845)
    stability = report["stability_final_minus_pilot_end"]["qfctde"]["coverage"]
    assert stability["mean"] == pytest.approx(0.6)
    comparison = report["final_eqmarl_minus_qfctde"]["coverage"]
    assert comparison["mean"] == pytest.approx(-0.1)


def test_audit_rejects_partial_checkpoint(tmp_path):
    for framework in FRAMEWORKS:
        for seed in (3, 4):
            episodes = 999 if framework == "sctde" and seed == 4 else EXPECTED_EPISODES
            _write_run(tmp_path, framework, seed, episodes=episodes)

    with pytest.raises(ProtocolError, match="reward has 999 episodes"):
        load_faithful_full_runs(tmp_path, seeds=(3, 4))


def test_audit_rejects_duplicate_seed(tmp_path):
    for framework in FRAMEWORKS:
        _write_run(tmp_path, framework, 3)
    duplicate = tmp_path / "active_slam_faithful_full_fctde" / "other-seed3"
    duplicate.mkdir()
    (duplicate / "config.yml").write_text(yaml.safe_dump(_config("fctde", 3)))
    source = tmp_path / "active_slam_faithful_full_fctde" / "run-seed3" / "metrics-0.json"
    (duplicate / "metrics-0.json").write_text(source.read_text())

    with pytest.raises(ProtocolError, match="duplicate faithful full run"):
        load_faithful_full_runs(tmp_path, seeds=(3,))


def test_audit_rejects_nonfinite_diagnostics(tmp_path):
    for framework in FRAMEWORKS:
        session = _write_run(tmp_path, framework, 3)
        if framework == "qfctde":
            metrics_path = session / "metrics-0.json"
            payload = json.loads(metrics_path.read_text())
            payload["metrics"]["critic_loss"][500] = float("nan")
            metrics_path.write_text(json.dumps(payload))

    with pytest.raises(ProtocolError, match="non-finite metrics.*critic_loss"):
        load_faithful_full_runs(tmp_path, seeds=(3,))
