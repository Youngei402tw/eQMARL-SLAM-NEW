import os
import pytest
from pathlib import Path

import eqmarl
import yaml

from scripts.experiment_runner import (
    apply_fast_preset,
    apply_output_protocol,
    apply_pilot_monitor,
    apply_seed_override,
    apply_train_overrides,
    set_round_seed,
)


def test_runner_enables_incremental_gpu_allocation_before_eqmarl_import():
    runner_path = Path(__file__).parents[1] / "scripts" / "experiment_runner.py"
    source = runner_path.read_text()
    setting = 'os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")'
    assert os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] == "true"
    assert source.index(setting) < source.index("import eqmarl")


def test_round_seeds_use_non_overlapping_ranges():
    config = {
        "experiment": {
            "algorithm": {
                "init_params": {"seed": 7, "env": {"params": {"seed": 7}}}
            }
        }
    }
    set_round_seed(config, 3)
    params = config["experiment"]["algorithm"]["init_params"]
    assert params["seed"] == 300007
    assert params["env"]["params"]["seed"] == 300007


def test_training_overrides_are_validated():
    config = {
        "experiment": {
            "train": {"n_episodes": 2000, "max_steps_per_episode": 250},
            "algorithm": {"init_params": {"env": {"params": {"time_limit": 250}}}},
        }
    }
    apply_train_overrides(config, n_episodes=2, max_steps_per_episode=3)
    assert config["experiment"]["train"] == {
        "n_episodes": 2,
        "max_steps_per_episode": 3,
    }
    with pytest.raises(ValueError, match="positive"):
        apply_train_overrides(config, n_episodes=0)


def test_seed_override_and_pilot_monitor_are_explicit():
    config = {
        "experiment": {
            "train": {},
            "algorithm": {"init_params": {"seed": 0, "env": {"params": {"seed": 0}}}},
        }
    }
    apply_seed_override(config, 12)
    apply_pilot_monitor(config)
    params = config["experiment"]["algorithm"]["init_params"]
    assert params["seed"] == params["env"]["params"]["seed"] == 12
    assert config["experiment"]["train"]["callbacks"][-1]["func"].endswith(
        "ActiveSLAMPilotMonitor"
    )


def test_output_protocol_rewrites_all_active_slam_paths():
    config = {"experiment": {
        "roots": {"root_dir": "experiment_output/active_slam_faithful_full_fctde"},
        "save": {"metrics_file": "active_slam_faithful_full_fctde/metrics.json"},
    }}
    apply_output_protocol(config, "pilot")
    assert "faithful_pilot" in config["experiment"]["roots"]["root_dir"]
    assert "faithful_pilot" in config["experiment"]["save"]["metrics_file"]


def test_fast_preset_reduces_active_slam_dimensions_and_budget():
    config = {
        "experiment": {
            "train": {"n_episodes": 2000, "max_steps_per_episode": 250},
            "algorithm": {"init_params": {
                "env": {"params": {}},
                "model_actor": {"init_params": {"observation_dim": 411, "n_layers": 5}, "build_shape": [None, 411]},
                "model_critic": {"init_params": {"observation_dim": 411, "units": [33]}, "build_shape": [None, 2, 411]},
            }},
        }
    }
    apply_fast_preset(config)
    params = config["experiment"]["algorithm"]["init_params"]
    assert config["experiment"]["train"] == {"n_episodes": 50, "max_steps_per_episode": 50}
    assert params["env"]["params"] == {"map_size": 16, "n_beams": 16, "patch_size": 7, "time_limit": 50}
    assert params["model_actor"]["init_params"]["n_layers"] == 2
    assert params["model_actor"]["build_shape"] == [None, 147]
    assert params["model_critic"]["init_params"]["units"] == [16]
    assert params["model_critic"]["build_shape"] == [None, 2, 147]


def test_active_slam_configs_follow_four_method_minigrid_protocol():
    experiment_dir = Path(__file__).parents[1] / "experiments"
    paths = sorted(experiment_dir.glob("active_slam_maa2c_*.yml"))
    assert [path.stem.removeprefix("active_slam_maa2c_") for path in paths] == [
        "eqmarl_psi+", "fctde", "qfctde", "sctde"
    ]
    for path in paths:
        config = yaml.load(path.read_text(), Loader=eqmarl.yaml.ConfigLoader)
        experiment = config["experiment"]
        params = experiment["algorithm"]["init_params"]
        actor = params["model_actor"]
        critic = params["model_critic"]
        assert experiment["roots"]["root_dir"].startswith(
            "experiment_output/active_slam_faithful_full_"
        )
        assert experiment["train"]["n_episodes"] == 1000
        assert experiment["train"]["max_steps_per_episode"] == 250
        assert [item["params"]["save_freq"] for item in experiment["train"]["callbacks"]] == [100, 100, 100]
        assert params["gamma"] == 0.99
        assert params["reward_aggregation"] == "mean"
        assert params["normalize_advantages"] is True
        assert params["gradient_clip_norm"] == 1.0
        assert params["alpha"] == 0.01
        assert params["alpha_final"] == 0.001
        assert params["alpha_decay_episodes"] == 500
        assert params["env"]["params"] == {
            "map_size": 24,
            "n_agents": 2,
            "n_beams": 36,
            "patch_size": 7,
            "time_limit": 250,
            "seed": 0,
        }
        assert actor["init_func"] == "eqmarl.active_slam_models.generate_actor_classical"
        assert actor["init_params"]["observation_dim"] == 147
        assert actor["init_params"]["n_actions"] == 3
        assert actor["init_params"]["units"] == [100]
        assert actor["build_shape"] == [None, 147]
        assert critic["init_params"]["observation_dim"] == 147
        assert critic["init_params"]["observation_dim"] == actor["init_params"]["observation_dim"]
        assert critic["init_params"]["n_actions"] == 3
        assert critic["build_shape"] == [None, 2, 147]
        assert params["optimizer_actor"]["params"]["learning_rate"] == 0.0001
        assert any(item["name"] == actor["init_params"]["name"] for item in experiment["save"]["model_files"])
        assert any(item["name"] == critic["init_params"]["name"] for item in experiment["save"]["model_files"])
        framework = path.stem.removeprefix("active_slam_maa2c_")
        critic_optimizer = params["optimizer_critic"]
        if framework in {"eqmarl_psi+", "qfctde"}:
            assert critic["init_params"]["d_qubits"] == 4
            assert critic["init_params"]["n_layers"] == 5
            assert critic["init_params"]["nn_activation"] == "linear"
            assert critic["init_params"]["trainable_w_enc"] is False
            assert [item["params"]["learning_rate"] for item in critic_optimizer] == [
                0.001, 0.001, 0.01, 0.01
            ]
            if framework == "eqmarl_psi+":
                assert critic["init_params"]["input_entanglement_type"] == "psi+"
        else:
            assert critic["init_params"]["units"] == [100]
            assert critic_optimizer["params"]["learning_rate"] == 0.0001
