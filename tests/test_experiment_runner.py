import pytest
from pathlib import Path

import eqmarl
import yaml

from scripts.experiment_runner import apply_fast_preset, apply_train_overrides, set_round_seed


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
    assert params["model_actor"]["build_shape"] == [None, 175]
    assert params["model_critic"]["init_params"]["units"] == [16]


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
        assert actor["init_func"] == "eqmarl.active_slam_models.generate_actor_classical"
        assert actor["init_params"]["units"] == [100]
        assert params["optimizer_actor"]["params"]["learning_rate"] == 0.0001
        assert any(item["name"] == actor["init_params"]["name"] for item in experiment["save"]["model_files"])
        framework = path.stem.removeprefix("active_slam_maa2c_")
        critic_optimizer = params["optimizer_critic"]
        if framework in {"eqmarl_psi+", "qfctde"}:
            assert [item["params"]["learning_rate"] for item in critic_optimizer] == [
                0.001, 0.001, 0.01, 0.1
            ]
        else:
            assert critic_optimizer["params"]["learning_rate"] == 0.0001
