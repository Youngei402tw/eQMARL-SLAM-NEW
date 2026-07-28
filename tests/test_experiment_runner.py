import pytest

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
