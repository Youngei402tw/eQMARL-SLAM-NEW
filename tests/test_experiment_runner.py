import pytest

from scripts.experiment_runner import apply_train_overrides, set_round_seed


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
    config = {"experiment": {"train": {"n_episodes": 2000, "max_steps_per_episode": 250}}}
    apply_train_overrides(config, n_episodes=2, max_steps_per_episode=3)
    assert config["experiment"]["train"] == {
        "n_episodes": 2,
        "max_steps_per_episode": 3,
    }
    with pytest.raises(ValueError, match="positive"):
        apply_train_overrides(config, n_episodes=0)
