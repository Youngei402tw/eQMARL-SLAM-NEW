import gymnasium as gym
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from eqmarl.algorithms.maa2c import MAA2C


class TinyMultiAgentEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.MultiDiscrete([2, 2])
        entry = gym.spaces.Box(-1.0, 1.0, shape=(2, 3), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({"local": entry, "critic": entry})
        self.steps = 0

    def _observation(self):
        values = np.full((2, 3), self.steps / 2.0, dtype=np.float32)
        return {"local": values, "critic": values.copy()}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return self._observation(), {}

    def step(self, action):
        self.steps += 1
        truncated = self.steps >= 2
        return self._observation(), np.ones(2, dtype=np.float32), False, truncated, {}


def make_algorithm(alpha=0.01, **kwargs):
    actor = keras.Sequential(
        [keras.Input((3,)), keras.layers.Dense(2, activation="softmax")], name="actor"
    )
    critic = keras.Sequential(
        [keras.Input((2, 3)), keras.layers.Flatten(), keras.layers.Dense(1)], name="critic"
    )
    return MAA2C(
        TinyMultiAgentEnv(),
        actor,
        critic,
        keras.optimizers.Adam(1e-3),
        keras.optimizers.Adam(1e-3),
        alpha=alpha,
        seed=9,
        reward_aggregation="mean",
        **kwargs,
    )


def test_episode_preserves_truncation_and_updates_all_agents():
    algorithm = make_algorithm()
    reward, transitions, steps = algorithm.run_episode(0, 0, 5)
    assert steps == 2
    assert np.array_equal(reward, [2.0, 2.0])
    assert transitions[-1].truncated
    assert np.isfinite(list(algorithm.last_losses.values())).all()
    assert np.isclose(sum(
        algorithm.episode_diagnostics[f"action_{index}_fraction"] for index in range(2)
    ), 1.0)


def test_entropy_reduces_minimized_actor_objective():
    low_entropy = MAA2C._actor_objective(1.0, 0.1, 0.2)
    high_entropy = MAA2C._actor_objective(1.0, 0.9, 0.2)
    assert high_entropy < low_entropy


def test_td_target_is_detached_and_terminal_does_not_bootstrap():
    next_values = tf.Variable([[3.0], [4.0]])
    with tf.GradientTape() as tape:
        targets = MAA2C._td_targets(
            tf.constant([[1.0], [1.0]]),
            tf.constant([[0.0], [1.0]]),
            0.5,
            next_values,
        )
        total = tf.reduce_sum(targets)
    assert tape.gradient(total, next_values) is None
    assert np.allclose(targets.numpy(), [[2.5], [1.0]])


def test_advantage_normalization_and_entropy_decay():
    values = MAA2C._normalize_advantage_tensor(tf.constant([[1.0], [2.0], [3.0]]))
    assert np.isclose(tf.reduce_mean(values).numpy(), 0.0)
    assert np.isclose(tf.math.reduce_std(values).numpy(), 1.0)
    algorithm = make_algorithm(alpha=0.01, alpha_final=0.001, alpha_decay_episodes=100)
    assert np.isclose(algorithm._entropy_alpha(0), 0.01)
    assert np.isclose(algorithm._entropy_alpha(50), 0.0055)
    assert np.isclose(algorithm._entropy_alpha(100), 0.001)
