import random
from typing import Union

import gymnasium as gym
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from .algorithm import MultiAgentInteraction, VectorAlgorithm


class MAA2C(VectorAlgorithm):
    """Shared-policy multi-agent advantage actor-critic with a joint critic.

    New environments should be normal ``gym.Env`` instances with a
    ``MultiDiscrete`` action space. Their observation may either be an array of
    per-agent observations or a dictionary containing ``local`` and ``critic``
    arrays. ``gym.vector.VectorEnv`` remains supported for the paper's legacy
    experiments, where its vector axis represented agents.
    """

    def __init__(
        self,
        env: Union[gym.Env, gym.vector.VectorEnv],
        model_actor: keras.Model,
        model_critic: keras.Model,
        optimizer_actor: Union[keras.optimizers.Optimizer, list[keras.optimizers.Optimizer]],
        optimizer_critic: Union[keras.optimizers.Optimizer, list[keras.optimizers.Optimizer]],
        gamma: float = 1.0,
        alpha: float = 0.001,
        episode_metrics_callback=None,
        seed: int = 0,
        reward_aggregation: str = "sum",
    ):
        if isinstance(env, gym.vector.VectorEnv):
            super().__init__(env, episode_metrics_callback)
            self.legacy_vector_env = True
            self.n_agents = env.num_envs
        elif isinstance(env, gym.Env):
            if not isinstance(env.action_space, gym.spaces.MultiDiscrete):
                raise TypeError("MAA2C requires a MultiDiscrete joint action space")
            self.env = env
            self.episode_metrics_callback = episode_metrics_callback
            self.n_agents = len(env.action_space.nvec)
            self.n_envs = self.n_agents  # Backward-compatible public attribute.
            self._episode_reward_history = []
            self._episode_metrics_history = []
            self._models = {}
            self.legacy_vector_env = False
        else:
            raise TypeError("env must be a gymnasium Env or VectorEnv")

        if not isinstance(env.action_space, gym.spaces.MultiDiscrete):
            raise TypeError("MAA2C requires a MultiDiscrete joint action space")
        if len(set(int(n) for n in env.action_space.nvec)) != 1:
            raise ValueError("the shared actor requires the same action count for every agent")

        self.models = {model_actor.name: model_actor, model_critic.name: model_critic}
        self.model_actor = model_actor
        self.model_critic = model_critic
        self.optimizer_actor = optimizer_actor
        self.optimizer_critic = optimizer_critic
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.seed = int(seed)
        if reward_aggregation not in ("sum", "mean"):
            raise ValueError("reward_aggregation must be 'sum' or 'mean'")
        self.reward_aggregation = reward_aggregation
        self.rng = np.random.default_rng(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        self.last_losses = {}

    @staticmethod
    def _split_observation(observation):
        if isinstance(observation, dict):
            if "local" not in observation:
                raise KeyError("multi-agent observations require a 'local' entry")
            local = np.asarray(observation["local"], dtype=np.float32)
            critic = np.asarray(observation.get("critic", local), dtype=np.float32)
            return local, critic
        array = np.asarray(observation, dtype=np.float32)
        return array, array

    def policy(self, observations, batched: bool = False) -> tuple[list[int], list[tf.Tensor]]:
        """Sample one action per agent from the shared decentralized policy."""
        local, _ = self._split_observation(observations)
        joint_action, joint_action_probs = [], []
        for observation in local:
            tensor = tf.convert_to_tensor(observation, dtype=tf.float32)
            if not batched:
                tensor = tf.expand_dims(tensor, axis=0)
            action_probs = self.model_actor(tensor)
            probabilities = np.asarray(action_probs[0], dtype=np.float64)
            probabilities = probabilities / probabilities.sum()
            action = int(self.rng.choice(len(probabilities), p=probabilities))
            joint_action.append(action)
            joint_action_probs.append(action_probs)
        return joint_action, joint_action_probs

    def values(self, observations, batched: bool = False) -> tf.Tensor:
        """Estimate the joint value from the centralized critic state."""
        _, critic = self._split_observation(observations)
        tensor = tf.convert_to_tensor(critic, dtype=tf.float32)
        if not batched:
            tensor = tf.expand_dims(tensor, axis=0)
        return self.model_critic(tensor)

    @staticmethod
    def _apply_gradients(optimizer, gradients, variables):
        pairs = [(gradient, variable) for gradient, variable in zip(gradients, variables) if gradient is not None]
        if isinstance(optimizer, (list, tuple)):
            if len(optimizer) != len(variables):
                raise ValueError(
                    f"received {len(optimizer)} optimizers for {len(variables)} trainable variables"
                )
            for item, gradient, variable in zip(optimizer, gradients, variables):
                if gradient is not None:
                    item.apply_gradients([(gradient, variable)])
        elif pairs:
            optimizer.apply_gradients(pairs)

    @staticmethod
    def _td_targets(team_rewards, terminals, gamma, next_values):
        return tf.stop_gradient(team_rewards + (1.0 - terminals) * gamma * next_values)

    @staticmethod
    def _actor_objective(policy_loss, entropy, alpha):
        return policy_loss - alpha * entropy

    def update(self, batch: list[MultiAgentInteraction]):
        if not batch:
            return

        actor_observations = tf.convert_to_tensor(
            np.asarray([item.actor_observations for item in batch]), dtype=tf.float32
        )
        critic_states = tf.convert_to_tensor(
            np.asarray([item.critic_state for item in batch]), dtype=tf.float32
        )
        actions = tf.convert_to_tensor(
            np.asarray([item.actions for item in batch]), dtype=tf.int32
        )
        rewards = tf.convert_to_tensor(
            np.asarray([item.rewards for item in batch]), dtype=tf.float32
        )
        next_critic_states = tf.convert_to_tensor(
            np.asarray([item.next_critic_state for item in batch]), dtype=tf.float32
        )
        terminals = tf.convert_to_tensor(
            [[float(item.terminated or item.truncated)] for item in batch], dtype=tf.float32
        )
        if self.reward_aggregation == "mean":
            team_rewards = tf.reduce_mean(rewards, axis=-1, keepdims=True)
        else:
            team_rewards = tf.reduce_sum(rewards, axis=-1, keepdims=True)

        with tf.GradientTape() as critic_tape, tf.GradientTape() as actor_tape:
            values = self.model_critic(critic_states)
            next_values = self.model_critic(next_critic_states)
            targets = self._td_targets(team_rewards, terminals, self.gamma, next_values)
            advantages = tf.stop_gradient(targets - values)

            all_probs = []
            all_log_probs = []
            for agent_index in range(self.n_agents):
                probabilities = self.model_actor(actor_observations[:, agent_index])
                probabilities = tf.clip_by_value(probabilities, 1e-7, 1.0)
                chosen = tf.reduce_sum(
                    probabilities
                    * tf.one_hot(actions[:, agent_index], depth=tf.shape(probabilities)[-1]),
                    axis=-1,
                )
                all_probs.append(probabilities)
                all_log_probs.append(tf.math.log(chosen))

            probabilities = tf.stack(all_probs, axis=1)
            chosen_log_probs = tf.stack(all_log_probs, axis=1)
            entropy = -tf.reduce_mean(
                tf.reduce_sum(probabilities * tf.math.log(probabilities), axis=-1)
            )
            policy_loss = tf.reduce_mean(-chosen_log_probs * advantages)
            actor_loss = self._actor_objective(policy_loss, entropy, self.alpha)
            critic_loss = tf.reduce_mean(keras.losses.huber(targets, values))

        actor_gradients = actor_tape.gradient(actor_loss, self.model_actor.trainable_variables)
        critic_gradients = critic_tape.gradient(critic_loss, self.model_critic.trainable_variables)
        self._apply_gradients(
            self.optimizer_actor, actor_gradients, self.model_actor.trainable_variables
        )
        self._apply_gradients(
            self.optimizer_critic, critic_gradients, self.model_critic.trainable_variables
        )
        self.last_losses = {
            "actor": float(actor_loss.numpy()),
            "critic": float(critic_loss.numpy()),
            "entropy": float(entropy.numpy()),
        }

    def _reset_environment(self, episode: int):
        try:
            return self.env.reset(seed=self.seed + episode)
        except TypeError:
            return self.env.reset()

    def run_episode(self, episode: int, total_steps: int, max_steps_per_episode: int):
        episode_reward = np.zeros(self.n_agents, dtype=np.float32)
        batch = []
        observation, _ = self._reset_environment(episode)

        for step_index in range(max_steps_per_episode):
            local, critic = self._split_observation(observation)
            actions, _ = self.policy(observation)
            next_observation, rewards, terminated, truncated, _ = self.env.step(actions)
            next_local, next_critic = self._split_observation(next_observation)
            terminated_flag = bool(np.any(terminated))
            truncated_flag = bool(np.any(truncated))
            reward_array = np.asarray(rewards, dtype=np.float32)

            batch.append(
                MultiAgentInteraction(
                    actor_observations=local,
                    critic_state=critic,
                    actions=actions,
                    rewards=reward_array,
                    next_actor_observations=next_local,
                    next_critic_state=next_critic,
                    terminated=terminated_flag,
                    truncated=truncated_flag,
                )
            )
            episode_reward += reward_array
            observation = next_observation
            if terminated_flag or truncated_flag:
                break

        self.update(batch)
        return episode_reward, batch, step_index + 1
