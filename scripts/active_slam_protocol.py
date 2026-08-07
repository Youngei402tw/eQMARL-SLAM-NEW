"""Expected configuration contract for audited Active-SLAM protocols."""

from __future__ import annotations


def expected_config(
    framework: str, seed: int, protocol: str, prefix: str
) -> dict[str, object]:
    critic_name = "eqmarl" if framework == "eqmarl_psi+" else framework
    expected = {
        "experiment.roots.root_dir": f"experiment_output/{prefix}_{framework}",
        "experiment.train.n_episodes": 1000,
        "experiment.train.max_steps_per_episode": 250,
        "experiment.algorithm.init_func": "eqmarl.algorithms.MAA2C",
        "experiment.algorithm.init_params.gamma": 0.99,
        "experiment.algorithm.init_params.alpha": 0.01,
        "experiment.algorithm.init_params.alpha_final": 0.001,
        "experiment.algorithm.init_params.alpha_decay_episodes": 500,
        "experiment.algorithm.init_params.seed": seed,
        "experiment.algorithm.init_params.reward_aggregation": "mean",
        "experiment.algorithm.init_params.normalize_advantages": True,
        "experiment.algorithm.init_params.gradient_clip_norm": 1.0,
        "experiment.algorithm.init_params.episode_metrics_callback": (
            "eqmarl.environments.active_slam.episode_metrics_callback"
        ),
        "experiment.algorithm.init_params.env.func": (
            "eqmarl.environments.active_slam.active_slam_make"
        ),
        "experiment.algorithm.init_params.env.params.map_size": 24,
        "experiment.algorithm.init_params.env.params.n_agents": 2,
        "experiment.algorithm.init_params.env.params.n_beams": 36,
        "experiment.algorithm.init_params.env.params.patch_size": 7,
        "experiment.algorithm.init_params.env.params.time_limit": 250,
        "experiment.algorithm.init_params.env.params.seed": seed,
        "experiment.algorithm.init_params.model_actor.init_func": (
            "eqmarl.active_slam_models.generate_actor_classical"
        ),
        "experiment.algorithm.init_params.model_actor.init_params.observation_dim": 147,
        "experiment.algorithm.init_params.model_actor.init_params.n_actions": 3,
        "experiment.algorithm.init_params.model_actor.init_params.units": [100],
        "experiment.algorithm.init_params.model_actor.build_shape": [None, 147],
        "experiment.algorithm.init_params.optimizer_actor.params.learning_rate": 0.0001,
        "experiment.algorithm.init_params.model_critic.init_func": (
            f"eqmarl.active_slam_models.generate_critic_{critic_name}"
        ),
        "experiment.algorithm.init_params.model_critic.init_params.observation_dim": 147,
        "experiment.algorithm.init_params.model_critic.init_params.n_agents": 2,
        "experiment.algorithm.init_params.model_critic.init_params.n_actions": 3,
        "experiment.algorithm.init_params.model_critic.build_shape": [None, 2, 147],
    }
    if protocol in {"bounded_pose", "milestone99"}:
        expected[
            "experiment.algorithm.init_params.env.params.bounded_slam_pose"
        ] = True
    if protocol == "milestone99":
        env = "experiment.algorithm.init_params.env.params"
        expected[f"{env}.coverage_target"] = 0.99
        expected[f"{env}.coverage_milestones"] = {
            0.90: 0.5,
            0.95: 1.0,
            0.98: 2.0,
            0.99: 5.0,
        }
    critic = "experiment.algorithm.init_params.model_critic.init_params"
    optimizer = "experiment.algorithm.init_params.optimizer_critic"
    if framework in {"eqmarl_psi+", "qfctde"}:
        expected.update(
            {
                f"{critic}.d_qubits": 4,
                f"{critic}.n_layers": 5,
                f"{critic}.nn_activation": "linear",
                f"{critic}.trainable_w_enc": False,
                optimizer: [
                    {
                        "func": "tensorflow.keras.optimizers.Adam",
                        "params": {"learning_rate": rate},
                    }
                    for rate in (0.001, 0.001, 0.01, 0.01)
                ],
            }
        )
        if framework == "eqmarl_psi+":
            expected[f"{critic}.input_entanglement_type"] = "psi+"
    else:
        expected[f"{critic}.units"] = [100]
        expected[f"{optimizer}.params.learning_rate"] = 0.0001
    return expected
