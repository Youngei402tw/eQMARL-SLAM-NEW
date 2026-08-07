import inspect

import numpy as np

from eqmarl.environments.active_slam import (
    ACTION_FORWARD,
    ACTION_LEFT,
    ACTION_RIGHT,
    GridSLAMBackend,
    MultiAgentSLAMEnv,
)
from eqmarl.policies import FrontierJointPolicy


def test_faithful_environment_defaults_are_locked():
    parameters = inspect.signature(MultiAgentSLAMEnv).parameters
    expected = {
        "map_size": 24,
        "n_agents": 2,
        "n_beams": 36,
        "lidar_range": 8.0,
        "patch_size": 7,
        "time_limit": 250,
        "coverage_target": 0.9,
        "obstacle_rectangles": 9,
        "lidar_noise": 0.03,
        "odometry_translation_noise": 0.04,
        "odometry_rotation_noise": 0.015,
        "seed": 0,
    }
    assert {name: parameters[name].default for name in expected} == expected


def test_reset_is_seeded_and_matches_declared_space():
    env = MultiAgentSLAMEnv(map_size=18, time_limit=10, seed=7)
    first, info = env.reset(seed=31)
    first_map = env.ground_truth.copy()
    second, second_info = env.reset(seed=31)
    assert info == second_info == {"map_seed": 31}
    assert np.array_equal(first_map, env.ground_truth)
    assert np.array_equal(first["local"], second["local"])
    assert env.observation_space.contains(second)
    assert np.array_equal(env.action_space.nvec, [3, 3])
    assert second["local"].shape == (2, 7 * 7 * 3)
    assert second["critic"].shape == (2, 147)
    assert np.array_equal(second["local"], second["critic"])
    assert not np.shares_memory(second["local"], second["critic"])


def test_grid_slam_marks_free_and_occupied_cells():
    backend = GridSLAMBackend(np.asarray([0.0]), max_range=4.0)
    backend.reset(np.asarray([5.0, 5.0, 0.0]), (12, 12))
    belief = backend.update(np.zeros(3), np.asarray([2.0]))
    assert belief.observed[5, 6]
    assert belief.occupancy_log_odds[5, 6] < 0.0
    assert belief.observed[5, 7]
    assert belief.occupancy_log_odds[5, 7] > 0.0


def test_repeated_scan_without_new_cells_does_not_reduce_covariance():
    backend = GridSLAMBackend(np.asarray([0.0]), max_range=4.0)
    backend.reset(np.asarray([5.0, 5.0, 0.0]), (12, 12))
    first = backend.update(np.zeros(3), np.asarray([2.0]))
    first_trace = np.trace(first.covariance)
    second = backend.update(np.zeros(3), np.asarray([2.0]))
    assert np.trace(second.covariance) >= first_trace


def test_step_returns_shared_reward_and_metrics():
    env = MultiAgentSLAMEnv(map_size=18, time_limit=2, seed=4)
    observation, _ = env.reset(seed=4)
    next_observation, rewards, terminated, truncated, info = env.step(
        [ACTION_LEFT, ACTION_RIGHT]
    )
    assert env.observation_space.contains(observation)
    assert env.observation_space.contains(next_observation)
    assert rewards.shape == (2,)
    assert rewards[0] == rewards[1]
    assert not terminated
    assert not truncated
    assert {
        "coverage", "occupancy_iou", "pose_rmse", "success", "new_observed_cells"
    } <= info.keys()
    assert info["step_reward_components"].keys() == {
        "coverage_reward",
        "uncertainty_reward",
        "milestone_reward",
        "collision_penalty",
        "step_penalty",
    }


def test_milestone_rewards_are_progressive_and_paid_once():
    env = MultiAgentSLAMEnv(
        map_size=18,
        coverage_target=0.99,
        coverage_milestones={0.90: 0.5, 0.95: 1.0, 0.98: 2.0, 0.99: 5.0},
    )

    assert env._milestone_reward(0.89, 0.90) == 0.5
    assert env._milestone_reward(0.94, 0.99) == 8.0
    assert env._milestone_reward(0.99, 0.99) == 0.0
    assert env._milestone_metric_name(0.98) == "steps_to_coverage_98"


def test_milestone_configuration_rejects_unreachable_or_negative_rewards():
    with np.testing.assert_raises_regex(ValueError, "no greater than coverage_target"):
        MultiAgentSLAMEnv(
            coverage_target=0.95, coverage_milestones={0.99: 5.0}
        )
    with np.testing.assert_raises_regex(ValueError, "non-negative"):
        MultiAgentSLAMEnv(
            coverage_target=0.99, coverage_milestones={0.99: -1.0}
        )


def test_crossing_final_milestone_pays_jackpot_and_terminates():
    env = MultiAgentSLAMEnv(
        map_size=18,
        time_limit=250,
        coverage_target=0.99,
        coverage_milestones={0.90: 0.5, 0.95: 1.0, 0.98: 2.0, 0.99: 5.0},
        seed=4,
    )
    env.reset(seed=4)
    coverages = iter((0.89, 0.99, 0.99, 0.99))
    env._coverage = lambda: next(coverages)

    _, _, terminated, truncated, info = env.step([ACTION_LEFT, ACTION_RIGHT])

    assert terminated
    assert not truncated
    assert info["step_reward_components"]["milestone_reward"] == 8.5
    assert info["milestone_reward"] == 8.5
    assert info["steps_to_coverage_90"] == 1.0
    assert info["steps_to_coverage_99"] == 1.0


def test_reward_component_totals_reconstruct_episode_return():
    env = MultiAgentSLAMEnv(map_size=18, time_limit=2, seed=4)
    env.reset(seed=4)
    episode_return = 0.0
    for _ in range(2):
        _, rewards, _, _, _ = env.step([ACTION_LEFT, ACTION_RIGHT])
        episode_return += float(rewards[0])
    metrics = env.episode_metrics()
    reconstructed = (
        metrics["coverage_reward"]
        + metrics["uncertainty_reward"]
        + metrics["milestone_reward"]
        - metrics["collision_penalty"]
        - metrics["step_penalty"]
    )
    np.testing.assert_allclose(reconstructed, episode_return, atol=1e-6)


def test_collision_is_counted():
    env = MultiAgentSLAMEnv(map_size=18, time_limit=5, seed=11)
    env.reset(seed=11)
    pose = env.true_poses[0].copy()
    pose[:2] = [1.0, 1.0]
    pose[2] = np.pi
    env.true_poses[0] = pose
    _, _, _, _, info = env.step([ACTION_FORWARD, ACTION_LEFT])
    assert info["collisions"] >= 1.0


def test_out_of_bounds_physical_motion_is_a_collision():
    env = MultiAgentSLAMEnv(
        map_size=18,
        odometry_translation_noise=0.0,
        odometry_rotation_noise=0.0,
        seed=11,
    )
    env.reset(seed=11)
    env.ground_truth[:] = False
    cases = (
        np.asarray([0.0, 5.0, np.pi], dtype=np.float32),
        np.asarray([17.0, 5.0, 0.0], dtype=np.float32),
    )
    for pose in cases:
        env.true_poses = np.stack(
            [pose, np.asarray([10.0, 10.0, 0.0], dtype=np.float32)]
        )
        old_pose = env.true_poses[0].copy()
        _, collisions = env._execute_actions([ACTION_FORWARD, ACTION_LEFT])
        assert collisions == 1
        assert np.array_equal(env.true_poses[0], old_pose)


def test_bounded_grid_slam_rejects_out_of_map_pose_estimates():
    backend = GridSLAMBackend(
        np.asarray([0.0]), max_range=4.0, constrain_pose_to_map=True
    )
    backend.reset(np.asarray([5.0, 5.0, 0.0]), (12, 12))
    belief = backend.update(np.asarray([-50.0, 50.0, 0.0]), np.asarray([4.0]))
    assert 0.0 <= belief.pose[0] <= 11.0
    assert 0.0 <= belief.pose[1] <= 11.0


def test_unbounded_grid_slam_preserves_faithful_pose_behavior():
    backend = GridSLAMBackend(np.asarray([0.0]), max_range=4.0)
    backend.reset(np.asarray([5.0, 5.0, 0.0]), (12, 12))
    belief = backend.update(np.asarray([-50.0, 0.0, 0.0]), np.asarray([4.0]))
    assert belief.pose[0] < 0.0


def test_bounded_slam_pose_option_configures_every_backend():
    faithful = MultiAgentSLAMEnv(map_size=18)
    improved = MultiAgentSLAMEnv(map_size=18, bounded_slam_pose=True)
    assert all(not backend.constrain_pose_to_map for backend in faithful.backends)
    assert all(backend.constrain_pose_to_map for backend in improved.backends)


def test_actions_match_minigrid_left_right_forward_order():
    env = MultiAgentSLAMEnv(map_size=18, seed=5)
    pose = np.asarray([5.0, 5.0, 0.0], dtype=np.float32)
    assert np.isclose(env._propose_motion(pose, ACTION_LEFT)[2], -np.pi / 2.0)
    assert np.isclose(env._propose_motion(pose, ACTION_RIGHT)[2], np.pi / 2.0)
    assert np.array_equal(env._propose_motion(pose, ACTION_FORWARD)[:2], [6.0, 5.0])


def test_ego_patch_places_forward_at_the_top_for_every_heading():
    env = MultiAgentSLAMEnv(map_size=18, seed=6)
    center = env.patch_size // 2
    headings = (
        (0.0, (6, 5)),
        (np.pi / 2.0, (5, 6)),
        (np.pi, (4, 5)),
        (-np.pi / 2.0, (5, 4)),
    )
    for angle, (ahead_x, ahead_y) in headings:
        channels = np.zeros((3, env.map_size, env.map_size), dtype=np.float32)
        channels[0, ahead_y, ahead_x] = 1.0
        patch = env._extract_patch(channels, np.asarray([5.0, 5.0, angle]))
        assert patch[0, center - 1, center] == 1.0


def test_frontier_policy_emits_valid_joint_action():
    env = MultiAgentSLAMEnv(map_size=18, time_limit=5, seed=2)
    env.reset(seed=2)
    action = np.asarray(FrontierJointPolicy().action(env), dtype=np.int64)
    assert env.action_space.contains(action)


def test_observation_patch_stays_fixed_when_estimate_leaves_map():
    env = MultiAgentSLAMEnv(map_size=18, time_limit=5, seed=3)
    env.reset(seed=3)
    env.backends[0].belief.pose[:2] = [-50.0, 50.0]
    observation = env._observation()
    assert env.observation_space.contains(observation)
