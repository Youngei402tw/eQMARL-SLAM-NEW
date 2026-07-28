import numpy as np

from eqmarl.environments.active_slam import (
    ACTION_FORWARD,
    ACTION_WAIT,
    GridSLAMBackend,
    MultiAgentSLAMEnv,
)
from eqmarl.policies import FrontierJointPolicy


def test_reset_is_seeded_and_matches_declared_space():
    env = MultiAgentSLAMEnv(map_size=18, time_limit=10, seed=7)
    first, info = env.reset(seed=31)
    first_map = env.ground_truth.copy()
    second, second_info = env.reset(seed=31)
    assert info == second_info == {"map_seed": 31}
    assert np.array_equal(first_map, env.ground_truth)
    assert np.array_equal(first["local"], second["local"])
    assert env.observation_space.contains(second)


def test_grid_slam_marks_free_and_occupied_cells():
    backend = GridSLAMBackend(np.asarray([0.0]), max_range=4.0)
    backend.reset(np.asarray([5.0, 5.0, 0.0]), (12, 12))
    belief = backend.update(np.zeros(3), np.asarray([2.0]))
    assert belief.observed[5, 6]
    assert belief.occupancy_log_odds[5, 6] < 0.0
    assert belief.observed[5, 7]
    assert belief.occupancy_log_odds[5, 7] > 0.0


def test_step_returns_shared_reward_and_metrics():
    env = MultiAgentSLAMEnv(map_size=18, time_limit=2, seed=4)
    observation, _ = env.reset(seed=4)
    next_observation, rewards, terminated, truncated, info = env.step(
        [ACTION_WAIT, ACTION_WAIT]
    )
    assert env.observation_space.contains(observation)
    assert env.observation_space.contains(next_observation)
    assert rewards.shape == (2,)
    assert rewards[0] == rewards[1]
    assert not terminated
    assert not truncated
    assert {"coverage", "occupancy_iou", "pose_rmse", "success"} <= info.keys()


def test_collision_is_counted():
    env = MultiAgentSLAMEnv(map_size=18, time_limit=5, seed=11)
    env.reset(seed=11)
    pose = env.true_poses[0].copy()
    pose[:2] = [1.0, 1.0]
    pose[2] = np.pi
    env.true_poses[0] = pose
    _, _, _, _, info = env.step([ACTION_FORWARD, ACTION_WAIT])
    assert info["collisions"] >= 1.0


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
