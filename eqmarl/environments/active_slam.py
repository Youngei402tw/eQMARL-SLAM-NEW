"""Gym-native two-robot active SLAM research environment.

The environment deliberately keeps control learning separate from geometric
SLAM. Agents choose motion primitives; :class:`GridSLAMBackend` consumes noisy
odometry and lidar scans to maintain pose and occupancy beliefs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import gymnasium as gym
import numpy as np

from ..active_slam_state import largest_free_component


ACTION_LEFT = 0
ACTION_RIGHT = 1
ACTION_FORWARD = 2
N_ACTIONS = 3


@dataclass
class SLAMBelief:
    pose: np.ndarray
    covariance: np.ndarray
    occupancy_log_odds: np.ndarray
    observed: np.ndarray
    scan_match_score: float


class SLAMBackend(Protocol):
    def reset(self, initial_pose: np.ndarray, map_shape: tuple[int, int]) -> SLAMBelief:
        ...

    def update(self, odometry: np.ndarray, lidar_scan: np.ndarray) -> SLAMBelief:
        ...


def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _bresenham(start: tuple[int, int], end: tuple[int, int]):
    """Yield integer grid cells along a line, including both endpoints."""
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        yield x0, y0
        if x0 == x1 and y0 == y1:
            break
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x0 += sx
        if doubled <= dx:
            error += dx
            y0 += sy


class GridSLAMBackend:
    """Small conventional scan-matching occupancy-grid SLAM backend.

    This is intentionally transparent and deterministic rather than a
    production SLAM implementation. It combines noisy odometry, local
    correlative endpoint matching, and a log-odds inverse sensor model.
    """

    def __init__(
        self,
        beam_angles: np.ndarray,
        max_range: float,
        occupied_increment: float = 0.9,
        free_increment: float = -0.35,
    ):
        self.beam_angles = np.asarray(beam_angles, dtype=np.float32)
        self.max_range = float(max_range)
        self.occupied_increment = float(occupied_increment)
        self.free_increment = float(free_increment)
        self.belief = None

    def reset(self, initial_pose: np.ndarray, map_shape: tuple[int, int]) -> SLAMBelief:
        self.belief = SLAMBelief(
            pose=np.asarray(initial_pose, dtype=np.float32).copy(),
            covariance=np.diag([0.08, 0.08, 0.03]).astype(np.float32),
            occupancy_log_odds=np.zeros(map_shape, dtype=np.float32),
            observed=np.zeros(map_shape, dtype=bool),
            scan_match_score=0.0,
        )
        return self.belief

    def _endpoint_score(self, pose: np.ndarray, scan: np.ndarray) -> float:
        log_odds = self.belief.occupancy_log_odds
        height, width = log_odds.shape
        scores = []
        for relative_angle, distance in zip(self.beam_angles, scan):
            if distance >= self.max_range * 0.995:
                continue
            angle = pose[2] + relative_angle
            x = int(round(pose[0] + distance * np.cos(angle)))
            y = int(round(pose[1] + distance * np.sin(angle)))
            if 0 <= x < width and 0 <= y < height and self.belief.observed[y, x]:
                scores.append(float(np.tanh(log_odds[y, x])))
        return float(np.mean(scores)) if scores else 0.0

    def _scan_match(self, predicted: np.ndarray, scan: np.ndarray) -> tuple[np.ndarray, float]:
        candidates = []
        for dx in (-0.25, 0.0, 0.25):
            for dy in (-0.25, 0.0, 0.25):
                for dtheta in (-np.deg2rad(4.0), 0.0, np.deg2rad(4.0)):
                    pose = predicted + np.asarray([dx, dy, dtheta], dtype=np.float32)
                    pose[2] = _wrap_angle(pose[2])
                    score = self._endpoint_score(pose, scan)
                    regularizer = 0.02 * (abs(dx) + abs(dy) + abs(dtheta))
                    candidates.append((score - regularizer, pose))
        score, pose = max(candidates, key=lambda item: item[0])
        return pose, float(score)

    def _update_map(self, pose: np.ndarray, scan: np.ndarray):
        log_odds = self.belief.occupancy_log_odds
        observed = self.belief.observed
        height, width = log_odds.shape
        origin = (int(round(pose[0])), int(round(pose[1])))
        for relative_angle, measured_distance in zip(self.beam_angles, scan):
            distance = min(float(measured_distance), self.max_range)
            angle = pose[2] + relative_angle
            endpoint = (
                int(round(pose[0] + distance * np.cos(angle))),
                int(round(pose[1] + distance * np.sin(angle))),
            )
            cells = [
                (x, y)
                for x, y in _bresenham(origin, endpoint)
                if 0 <= x < width and 0 <= y < height
            ]
            if not cells:
                continue
            free_cells = cells[:-1] if distance < self.max_range * 0.995 else cells
            for x, y in free_cells:
                log_odds[y, x] += self.free_increment
                observed[y, x] = True
            if distance < self.max_range * 0.995:
                x, y = cells[-1]
                log_odds[y, x] += self.occupied_increment
                observed[y, x] = True
        np.clip(log_odds, -5.0, 5.0, out=log_odds)

    def update(self, odometry: np.ndarray, lidar_scan: np.ndarray) -> SLAMBelief:
        if self.belief is None:
            raise RuntimeError("reset must be called before update")
        predicted = self.belief.pose + np.asarray(odometry, dtype=np.float32)
        predicted[2] = _wrap_angle(predicted[2])
        matched, score = self._scan_match(predicted, lidar_scan)
        self.belief.pose = matched
        previously_observed = int(np.count_nonzero(self.belief.observed))
        self._update_map(matched, lidar_scan)
        new_cells = int(np.count_nonzero(self.belief.observed)) - previously_observed
        process_noise = np.diag([0.025, 0.025, 0.01]).astype(np.float32)
        motion_scale = max(
            0.1,
            float(np.linalg.norm(odometry[:2])) + abs(float(odometry[2])) / np.pi,
        )
        confidence = float(np.clip((score + 1.0) / 2.0, 0.05, 1.0))
        information_scale = min(1.0, new_cells / max(1, len(self.beam_angles)))
        self.belief.covariance = self.belief.covariance + process_noise * motion_scale
        if new_cells:
            self.belief.covariance *= 1.0 - 0.35 * confidence * information_scale
        diagonal = np.clip(np.diag(self.belief.covariance), 1e-3, 2.0)
        self.belief.covariance = np.diag(diagonal).astype(np.float32)
        self.belief.scan_match_score = score
        return self.belief


class MultiAgentSLAMEnv(gym.Env):
    """Cooperative two-robot active occupancy-grid SLAM environment."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        map_size: int = 24,
        n_agents: int = 2,
        n_beams: int = 36,
        lidar_range: float = 8.0,
        patch_size: int = 7,
        time_limit: int = 250,
        coverage_target: float = 0.9,
        obstacle_rectangles: int = 9,
        lidar_noise: float = 0.03,
        odometry_translation_noise: float = 0.04,
        odometry_rotation_noise: float = 0.015,
        seed: int = 0,
    ):
        super().__init__()
        if n_agents != 2:
            raise ValueError("the initial active-SLAM environment supports exactly two robots")
        if patch_size % 2 != 1:
            raise ValueError("patch_size must be odd")
        self.map_size = int(map_size)
        self.n_agents = int(n_agents)
        self.n_beams = int(n_beams)
        self.lidar_range = float(lidar_range)
        self.patch_size = int(patch_size)
        self.time_limit = int(time_limit)
        self.coverage_target = float(coverage_target)
        self.obstacle_rectangles = int(obstacle_rectangles)
        self.lidar_noise = float(lidar_noise)
        self.odometry_translation_noise = float(odometry_translation_noise)
        self.odometry_rotation_noise = float(odometry_rotation_noise)
        self.initial_seed = int(seed)
        self.beam_angles = np.linspace(-np.pi, np.pi, self.n_beams, endpoint=False).astype(
            np.float32
        )
        self.action_space = gym.spaces.MultiDiscrete([N_ACTIONS] * self.n_agents)
        self.observation_dim = self.patch_size * self.patch_size * 3
        local_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_agents, self.observation_dim),
            dtype=np.float32,
        )
        critic_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_agents, self.observation_dim),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict({"local": local_space, "critic": critic_space})
        self.rng = np.random.default_rng(self.initial_seed)
        self.backends = [
            GridSLAMBackend(self.beam_angles, self.lidar_range) for _ in range(self.n_agents)
        ]
        self.ground_truth = None
        self.reachable = None
        self.true_poses = None
        self.step_count = 0
        self.collision_count = 0
        self.path_length = 0.0
        self.new_observed_cells = 0
        self.last_metrics = {}

    def _generate_map(self) -> tuple[np.ndarray, np.ndarray]:
        grid = np.zeros((self.map_size, self.map_size), dtype=bool)
        grid[[0, -1], :] = True
        grid[:, [0, -1]] = True
        for _ in range(self.obstacle_rectangles):
            width = int(self.rng.integers(1, 4))
            height = int(self.rng.integers(1, 5))
            x = int(self.rng.integers(2, self.map_size - width - 2))
            y = int(self.rng.integers(2, self.map_size - height - 2))
            grid[y : y + height, x : x + width] = True
        reachable = largest_free_component(grid)
        grid[~reachable] = True
        return grid, reachable

    def _sample_initial_poses(self) -> np.ndarray:
        cells = np.argwhere(self.reachable)
        first = cells[int(self.rng.integers(len(cells)))]
        distances = np.linalg.norm(cells - first, axis=1)
        candidates = cells[distances >= max(4.0, self.map_size / 4.0)]
        second = candidates[int(self.rng.integers(len(candidates)))] if len(candidates) else cells[-1]
        return np.asarray(
            [[first[1], first[0], 0.0], [second[1], second[0], np.pi]], dtype=np.float32
        )

    def _ray_cast(self, pose: np.ndarray) -> np.ndarray:
        distances = np.full(self.n_beams, self.lidar_range, dtype=np.float32)
        for index, relative_angle in enumerate(self.beam_angles):
            angle = pose[2] + relative_angle
            for distance in np.arange(0.1, self.lidar_range + 0.1, 0.1):
                x = int(round(pose[0] + distance * np.cos(angle)))
                y = int(round(pose[1] + distance * np.sin(angle)))
                if (
                    x < 0
                    or y < 0
                    or x >= self.map_size
                    or y >= self.map_size
                    or self.ground_truth[y, x]
                ):
                    distances[index] = min(float(distance), self.lidar_range)
                    break
        noisy = distances + self.rng.normal(0.0, self.lidar_noise, self.n_beams)
        return np.clip(noisy, 0.05, self.lidar_range).astype(np.float32)

    @staticmethod
    def _fuse_beliefs(beliefs: list[SLAMBelief]) -> tuple[np.ndarray, np.ndarray]:
        stacked = np.stack([belief.occupancy_log_odds for belief in beliefs])
        confidence_owner = np.argmax(np.abs(stacked), axis=0)
        fused = np.take_along_axis(stacked, confidence_owner[None, ...], axis=0)[0]
        observed = np.any(np.stack([belief.observed for belief in beliefs]), axis=0)
        return fused.astype(np.float32), observed

    @staticmethod
    def _frontiers(log_odds: np.ndarray, observed: np.ndarray) -> np.ndarray:
        free = observed & (log_odds < 0.0)
        unknown = ~observed
        adjacent_unknown = np.zeros_like(unknown)
        adjacent_unknown[1:] |= unknown[:-1]
        adjacent_unknown[:-1] |= unknown[1:]
        adjacent_unknown[:, 1:] |= unknown[:, :-1]
        adjacent_unknown[:, :-1] |= unknown[:, 1:]
        return free & adjacent_unknown

    def _extract_patch(self, channels: np.ndarray, pose: np.ndarray) -> np.ndarray:
        radius = self.patch_size // 2
        padded = np.pad(channels, ((0, 0), (radius, radius), (radius, radius)))
        x = np.clip(int(round(pose[0])) + radius, radius, padded.shape[2] - radius - 1)
        y = np.clip(int(round(pose[1])) + radius, radius, padded.shape[1] - radius - 1)
        patch = padded[:, y - radius : y + radius + 1, x - radius : x + radius + 1]
        quarter_turns = int(round(pose[2] / (np.pi / 2.0))) % 4
        return np.rot90(patch, k=(quarter_turns + 1) % 4, axes=(1, 2)).copy()

    def _observation(self) -> dict[str, np.ndarray]:
        beliefs = [backend.belief for backend in self.backends]
        fused_log_odds, fused_observed = self._fuse_beliefs(beliefs)
        occupancy = 2.0 / (1.0 + np.exp(-fused_log_odds)) - 1.0
        observed_channel = fused_observed.astype(np.float32) * 2.0 - 1.0
        frontier_channel = self._frontiers(fused_log_odds, fused_observed).astype(np.float32)
        channels = np.stack([occupancy, observed_channel, frontier_channel])
        observations = []
        for belief in beliefs:
            # MiniGrid flattens its channel-last 7x7x3 ego-centric image.
            patch = self._extract_patch(channels, belief.pose).transpose(1, 2, 0)
            observations.append(np.clip(patch.reshape(-1), -1.0, 1.0))
        local = np.stack(observations)
        # Match MiniGrid: each critic partition receives one agent observation.
        return {"local": local, "critic": local.copy()}

    def _coverage(self) -> float:
        _, observed = self._fuse_beliefs([backend.belief for backend in self.backends])
        return float(np.count_nonzero(observed & self.reachable) / np.count_nonzero(self.reachable))

    def _total_logdet(self) -> float:
        return float(
            sum(np.linalg.slogdet(backend.belief.covariance)[1] for backend in self.backends)
        )

    def _propose_motion(self, pose: np.ndarray, action: int) -> np.ndarray:
        proposal = pose.copy()
        if action == ACTION_LEFT:
            proposal[2] = _wrap_angle(proposal[2] - np.pi / 2.0)
        elif action == ACTION_RIGHT:
            proposal[2] = _wrap_angle(proposal[2] + np.pi / 2.0)
        elif action == ACTION_FORWARD:
            proposal[0] += round(float(np.cos(proposal[2])))
            proposal[1] += round(float(np.sin(proposal[2])))
        return proposal

    def _execute_actions(self, actions: np.ndarray) -> tuple[np.ndarray, int]:
        old_poses = self.true_poses.copy()
        proposed = np.stack(
            [self._propose_motion(pose, int(action)) for pose, action in zip(old_poses, actions)]
        )
        collided = np.zeros(self.n_agents, dtype=bool)
        for index, pose in enumerate(proposed):
            x, y = int(round(pose[0])), int(round(pose[1]))
            if self.ground_truth[y, x]:
                collided[index] = True
                proposed[index] = old_poses[index]
        old_cells = [tuple(np.rint(pose[:2]).astype(int)) for pose in old_poses]
        new_cells = [tuple(np.rint(pose[:2]).astype(int)) for pose in proposed]
        if new_cells[0] == new_cells[1] or new_cells == old_cells[::-1]:
            collided[:] = True
            proposed = old_poses
        odometry = proposed - old_poses
        odometry[:, 2] = [_wrap_angle(value) for value in odometry[:, 2]]
        moved = np.linalg.norm(odometry[:, :2], axis=1)
        self.path_length += float(np.sum(moved))
        translation_noise = self.rng.normal(
            0.0, self.odometry_translation_noise, size=(self.n_agents, 2)
        )
        rotation_noise = self.rng.normal(
            0.0, self.odometry_rotation_noise, size=(self.n_agents, 1)
        )
        noisy_odometry = odometry + np.concatenate([translation_noise, rotation_noise], axis=1)
        self.true_poses = proposed.astype(np.float32)
        return noisy_odometry.astype(np.float32), int(np.count_nonzero(collided))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        resolved_seed = self.initial_seed if seed is None else int(seed)
        self.rng = np.random.default_rng(resolved_seed)
        self.ground_truth, self.reachable = self._generate_map()
        self.true_poses = self._sample_initial_poses()
        self.step_count = 0
        self.collision_count = 0
        self.path_length = 0.0
        self.new_observed_cells = 0
        for index, backend in enumerate(self.backends):
            initial = self.true_poses[index].copy()
            initial[:2] += self.rng.normal(0.0, 0.05, 2)
            initial[2] = _wrap_angle(initial[2] + self.rng.normal(0.0, 0.02))
            backend.reset(initial, self.ground_truth.shape)
            backend.update(np.zeros(3, dtype=np.float32), self._ray_cast(self.true_poses[index]))
        self.last_metrics = self.episode_metrics()
        return self._observation(), {"map_seed": resolved_seed}

    def step(self, action):
        actions = np.asarray(action, dtype=np.int32)
        if not self.action_space.contains(actions):
            raise ValueError(f"invalid joint action {action}")
        previous_coverage = self._coverage()
        _, previous_observed = self._fuse_beliefs(
            [backend.belief for backend in self.backends]
        )
        previous_logdet = self._total_logdet()
        odometry, collisions = self._execute_actions(actions)
        for index, backend in enumerate(self.backends):
            backend.update(odometry[index], self._ray_cast(self.true_poses[index]))
        self.step_count += 1
        self.collision_count += collisions
        coverage = self._coverage()
        _, current_observed = self._fuse_beliefs([backend.belief for backend in self.backends])
        newly_observed = max(0, np.count_nonzero(current_observed) - np.count_nonzero(previous_observed))
        self.new_observed_cells += int(newly_observed)
        uncertainty_reduction = float(np.clip(previous_logdet - self._total_logdet(), -1.0, 1.0))
        if not newly_observed:
            uncertainty_reduction = min(0.0, uncertainty_reduction)
        team_reward = (
            10.0 * (coverage - previous_coverage)
            + 0.5 * uncertainty_reduction
            - 0.1 * collisions
            - 0.01
        )
        rewards = np.full(self.n_agents, team_reward, dtype=np.float32)
        terminated = bool(coverage >= self.coverage_target)
        truncated = bool(self.step_count >= self.time_limit)
        self.last_metrics = self.episode_metrics()
        info = {**self.last_metrics, "team_reward": team_reward}
        return self._observation(), rewards, terminated, truncated, info

    def episode_metrics(self) -> dict[str, float]:
        if self.ground_truth is None:
            return {}
        beliefs = [backend.belief for backend in self.backends]
        fused, observed = self._fuse_beliefs(beliefs)
        predicted_occupied = fused > 0.0
        evaluation_mask = observed
        intersection = np.count_nonzero(predicted_occupied & self.ground_truth & evaluation_mask)
        union = np.count_nonzero((predicted_occupied | self.ground_truth) & evaluation_mask)
        occupancy_iou = float(intersection / union) if union else 0.0
        position_errors = [
            np.linalg.norm(belief.pose[:2] - truth[:2])
            for belief, truth in zip(beliefs, self.true_poses)
        ]
        total_observed = sum(np.count_nonzero(belief.observed) for belief in beliefs)
        union_observed = max(1, np.count_nonzero(observed))
        return {
            "coverage": self._coverage(),
            "occupancy_iou": occupancy_iou,
            "pose_rmse": float(np.sqrt(np.mean(np.square(position_errors)))),
            "collisions": float(self.collision_count),
            "path_length": float(self.path_length),
            "new_observed_cells": float(self.new_observed_cells),
            "redundant_coverage": float(max(0.0, total_observed / union_observed - 1.0)),
            "pose_uncertainty": float(sum(np.trace(belief.covariance) for belief in beliefs)),
            "success": float(self._coverage() >= self.coverage_target),
        }

    def render(self):
        canvas = np.where(self.ground_truth, "#", " ").astype("<U1")
        for index, pose in enumerate(self.true_poses):
            canvas[int(round(pose[1])), int(round(pose[0]))] = str(index)
        return "\n".join("".join(row) for row in canvas)

def active_slam_make(params: dict) -> MultiAgentSLAMEnv:
    return MultiAgentSLAMEnv(**params)


def episode_metrics_callback(env: MultiAgentSLAMEnv) -> dict[str, float]:
    return env.episode_metrics()
