"""Grid-SLAM belief backend used by the Active-SLAM environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


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
    """Yield integer grid cells along a segment, including both endpoints."""
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
    """Transparent scan-matching occupancy-grid SLAM backend."""

    def __init__(
        self,
        beam_angles: np.ndarray,
        max_range: float,
        occupied_increment: float = 0.9,
        free_increment: float = -0.35,
        constrain_pose_to_map: bool = False,
    ):
        self.beam_angles = np.asarray(beam_angles, dtype=np.float32)
        self.max_range = float(max_range)
        self.occupied_increment = float(occupied_increment)
        self.free_increment = float(free_increment)
        self.constrain_pose_to_map = bool(constrain_pose_to_map)
        self.belief = None

    @staticmethod
    def _clip_pose(pose: np.ndarray, map_shape: tuple[int, int]) -> np.ndarray:
        height, width = map_shape
        clipped = np.asarray(pose, dtype=np.float32).copy()
        clipped[0] = np.clip(clipped[0], 0.0, width - 1.0)
        clipped[1] = np.clip(clipped[1], 0.0, height - 1.0)
        clipped[2] = _wrap_angle(clipped[2])
        return clipped

    def _pose_is_in_bounds(self, pose: np.ndarray) -> bool:
        height, width = self.belief.occupancy_log_odds.shape
        return bool(0.0 <= pose[0] <= width - 1.0 and 0.0 <= pose[1] <= height - 1.0)

    def reset(self, initial_pose: np.ndarray, map_shape: tuple[int, int]) -> SLAMBelief:
        pose = np.asarray(initial_pose, dtype=np.float32).copy()
        if self.constrain_pose_to_map:
            pose = self._clip_pose(pose, map_shape)
        self.belief = SLAMBelief(
            pose=pose,
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
                    if self.constrain_pose_to_map and not self._pose_is_in_bounds(pose):
                        continue
                    score = self._endpoint_score(pose, scan)
                    regularizer = 0.02 * (abs(dx) + abs(dy) + abs(dtheta))
                    candidates.append((score - regularizer, pose))
        if not candidates:
            pose = self._clip_pose(predicted, self.belief.occupancy_log_odds.shape)
            return pose, self._endpoint_score(pose, scan)
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
        raw_predicted = self.belief.pose + np.asarray(odometry, dtype=np.float32)
        raw_predicted[2] = _wrap_angle(raw_predicted[2])
        predicted = raw_predicted
        boundary_correction = 0.0
        if self.constrain_pose_to_map:
            predicted = self._clip_pose(raw_predicted, self.belief.occupancy_log_odds.shape)
            boundary_correction = float(np.linalg.norm(raw_predicted[:2] - predicted[:2]))
        matched, score = self._scan_match(predicted, lidar_scan)
        self.belief.pose = matched
        previously_observed = int(np.count_nonzero(self.belief.observed))
        self._update_map(matched, lidar_scan)
        new_cells = int(np.count_nonzero(self.belief.observed)) - previously_observed
        process_noise = np.diag([0.025, 0.025, 0.01]).astype(np.float32)
        motion_scale = max(
            0.1,
            float(np.linalg.norm(odometry[:2]))
            + abs(float(odometry[2])) / np.pi
            + boundary_correction,
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
