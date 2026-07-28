"""Non-learning policies used as active-SLAM experimental baselines."""

import numpy as np

from .environments.active_slam import (
    ACTION_FORWARD,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_WAIT,
    MultiAgentSLAMEnv,
    _wrap_angle,
)


class RandomJointPolicy:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def action(self, env: MultiAgentSLAMEnv, observation=None) -> list[int]:
        return self.rng.integers(0, 4, size=env.n_agents).tolist()


class FrontierJointPolicy:
    """Assign each robot a nearby distinct frontier in the fused belief map."""

    @staticmethod
    def _next_frontier_step(
        start: tuple[int, int],
        traversable: np.ndarray,
        frontiers: np.ndarray,
        excluded_targets: set[tuple[int, int]],
    ) -> tuple[np.ndarray, tuple[int, int]]:
        height, width = traversable.shape
        frontier_set = {
            (int(x), int(y))
            for y, x in np.argwhere(frontiers)
            if (int(x), int(y)) not in excluded_targets
        }
        queue = [start]
        parent = {start: None}
        target = None
        for cell in queue:
            if cell in frontier_set:
                target = cell
                break
            x, y = cell
            for neighbor in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                nx, ny = neighbor
                if (
                    0 <= nx < width
                    and 0 <= ny < height
                    and traversable[ny, nx]
                    and neighbor not in parent
                ):
                    parent[neighbor] = cell
                    queue.append(neighbor)
        if target is None:
            return np.asarray(start), start
        step = target
        while parent[step] is not None and parent[step] != start:
            step = parent[step]
        return np.asarray(step), target

    @staticmethod
    def _motion_toward(pose: np.ndarray, target: np.ndarray) -> int:
        delta = target.astype(np.float32) - pose[:2]
        if np.linalg.norm(delta) < 0.75:
            return ACTION_WAIT
        desired = np.arctan2(delta[1], delta[0])
        error = _wrap_angle(desired - float(pose[2]))
        if abs(error) <= np.pi / 4.0:
            return ACTION_FORWARD
        return ACTION_LEFT if error > 0 else ACTION_RIGHT

    def action(self, env: MultiAgentSLAMEnv, observation=None) -> list[int]:
        beliefs = [backend.belief for backend in env.backends]
        fused, observed = env._fuse_beliefs(beliefs)
        frontiers = env._frontiers(fused, observed)
        if not np.any(frontiers):
            return [ACTION_WAIT] * env.n_agents
        traversable = observed & (fused < 0.0)
        assigned = set()
        actions = []
        for belief in beliefs:
            start = tuple(np.rint(belief.pose[:2]).astype(int))
            traversable[start[1], start[0]] = True
            next_cell, target = self._next_frontier_step(
                start, traversable, frontiers, assigned
            )
            assigned.add(target)
            actions.append(self._motion_toward(belief.pose, next_cell))
        return actions


class LearnedJointPolicy:
    """Greedy decentralized policy backed by a trained shared actor."""

    def __init__(self, model):
        self.model = model

    def action(self, env: MultiAgentSLAMEnv, observation=None) -> list[int]:
        if observation is None:
            raise ValueError("learned policy requires the current observation")
        probabilities = self.model(observation["local"], training=False)
        return np.argmax(np.asarray(probabilities), axis=-1).astype(int).tolist()
