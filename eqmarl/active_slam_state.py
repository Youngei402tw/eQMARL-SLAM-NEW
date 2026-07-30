"""Grid-state utilities for the active-SLAM environment."""

from __future__ import annotations

import numpy as np

def largest_free_component(grid: np.ndarray) -> np.ndarray:
    """Return a mask for the largest four-connected free-space component."""
    free = ~grid
    visited = np.zeros_like(free)
    largest = []
    height, width = grid.shape
    for y in range(height):
        for x in range(width):
            if not free[y, x] or visited[y, x]:
                continue
            stack = [(x, y)]
            visited[y, x] = True
            component = []
            while stack:
                cx, cy = stack.pop()
                component.append((cx, cy))
                for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                    if (
                        0 <= nx < width
                        and 0 <= ny < height
                        and free[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        stack.append((nx, ny))
            if len(component) > len(largest):
                largest = component
    mask = np.zeros_like(free)
    for x, y in largest:
        mask[y, x] = True
    return mask
