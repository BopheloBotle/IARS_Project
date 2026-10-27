"""
Shared 2D occupancy-grid environment used by Component A analysis scripts
and mirrored by the Webots worlds (obstacles at matching coordinates).
"""
from __future__ import annotations

import numpy as np


def build_occupancy_grid(size: int = 40, resolution: float = 0.25) -> dict:
    """
    Build a square occupancy grid with static rectangular obstacles.

    Grid cell value: 0 = free, 1 = occupied.
    World coordinates: metres, origin at grid centre.
    """
    grid = np.zeros((size, size), dtype=np.uint8)

    # Obstacle rectangles in grid indices (row, col) inclusive ranges
    obstacles = [
        (8, 12, 5, 18),    # horizontal wall
        (18, 28, 22, 26),  # vertical block
        (25, 32, 8, 14),   # lower-left block
        (10, 16, 28, 34),  # upper-right block
        (20, 22, 10, 20),  # thin barrier
    ]
    for r0, r1, c0, c1 in obstacles:
        grid[r0 : r1 + 1, c0 : c1 + 1] = 1

    start = (5, 5)   # (row, col)
    goal = (35, 35)

    return {
        "grid": grid,
        "resolution": resolution,
        "size": size,
        "start": start,
        "goal": goal,
        "obstacles": obstacles,
    }


def grid_to_world(row: int, col: int, size: int, resolution: float) -> tuple[float, float]:
    """Convert grid (row, col) to world (x, y) metres, origin at centre."""
    x = (col - size / 2.0) * resolution
    y = (size / 2.0 - row) * resolution
    return x, y


def world_to_grid(x: float, y: float, size: int, resolution: float) -> tuple[int, int]:
    """Convert world (x, y) metres to nearest grid (row, col)."""
    col = int(round(x / resolution + size / 2.0))
    row = int(round(size / 2.0 - y / resolution))
    col = max(0, min(size - 1, col))
    row = max(0, min(size - 1, row))
    return row, col


def neighbours8(row: int, col: int, grid: np.ndarray):
    """Yield free 8-connected neighbours."""
    h, w = grid.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = row + dr, col + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 0:
                yield nr, nc
