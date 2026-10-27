"""
Motion Planning and Pathfinding (Component A, Section 2)

Implements Dijkstra and A* on an occupancy grid and compares:
computational efficiency, path optimality, and nodes explored.
"""
from __future__ import annotations

import heapq
import math
import os
import sys
import time
from typing import Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from shared.environment import build_occupancy_grid, neighbours8

Node = Tuple[int, int]
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES = os.path.join(RESULTS, "figures")
os.makedirs(FIGURES, exist_ok=True)


def _edge_cost(a: Node, b: Node) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def dijkstra(grid: np.ndarray, start: Node, goal: Node):
    """Dijkstra shortest path on an 8-connected occupancy grid."""
    open_heap: List[Tuple[float, Node]] = [(0.0, start)]
    came_from: dict = {}
    g_score = {start: 0.0}
    closed = set()
    nodes_explored = 0

    while open_heap:
        g, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)
        nodes_explored += 1
        if current == goal:
            return _reconstruct(came_from, current), g_score[goal], nodes_explored, closed

        for nb in neighbours8(*current, grid):
            tentative = g_score[current] + _edge_cost(current, nb)
            if tentative < g_score.get(nb, float("inf")):
                came_from[nb] = current
                g_score[nb] = tentative
                heapq.heappush(open_heap, (tentative, nb))

    return None, float("inf"), nodes_explored, closed


def astar(
    grid: np.ndarray,
    start: Node,
    goal: Node,
    heuristic: Callable[[Node, Node], float],
):
    """A* search with a caller-supplied heuristic."""
    open_heap: List[Tuple[float, float, Node]] = [(heuristic(start, goal), 0.0, start)]
    came_from: dict = {}
    g_score = {start: 0.0}
    closed = set()
    nodes_explored = 0

    while open_heap:
        f, g, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)
        nodes_explored += 1
        if current == goal:
            return _reconstruct(came_from, current), g_score[goal], nodes_explored, closed

        for nb in neighbours8(*current, grid):
            tentative = g_score[current] + _edge_cost(current, nb)
            if tentative < g_score.get(nb, float("inf")):
                came_from[nb] = current
                g_score[nb] = tentative
                f_score = tentative + heuristic(nb, goal)
                heapq.heappush(open_heap, (f_score, tentative, nb))

    return None, float("inf"), nodes_explored, closed


def euclidean(a: Node, b: Node) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def manhattan(a: Node, b: Node) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _reconstruct(came_from: dict, current: Node) -> List[Node]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def path_length(path: Optional[List[Node]]) -> float:
    if not path or len(path) < 2:
        return 0.0
    return sum(_edge_cost(path[i], path[i + 1]) for i in range(len(path) - 1))


def compare_algorithms(env: Optional[dict] = None) -> dict:
    """Run Dijkstra and A* (Euclidean + Manhattan) and return metrics."""
    env = env or build_occupancy_grid()
    grid, start, goal = env["grid"], env["start"], env["goal"]

    t0 = time.perf_counter()
    d_path, d_cost, d_nodes, d_closed = dijkstra(grid, start, goal)
    t_dijkstra = time.perf_counter() - t0

    t0 = time.perf_counter()
    a_path, a_cost, a_nodes, a_closed = astar(grid, start, goal, euclidean)
    t_astar_e = time.perf_counter() - t0

    t0 = time.perf_counter()
    am_path, am_cost, am_nodes, am_closed = astar(grid, start, goal, manhattan)
    t_astar_m = time.perf_counter() - t0

    metrics = {
        "Dijkstra": {
            "path": d_path,
            "cost": d_cost,
            "path_length": path_length(d_path),
            "nodes_explored": d_nodes,
            "time_s": t_dijkstra,
            "closed": d_closed,
        },
        "A*_Euclidean": {
            "path": a_path,
            "cost": a_cost,
            "path_length": path_length(a_path),
            "nodes_explored": a_nodes,
            "time_s": t_astar_e,
            "closed": a_closed,
        },
        "A*_Manhattan": {
            "path": am_path,
            "cost": am_cost,
            "path_length": path_length(am_path),
            "nodes_explored": am_nodes,
            "time_s": t_astar_m,
            "closed": am_closed,
        },
    }
    return {"env": env, "metrics": metrics}


def visualise_paths(comparison: dict) -> str:
    """Plot occupancy grid with Dijkstra and A* paths overlaid."""
    env = comparison["env"]
    metrics = comparison["metrics"]
    grid = env["grid"]
    start, goal = env["start"], env["goal"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    titles = ["Dijkstra", "A* (Euclidean)", "A* (Manhattan)"]
    keys = ["Dijkstra", "A*_Euclidean", "A*_Manhattan"]
    colours = ["#1f77b4", "#2ca02c", "#d62728"]

    for ax, title, key, colour in zip(axes, titles, keys, colours):
        m = metrics[key]
        display = np.ma.masked_where(grid == 0, grid)
        ax.imshow(grid, cmap="Greys", origin="upper", alpha=0.35)
        # explored nodes
        explored = np.zeros_like(grid, dtype=float)
        for r, c in m["closed"]:
            explored[r, c] = 1.0
        ax.imshow(np.ma.masked_where(explored == 0, explored), cmap="YlOrRd", origin="upper", alpha=0.45)
        path = m["path"]
        if path:
            rows = [p[0] for p in path]
            cols = [p[1] for p in path]
            ax.plot(cols, rows, "-o", color=colour, markersize=2, linewidth=2, label="path")
        ax.plot(start[1], start[0], "go", markersize=10, label="start")
        ax.plot(goal[1], goal[0], "r*", markersize=12, label="goal")
        ax.set_title(
            f"{title}\nnodes={m['nodes_explored']}, len={m['path_length']:.2f}, t={m['time_s']*1000:.2f} ms"
        )
        ax.set_xlabel("col")
        ax.set_ylabel("row")
        ax.legend(loc="lower right", fontsize=7)

    fig.suptitle("Path Planning Comparison on Occupancy Grid", fontsize=12)
    fig.tight_layout()
    out = os.path.join(FIGURES, "A2_path_planning_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def visualise_metrics_table(comparison: dict) -> str:
    """Bar charts for nodes explored and runtime."""
    metrics = comparison["metrics"]
    names = list(metrics.keys())
    nodes = [metrics[n]["nodes_explored"] for n in names]
    times = [metrics[n]["time_s"] * 1000 for n in names]
    lengths = [metrics[n]["path_length"] for n in names]

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    palette = sns.color_palette("muted", n_colors=3)

    axes[0].bar(names, nodes, color=palette)
    axes[0].set_title("Nodes Explored")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(names, times, color=palette)
    axes[1].set_title("Runtime (ms)")
    axes[1].tick_params(axis="x", rotation=20)

    axes[2].bar(names, lengths, color=palette)
    axes[2].set_title("Path Length (grid units)")
    axes[2].tick_params(axis="x", rotation=20)

    fig.suptitle("Dijkstra vs A*: Efficiency and Optimality")
    fig.tight_layout()
    out = os.path.join(FIGURES, "A2_planning_metrics.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save numeric table
    table_path = os.path.join(RESULTS, "A2_planning_metrics.csv")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("algorithm,path_length,nodes_explored,time_ms,cost\n")
        for n in names:
            m = metrics[n]
            f.write(
                f"{n},{m['path_length']:.4f},{m['nodes_explored']},{m['time_s']*1000:.4f},{m['cost']:.4f}\n"
            )
    return out


def main():
    print("=" * 60)
    print("COMPONENT A.2 — Motion Planning and Pathfinding")
    print("=" * 60)
    comparison = compare_algorithms()
    for name, m in comparison["metrics"].items():
        print(
            f"{name:16s}  path_len={m['path_length']:.3f}  "
            f"nodes={m['nodes_explored']:4d}  time={m['time_s']*1000:.3f} ms"
        )
    p1 = visualise_paths(comparison)
    p2 = visualise_metrics_table(comparison)
    print(f"Saved: {p1}")
    print(f"Saved: {p2}")
    # Prefer A* Euclidean for downstream control (same optimal length, fewer nodes)
    best_path = comparison["metrics"]["A*_Euclidean"]["path"]
    np.save(os.path.join(RESULTS, "planned_path.npy"), np.array(best_path))
    print("Saved planned_path.npy (A* Euclidean)")
    return comparison


if __name__ == "__main__":
    main()
