"""
Swarm Intelligence (Component B, Section 2)

Particle Swarm Optimisation (PSO) for distributed coverage / search:
particles (= agents) minimise a multimodal objective representing
"uncovered risk" over a 2D search area. Emergent flocking toward
promising regions is visualised.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES = os.path.join(RESULTS, "figures")
os.makedirs(FIGURES, exist_ok=True)


def coverage_objective(xy: np.ndarray) -> float:
    """
    Multimodal field: sum of inverted Gaussians (hotspots to visit).
    Lower is better (agents seek minima = high-priority search cells).
    """
    hotspots = np.array([[2.0, 2.0], [-2.5, 1.5], [1.0, -2.5], [-1.5, -1.0]])
    val = 0.0
    for h in hotspots:
        d2 = np.sum((xy - h) ** 2)
        val -= np.exp(-0.5 * d2)  # negative peaks at hotspots
    # Soft bounds penalty
    val += 0.05 * np.sum(np.maximum(np.abs(xy) - 4.0, 0.0) ** 2)
    return float(val)


def pso_coverage(
    n_agents: int = 12,
    steps: int = 60,
    w: float = 0.65,
    c1: float = 1.4,
    c2: float = 1.4,
    seed: int = 7,
) -> dict:
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-3.5, 3.5, size=(n_agents, 2))
    vel = rng.normal(0, 0.2, size=(n_agents, 2))
    pbest = pos.copy()
    pbest_val = np.array([coverage_objective(p) for p in pos])
    gbest_idx = int(np.argmin(pbest_val))
    gbest = pbest[gbest_idx].copy()
    gbest_val = pbest_val[gbest_idx]

    traj = np.zeros((steps + 1, n_agents, 2))
    traj[0] = pos
    gbest_hist = [gbest_val]

    for t in range(steps):
        r1 = rng.random((n_agents, 2))
        r2 = rng.random((n_agents, 2))
        vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
        # Clamp velocity
        speed = np.linalg.norm(vel, axis=1, keepdims=True) + 1e-9
        max_v = 0.55
        vel = np.where(speed > max_v, vel * max_v / speed, vel)
        pos = pos + vel
        pos = np.clip(pos, -4.5, 4.5)

        vals = np.array([coverage_objective(p) for p in pos])
        improved = vals < pbest_val
        pbest[improved] = pos[improved]
        pbest_val[improved] = vals[improved]
        if pbest_val.min() < gbest_val:
            gbest_idx = int(np.argmin(pbest_val))
            gbest = pbest[gbest_idx].copy()
            gbest_val = pbest_val[gbest_idx]

        traj[t + 1] = pos
        gbest_hist.append(gbest_val)

    return {
        "traj": traj,
        "gbest": gbest,
        "gbest_hist": np.array(gbest_hist),
        "final_pos": pos,
        "pbest_val": pbest_val,
    }


def visualise_pso(result: dict) -> str:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Objective field background
    xs = np.linspace(-4.5, 4.5, 120)
    ys = np.linspace(-4.5, 4.5, 120)
    XX, YY = np.meshgrid(xs, ys)
    ZZ = np.zeros_like(XX)
    for i in range(XX.shape[0]):
        for j in range(XX.shape[1]):
            ZZ[i, j] = coverage_objective(np.array([XX[i, j], YY[i, j]]))

    ax = axes[0]
    cs = ax.contourf(XX, YY, ZZ, levels=20, cmap="viridis")
    fig.colorbar(cs, ax=ax, fraction=0.046)
    traj = result["traj"]
    for a in range(traj.shape[1]):
        ax.plot(traj[:, a, 0], traj[:, a, 1], "-", alpha=0.35, linewidth=0.8)
        ax.plot(traj[-1, a, 0], traj[-1, a, 1], "o", markersize=5)
    ax.plot(result["gbest"][0], result["gbest"][1], "r*", markersize=14, label="gbest")
    ax.set_title("PSO agent trajectories on coverage field")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(result["gbest_hist"], color="#d62728", linewidth=2)
    ax.set_title("Global-best objective (convergence)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("gbest value (lower=better)")

    fig.suptitle("Swarm Intelligence: PSO for Cooperative Search/Coverage")
    fig.tight_layout()
    out = os.path.join(FIGURES, "B2_pso_swarm.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    print("=" * 60)
    print("COMPONENT B.2b — Swarm Intelligence (PSO)")
    print("=" * 60)
    result = pso_coverage()
    print(f"Final gbest position: {result['gbest']}")
    print(f"Final gbest value:    {result['gbest_hist'][-1]:.4f}")
    print(f"Initial gbest value:  {result['gbest_hist'][0]:.4f}")
    out = visualise_pso(result)
    np.savez(
        os.path.join(RESULTS, "pso_history.npz"),
        traj=result["traj"],
        gbest_hist=result["gbest_hist"],
        gbest=result["gbest"],
    )
    print(f"Saved: {out}")
    return result


if __name__ == "__main__":
    main()
