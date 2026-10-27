"""
Consensus Algorithm (Component B, Section 2)

Average consensus on a fixed communication graph so agents agree on a
shared scalar state (e.g. estimated search priority / meeting point x).
Also includes a simple leader-election (max-id) for role assignment.
"""
from __future__ import annotations

import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES = os.path.join(RESULTS, "figures")
os.makedirs(FIGURES, exist_ok=True)


def ring_adjacency(n: int) -> np.ndarray:
    """Ring communication topology (each agent talks to two neighbours)."""
    A = np.zeros((n, n))
    for i in range(n):
        A[i, (i - 1) % n] = 1
        A[i, (i + 1) % n] = 1
    return A


def average_consensus(
    x0: np.ndarray,
    A: np.ndarray,
    steps: int = 80,
    epsilon: float = 0.35,
) -> np.ndarray:
    """
    Discrete-time average consensus:
        x(k+1) = x(k) + epsilon * A * (neighbour_avg_diff)
    Implemented as: x <- x + eps * A @ (broadcast) style Laplacian update
        x <- x - eps * L @ x
    where L = D - A.
    """
    n = len(x0)
    D = np.diag(A.sum(axis=1))
    L = D - A
    X = np.zeros((steps + 1, n))
    X[0] = x0
    x = x0.copy()
    for k in range(steps):
        x = x - epsilon * (L @ x)
        X[k + 1] = x
    return X


def leader_election(agent_ids: List[int], A: np.ndarray, steps: int = 20) -> dict:
    """
    Flood-max leader election: each agent keeps max(own_id, neighbours' known max).
    """
    n = len(agent_ids)
    known = np.array(agent_ids, dtype=float)
    history = [known.copy()]
    for _ in range(steps):
        nxt = known.copy()
        for i in range(n):
            neigh = known[A[i] > 0]
            if len(neigh):
                nxt[i] = max(known[i], neigh.max())
        known = nxt
        history.append(known.copy())
    leader = int(known[0])  # all equal when converged
    return {"leader_id": leader, "history": np.array(history), "final": known}


def visualise_consensus(X: np.ndarray, leader_hist: np.ndarray) -> str:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    for i in range(X.shape[1]):
        ax.plot(X[:, i], label=f"agent {i}")
    ax.axhline(X[0].mean(), color="k", linestyle="--", linewidth=1, label="true average")
    ax.set_title("Average Consensus Convergence")
    ax.set_xlabel("iteration")
    ax.set_ylabel("state")
    ax.legend(fontsize=7)

    ax = axes[1]
    for i in range(leader_hist.shape[1]):
        ax.step(range(len(leader_hist)), leader_hist[:, i], where="post", label=f"agent {i}")
    ax.set_title("Leader Election (max-id flood)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("known max id")
    ax.legend(fontsize=7)

    fig.suptitle("Multi-Agent Coordination: Consensus & Leader Election")
    fig.tight_layout()
    out = os.path.join(FIGURES, "B2_consensus.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    print("=" * 60)
    print("COMPONENT B.2a — Consensus Algorithms")
    print("=" * 60)
    n = 5
    A = ring_adjacency(n)
    print("Communication topology: ring (each agent ↔ 2 neighbours)")
    x0 = np.array([1.0, 3.5, -2.0, 4.0, 0.5])
    X = average_consensus(x0, A)
    print(f"Initial states: {x0}")
    print(f"Final states:   {X[-1]}")
    print(f"Target average: {x0.mean():.4f}")

    ids = [2, 5, 1, 7, 3]
    elect = leader_election(ids, A)
    print(f"Agent IDs: {ids} -> elected leader: {elect['leader_id']}")

    out = visualise_consensus(X, elect["history"])
    np.savez(os.path.join(RESULTS, "consensus_history.npz"), X=X, leader=elect["history"])
    print(f"Saved: {out}")
    return {"X": X, "leader": elect}


if __name__ == "__main__":
    main()
