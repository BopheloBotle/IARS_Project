"""
Multi-Agent Simulation Setup (Component B, Section 1)

Documents agent roles, communication topology, cooperative task objective,
and environment/controller design choices for the Webots swarm world.
Also emits a Matplotlib schematic of the multi-robot arena.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES = os.path.join(RESULTS, "figures")
os.makedirs(FIGURES, exist_ok=True)

SETUP_JUSTIFICATION = """
MULTI-AGENT ENVIRONMENT & CONTROLLER DESIGN JUSTIFICATION
---------------------------------------------------------
Platform: Webots world `multi_agent_swarm.wbt` with >=3 E-puck agents.

Cooperative task: Distributed area coverage / cooperative search.
  Agents must collectively visit high-priority search hotspots (PSO objective)
  while avoiding collisions. A consensus layer agrees on a shared priority
  scalar; a leader elected via max-id flood can broadcast a rendezvous cue.

Agent roles:
  • Explorer (default): runs local PSO update + LiDAR avoidance.
  • Leader (elected): same motion policy, plus broadcasts consensus target.
  • Relay: maintains ring-neighbour message forwarding (simulated in scripts;
    in Webots via Emitter/Receiver on a common channel).

Communication topology: Ring graph (each agent ↔ two neighbours).
  Chosen because it is sparse, scalable, and sufficient for average consensus
  convergence while limiting bandwidth — realistic for low-cost robots.

Controller structure:
  • swarm_agent: per-robot controller (sensing, local PSO particle update,
    ML action selection for obstacle negotiation, motor commands).
  • swarm_supervisor (optional): logs global metrics / hotspot coverage.

Environment design: bounded 10x10 m arena with four cardboard-box obstacles
mirroring search clutter. Hotspots are virtual (objective field), not physical
props, so coverage quality is measurable from agent trajectories alone.
"""


def visualise_arena() -> str:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")
    ax.add_patch(patches.Rectangle((-5, -5), 10, 10, fill=False, linewidth=2))

    obstacles = [(-2.0, -1.0, 1.0, 1.0), (1.5, 1.5, 1.2, 0.8), (0.0, -2.5, 1.5, 0.6)]
    for x, y, w, h in obstacles:
        ax.add_patch(patches.Rectangle((x, y), w, h, color="#777777", alpha=0.7))

    starts = [(-3.5, -3.5), (0.0, -3.5), (3.5, -3.5), (-3.5, 0.0), (3.5, 0.0)]
    for i, (x, y) in enumerate(starts):
        ax.plot(x, y, "o", markersize=10, label=f"agent {i}")
        ax.text(x + 0.15, y + 0.15, f"A{i}")

    # Ring topology edges
    for i in range(len(starts)):
        a = starts[i]
        b = starts[(i + 1) % len(starts)]
        ax.annotate(
            "",
            xy=b,
            xytext=a,
            arrowprops=dict(arrowstyle="<->", color="#1f77b4", lw=1),
        )

    hotspots = [(2.0, 2.0), (-2.5, 1.5), (1.0, -2.5), (-1.5, -1.0)]
    for hx, hy in hotspots:
        ax.plot(hx, hy, "r*", markersize=14)
    ax.plot([], [], "r*", label="search hotspot")

    ax.set_title("Component B Arena: Agents, Ring Comms, Hotspots")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIGURES, "B1_multi_agent_setup.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    print("=" * 60)
    print("COMPONENT B.1 — Multi-Agent Simulation Setup")
    print("=" * 60)
    print(SETUP_JUSTIFICATION)
    out = visualise_arena()
    print(f"Saved: {out}")
    with open(os.path.join(RESULTS, "B1_setup_notes.txt"), "w", encoding="utf-8") as f:
        f.write(SETUP_JUSTIFICATION)
    return out


if __name__ == "__main__":
    main()
