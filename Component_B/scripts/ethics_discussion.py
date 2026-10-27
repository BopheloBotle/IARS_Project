"""
Ethics, Risk, and Responsible Autonomy (Component B, Section 4)

Writes the critical discussion text used in the PDF report and a short
summary figure (risk matrix style) for the results folder.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES = os.path.join(RESULTS, "figures")
os.makedirs(FIGURES, exist_ok=True)

ETHICS_TEXT = """
ETHICS, RISK, AND RESPONSIBLE AUTONOMY
======================================

1. Bias and reliability risks in sensor data, simulation assumptions, and learned models
---------------------------------------------------------------------------------------
LiDAR and camera simulations used here assume relatively clean geometry, Gaussian
range noise, and lighting-invariant greyscale patches. Real South African outdoor
and informal-settlement environments introduce dust, glare, non-Lambertian surfaces,
and clutter that violate these assumptions. Datasets labelled by a rule-based expert
policy encode that expert's biases (e.g., always prefer left when tied); a neural
network will reproduce those preferences and may systematically fail on under-
represented edge cases. Simulation-to-reality gap is therefore an ethical risk:
over-trusting high simulation accuracy can understate real harm potential.

2. Transparency and explainability challenges in neural-network decision-making
------------------------------------------------------------------------------
The MLP maps sector ranges and goal bearing to discrete actions. Intermediate
hidden units do not correspond to human-interpretable rules. In incident review
(collision, near-miss), stakeholders need causal accounts ("why turn left?").
Pure network outputs complicate accountability. Mitigations used conceptually in
this project: (i) keep a rule-based safety envelope (hard STOP if front < threshold)
overriding the network; (ii) log features + predicted action for post-hoc audit;
(iii) prefer simple models when performance is comparable.

3. Risks of autonomous decision-making in safety-critical / human-shared spaces
------------------------------------------------------------------------------
Even cooperative coverage/search can intersect with pedestrians, workers, or
emergency responders. Failure modes include: false-negative obstacle detections,
consensus on a wrong rendezvous point, swarm congestion at a hotspot, and
centralised planner single points of failure. High-speed or heavy platforms raise
injury severity. Ethical deployment requires speed limits, human override, clear
external signalling, and conservative planning near people — even if that sacrifices
PSO optimality or shortest-path length.

4. Fault tolerance, robustness, and failure-mode mitigation for multi-agent systems
---------------------------------------------------------------------------------
Ring consensus degrades if a link fails; leader election can be re-run periodically.
Mitigations: redundant topology (add a few long-range links), heartbeat timeouts,
local safe-stop when communications drop, and diversity in agent policies so a
shared model bug does not identically fail all robots. Distributed ML training
(shown with lower but non-zero accuracy) improves resilience to central server loss
at the cost of consistency.

5. Responsible AI and robotics deployment in the South African context
---------------------------------------------------------------------
South Africa's digital and robotics landscape spans world-class labs and under-
resourced public services. Deployment considerations include:
  • Equity: automation in mining, warehousing, and security must not silently
    displace workers without reskilling pathways aligned with national skills policy.
  • POPIA: camera-based perception in public/workplace settings processes personal
    information; data minimisation, purpose limitation, and secure retention apply.
  • Infrastructure reality: intermittent connectivity favours on-robot (distributed)
    decision-making over cloud-centralised control.
  • Public trust and language access: explainability materials and incident reporting
    should be usable beyond English-only technical audiences where systems affect
    communities.
  • Dual-use caution: coverage/search swarms have policing and military dual-use;
    civilian research releases should avoid providing operational targeting playbooks
    and should emphasise safety, consent, and civilian benefit (e.g., disaster SAR).

Responsible autonomy here means measurable safety envelopes, human accountability,
privacy compliance under POPIA, and honest communication of simulation limits
before any real-world pilot.
"""


def visualise_risk_matrix() -> str:
    risks = [
        ("Sim-to-real sensor gap", 4, 4),
        ("NN opacity", 3, 3),
        ("Human collision", 5, 5),
        ("Comms/leader failure", 3, 4),
        ("POPIA / privacy", 4, 3),
        ("Workforce displacement", 3, 3),
    ]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0.5, 5.5)
    ax.set_xlabel("Likelihood (1-5)")
    ax.set_ylabel("Impact (1-5)")
    ax.set_title("Responsible Autonomy — Qualitative Risk Matrix")
    ax.axhspan(3.5, 5.5, xmin=0.6, alpha=0.08, color="red")
    ax.axvspan(3.5, 5.5, ymin=0.6, alpha=0.05, color="red")
    for name, L, I in risks:
        ax.scatter(L, I, s=120)
        ax.text(L + 0.08, I + 0.08, name, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(FIGURES, "B4_ethics_risk_matrix.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    print("=" * 60)
    print("COMPONENT B.4 — Ethics, Risk, Responsible Autonomy")
    print("=" * 60)
    print(ETHICS_TEXT)
    path = os.path.join(RESULTS, "B4_ethics_discussion.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(ETHICS_TEXT)
    out = visualise_risk_matrix()
    print(f"Saved text: {path}")
    print(f"Saved figure: {out}")
    return ETHICS_TEXT


if __name__ == "__main__":
    main()
