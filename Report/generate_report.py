"""
Generate the professional PDF report (3–9 pages) for the IARS project.
"""
from __future__ import annotations

import os
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib import colors

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(os.path.dirname(__file__), "IARS_Project_Report.pdf")
A_FIG = os.path.join(ROOT, "Component_A", "results", "figures")
B_FIG = os.path.join(ROOT, "Component_B", "results", "figures")


def P(text, style):
    return Paragraph(text.replace("\n", "<br/>"), style)


def maybe_image(path, width=15 * cm, height=6.5 * cm):
    if os.path.exists(path):
        return Image(path, width=width, height=height)
    return Paragraph(f"[Figure missing: {os.path.basename(path)}]", getSampleStyleSheet()["Normal"])


def build():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCenter", parent=styles["Title"], alignment=TA_CENTER, fontSize=16, spaceAfter=8))
    styles.add(ParagraphStyle(name="SubCenter", parent=styles["Normal"], alignment=TA_CENTER, fontSize=10, spaceAfter=12))
    styles.add(ParagraphStyle(name="BodyJust", parent=styles["Normal"], alignment=TA_JUSTIFY, fontSize=9.5, leading=12, spaceAfter=6))
    styles.add(ParagraphStyle(name="H", parent=styles["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle(name="H3", parent=styles["Heading3"], fontSize=10.5, spaceBefore=8, spaceAfter=4))

    doc = SimpleDocTemplate(
        OUT,
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )
    story = []

    story.append(P("Intelligent Autonomous Robotic Systems", styles["TitleCenter"]))
    story.append(P(
        "Component A: Perception, Motion Planning and Control<br/>"
        "Component B: Multi-Agent Swarm with Learning-Based Decision-Making<br/>"
        "Platform: Webots + Matplotlib / Python (no ROS/Gazebo)",
        styles["SubCenter"],
    ))

    # ---- Introduction ----
    story.append(P("1. Introduction", styles["H"]))
    story.append(P(
        "This project designs, simulates, and evaluates two integrated intelligent robotic systems. "
        "Component A implements a hybrid perception–planning–control stack for a single autonomous "
        "mobile robot: LiDAR and camera sensing are fused into an occupancy map, Dijkstra and A* "
        "compute feasible paths, and a PID controller tracks the planned trajectory on a differential-drive "
        "kinematic model. Component B extends autonomy to a cooperative multi-agent swarm for distributed "
        "area coverage / search, combining average consensus and leader election, Particle Swarm Optimisation "
        "(PSO), and a neural network for local navigation actions. All robot modelling and multi-robot "
        "execution are specified as Webots worlds/controllers; all graphs, maps, trajectories, confusion "
        "matrices, and performance plots are produced with Matplotlib/seaborn. Ethical risks and South "
        "African deployment considerations are analysed critically.",
        styles["BodyJust"],
    ))

    # ---- Methodology ----
    story.append(P("2. Methodology", styles["H"]))
    story.append(P("2.1 Component A — Perception, Planning, Control", styles["H3"]))
    story.append(P(
        "<b>Perception.</b> A 360° LiDAR scan is ray-cast on a 40×40 occupancy grid (0.25 m cells) with "
        "Gaussian range noise and median filtering for spike rejection. A local camera patch is processed "
        "with Gaussian blur, thresholding, Canny edges, and morphological closing (OpenCV). Sensor fusion "
        "uses probabilistic log-odds occupancy updates: LiDAR hits contribute a higher occupied likelihood "
        "than camera detections because ranges are metrically grounded, while camera masks improve coverage "
        "of walls not yet densely swept by LiDAR. Free space along LiDAR rays is decayed with a free-space "
        "log-odds term. The fused map is the environment representation for planning.",
        styles["BodyJust"],
    ))
    story.append(P(
        "<b>Motion planning.</b> The environment is an 8-connected occupancy graph. Dijkstra computes a "
        "shortest path by expanding nodes in order of path cost only. A* uses admissible heuristics "
        "(Euclidean and Manhattan) to prioritise expansions toward the goal. Algorithms are compared on "
        "path length (optimality), nodes explored, and runtime.",
        styles["BodyJust"],
    ))
    story.append(P(
        "<b>Control.</b> The robot is modelled as a differential-drive unicycle: "
        "ẋ = v cos θ, ẏ = v sin θ, θ̇ = ω. A PID on heading error commands ω; linear speed is modulated "
        "by heading error and distance to the active waypoint. Gain sets (underdamped, well-tuned, "
        "aggressive) demonstrate overshoot, rise/settling time, and steady-state error trade-offs.",
        styles["BodyJust"],
    ))
    story.append(P(
        "<b>Integration.</b> Perception, A* planning, and the well-tuned PID form a closed-loop pipeline. "
        "Evaluation reports planned vs actual path length, time-to-goal, and tracking error.",
        styles["BodyJust"],
    ))

    story.append(P("2.2 Component B — Multi-Agent Swarm with Learning", styles["H3"]))
    story.append(P(
        "The Webots world hosts five E-puck agents (≥3 required) with LiDAR, GPS, compass, Emitter/Receiver. "
        "The cooperative task is distributed coverage of multimodal search hotspots. Communication uses a "
        "<b>ring topology</b> (sparse, scalable). <b>Average consensus</b> drives agent states to the mean; "
        "<b>max-id flood</b> elects a leader. <b>PSO</b> assigns particles to agents to minimise a coverage "
        "objective (inverted Gaussian hotspots). An <b>MLPClassifier</b> (32–16 ReLU) maps LiDAR sector "
        "minima and goal bearing (sin/cos) to actions {FORWARD, LEFT, RIGHT, STOP}. Centralised training "
        "on the full dataset is compared with distributed agents trained on bootstrap subsets.",
        styles["BodyJust"],
    ))

    # ---- Results ----
    story.append(P("3. Results and Evaluation", styles["H"]))
    story.append(P("3.1 Component A Results", styles["H3"]))
    story.append(maybe_image(os.path.join(A_FIG, "A1_perception_pipeline.png"), 15.5 * cm, 9.5 * cm))
    story.append(P("Figure 1. Perception pipeline: ground truth, LiDAR, camera processing, fused occupancy.", styles["SubCenter"]))
    story.append(maybe_image(os.path.join(A_FIG, "A2_path_planning_comparison.png"), 15.5 * cm, 5.2 * cm))
    story.append(P("Figure 2. Dijkstra vs A* paths and explored nodes on the occupancy grid.", styles["SubCenter"]))

    # Try to load metrics for a small table
    metrics_csv = os.path.join(ROOT, "Component_A", "results", "A2_planning_metrics.csv")
    if os.path.exists(metrics_csv):
        rows = [line.strip().split(",") for line in open(metrics_csv, encoding="utf-8")]
        t = Table(rows, hAlign="CENTER")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334455")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f7f7f7")),
        ]))
        story.append(t)
        story.append(P("Table 1. Path planning comparison metrics.", styles["SubCenter"]))

    story.append(PageBreak())
    story.append(maybe_image(os.path.join(A_FIG, "A3_pid_tuning.png"), 15.5 * cm, 10.5 * cm))
    story.append(P("Figure 3. PID tuning effects on trajectory and heading/position error.", styles["SubCenter"]))
    story.append(maybe_image(os.path.join(A_FIG, "A4_planned_vs_actual.png"), 15.5 * cm, 6.2 * cm))
    story.append(P("Figure 4. Integrated system: planned vs actual trajectory and performance table.", styles["SubCenter"]))
    story.append(P(
        "Interpretation: A* (Euclidean) matches Dijkstra path length while exploring fewer nodes, "
        "improving replanning efficiency. Well-tuned PID (Kp≈1.8, Ki≈0.05, Kd≈0.35) reduces oscillatory "
        "heading error relative to the aggressive high-Kp/low-Kd case. Residual tracking error arises from "
        "waypoint discretisation and speed reduction near corners — an expected perception–planning–control trade-off.",
        styles["BodyJust"],
    ))

    story.append(P("3.2 Component B Results", styles["H3"]))
    story.append(maybe_image(os.path.join(B_FIG, "B1_multi_agent_setup.png"), 10 * cm, 10 * cm))
    story.append(P("Figure 5. Multi-agent arena schematic (roles, ring links, hotspots).", styles["SubCenter"]))
    story.append(maybe_image(os.path.join(B_FIG, "B2_consensus.png"), 15.5 * cm, 5.5 * cm))
    story.append(P("Figure 6. Average consensus convergence and leader-election flood-max.", styles["SubCenter"]))
    story.append(maybe_image(os.path.join(B_FIG, "B2_pso_swarm.png"), 15.5 * cm, 5.8 * cm))
    story.append(P("Figure 7. PSO trajectories on the coverage field and gbest convergence.", styles["SubCenter"]))
    story.append(PageBreak())
    story.append(maybe_image(os.path.join(B_FIG, "B3_ml_confusion_and_comparison.png"), 15.5 * cm, 5.8 * cm))
    story.append(P("Figure 8. MLP confusion matrix; centralised vs distributed accuracy.", styles["SubCenter"]))
    story.append(P(
        "Consensus states converge to the mean under the ring Laplacian update. PSO agents exhibit "
        "emergent aggregation toward hotspot minima with decreasing gbest objective. The MLP achieves "
        "high action-classification accuracy on the held-out set; navigation rollouts yield a measurable "
        "success rate. Centralised training slightly outperforms distributed bootstrap models, illustrating "
        "the consistency–resilience trade-off discussed ethically below.",
        styles["BodyJust"],
    ))

    # ---- Discussion / Ethics ----
    story.append(P("4. Discussion and Ethical Considerations", styles["H"]))
    story.append(P(
        "<b>System trade-offs.</b> Higher perception recall of obstacles lengthens planned paths; "
        "aggressive PID tracking cuts time-to-goal but risks overshoot near humans. In the swarm, "
        "faster PSO exploitation can cause congestion at a single hotspot, reducing coverage diversity.",
        styles["BodyJust"],
    ))
    story.append(P(
        "<b>Bias and reliability.</b> Simulated LiDAR/camera noise understates dust, glare, and clutter "
        "common in real deployments. Expert-labelled training data transfers rule biases into the MLP. "
        "Over-trusting simulation metrics is itself an ethical hazard (sim-to-real gap).",
        styles["BodyJust"],
    ))
    story.append(P(
        "<b>Explainability.</b> Hidden-layer MLP decisions are not inherently interpretable. A hard "
        "safety envelope (forced avoidance when front range is critical) plus feature/action logging "
        "preserve a minimum audit trail for incident review.",
        styles["BodyJust"],
    ))
    story.append(P(
        "<b>Safety-critical / human-shared risk.</b> False-negative detections, wrong consensus targets, "
        "and identical policy bugs across clones can cause collisions or unsafe swarm behaviour. "
        "Mitigations: speed limits, human override, heartbeats, redundant links beyond a pure ring, "
        "and safe-stop on communications loss.",
        styles["BodyJust"],
    ))
    story.append(P(
        "<b>South African context.</b> POPIA constrains camera-derived personal data (minimisation, "
        "purpose limitation). Intermittent connectivity favours on-robot distributed decisions over "
        "cloud-centralised control. Equity requires that automation in mining, logistics, or security "
        "pairs with reskilling—not silent displacement. Dual-use of search/coverage swarms demands "
        "civilian-benefit framing and refusal to supply operational targeting guidance. Public trust "
        "benefits from accessible explanations beyond English-only technical reports.",
        styles["BodyJust"],
    ))
    story.append(maybe_image(os.path.join(B_FIG, "B4_ethics_risk_matrix.png"), 12 * cm, 8.5 * cm))
    story.append(P("Figure 9. Qualitative risk matrix for responsible autonomy.", styles["SubCenter"]))

    # ---- Conclusion ----
    story.append(P("5. Conclusion and Recommendations", styles["H"]))
    story.append(P(
        "Both mandatory components were implemented within the allowed toolchain (Webots + Matplotlib/Python). "
        "Component A demonstrates a reproducible closed-loop navigation stack with justified fusion, dual "
        "planners, and PID tuning evidence. Component B demonstrates multi-agent coordination (consensus, "
        "leader election), swarm optimisation for coverage, and ML action selection with quantitative "
        "evaluation and ethics analysis. Recommendations: (1) add online map updates from streaming Webots "
        "sensors before physical trials; (2) keep a non-learning safety supervisor above the MLP; "
        "(3) validate POPIA-compliant data handling before any camera pilot; (4) stress-test consensus under "
        "packet loss; (5) treat simulation success as necessary but not sufficient for real-world deployment.",
        styles["BodyJust"],
    ))

    doc.build(story)
    print(f"Wrote {OUT}")
    return OUT


if __name__ == "__main__":
    build()
