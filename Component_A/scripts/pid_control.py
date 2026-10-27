"""
Control System Design (Component A, Section 3)

Differential-drive kinematic model + PID path tracking.
Demonstrates effect of gain tuning on overshoot, rise time,
settling time, and steady-state error.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from shared.environment import build_occupancy_grid, grid_to_world

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES = os.path.join(RESULTS, "figures")
os.makedirs(FIGURES, exist_ok=True)


# ---------------------------------------------------------------------------
# Kinematic model (differential drive)
# ---------------------------------------------------------------------------
# State: [x, y, theta]
# Inputs: v (linear m/s), omega (angular rad/s)
#
#   x_dot     = v * cos(theta)
#   y_dot     = v * sin(theta)
#   theta_dot = omega
#
# Wheel speeds (optional dynamics mapping):
#   v = (v_r + v_l) / 2
#   omega = (v_r - v_l) / L
# where L is axle length. For this assignment we control (v, omega) directly.


@dataclass
class PIDGains:
    kp: float
    ki: float
    kd: float
    name: str = ""


class PID:
    def __init__(self, gains: PIDGains, u_min: float = -10.0, u_max: float = 10.0):
        self.gains = gains
        self.integral = 0.0
        self.prev_error = 0.0
        self.u_min = u_min
        self.u_max = u_max

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def step(self, error: float, dt: float) -> float:
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        u = self.gains.kp * error + self.gains.ki * self.integral + self.gains.kd * derivative
        self.prev_error = error
        return float(np.clip(u, self.u_min, self.u_max))


def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def simulate_tracking(
    path_xy: np.ndarray,
    gains_heading: PIDGains,
    gains_speed: PIDGains,
    dt: float = 0.05,
    v_nom: float = 0.6,
    max_steps: int = 4000,
) -> dict:
    """
    Closed-loop path tracking with separate heading and speed PIDs.
    Heading PID drives omega from heading error to the current waypoint.
    Speed PID drives v from along-track progress / distance error.
    """
    x, y, theta = path_xy[0, 0], path_xy[0, 1], 0.0
    # Initial heading toward second waypoint if available
    if len(path_xy) > 1:
        theta = np.arctan2(path_xy[1, 1] - y, path_xy[1, 0] - x)

    pid_h = PID(gains_heading, u_min=-2.5, u_max=2.5)
    pid_v = PID(gains_speed, u_min=0.0, u_max=1.2)

    traj = []
    heading_err = []
    pos_err = []
    times = []
    wp_idx = 1
    t = 0.0

    for _ in range(max_steps):
        target = path_xy[min(wp_idx, len(path_xy) - 1)]
        dx, dy = target[0] - x, target[1] - y
        dist = np.hypot(dx, dy)
        desired_heading = np.arctan2(dy, dx)
        e_theta = wrap_angle(desired_heading - theta)
        e_pos = dist

        omega = pid_h.step(e_theta, dt)
        # Slow down when heading error is large (stability)
        speed_ref = v_nom * max(0.15, 1.0 - 0.8 * abs(e_theta) / np.pi)
        v = pid_v.step(speed_ref - 0.0, dt)  # regulate toward speed_ref via error=speed_ref
        # Simpler: use speed_ref directly modulated, keep speed PID on distance for last segment
        if wp_idx >= len(path_xy) - 1:
            v = pid_v.step(dist, dt)
        else:
            v = speed_ref

        # Integrate kinematics
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        theta = wrap_angle(theta + omega * dt)

        traj.append((x, y, theta))
        heading_err.append(e_theta)
        pos_err.append(e_pos)
        times.append(t)
        t += dt

        if dist < 0.15 and wp_idx < len(path_xy) - 1:
            wp_idx += 1
        if wp_idx >= len(path_xy) - 1 and dist < 0.12:
            break

    traj = np.array(traj)
    return {
        "traj": traj,
        "heading_err": np.array(heading_err),
        "pos_err": np.array(pos_err),
        "times": np.array(times),
        "gains_heading": gains_heading,
        "gains_speed": gains_speed,
    }


def step_response_metrics(error: np.ndarray, times: np.ndarray, band: float = 0.1) -> dict:
    """
    Metrics on heading-error time series (radians).
    Rise: first time |e| falls below 10% of the early peak |e|.
    Settling: last time |e| exceeds absolute band (default 0.1 rad).
    Overshoot: fraction of peak opposite-sign excursion vs early peak magnitude.
    SSE: mean |e| over the final 5% of the record.
    """
    abs_e = np.abs(error)
    early = abs_e[: max(5, len(abs_e) // 10)]
    e_peak = float(early.max()) + 1e-9

    below10 = np.where(abs_e <= 0.1 * e_peak)[0]
    rise_time = float(times[below10[0]]) if len(below10) else None

    outside = np.where(abs_e > band)[0]
    settling = float(times[outside[-1]]) if len(outside) else float(times[0])

    # Opposite-sign peak relative to the sign of the early dominant error
    early_signed = error[: len(early)]
    dominant_sign = 1.0 if early_signed[np.argmax(np.abs(early_signed))] >= 0 else -1.0
    opposite = error * (-dominant_sign)
    overshoot = float(max(0.0, opposite.max()) / e_peak)

    tail = abs_e[-max(5, len(abs_e) // 20) :]
    sse = float(tail.mean())
    return {
        "rise_time": rise_time,
        "settling_time": settling,
        "overshoot_ratio": overshoot,
        "steady_state_error": sse,
        "rms_error": float(np.sqrt(np.mean(error ** 2))),
        "max_abs_error": float(abs_e.max()),
    }


def planned_path_xy_from_grid(path_rc: List[Tuple[int, int]], env: dict) -> np.ndarray:
    size, res = env["size"], env["resolution"]
    return np.array([grid_to_world(r, c, size, res) for r, c in path_rc])


def run_tuning_study(path_xy: np.ndarray) -> dict:
    """Compare underdamped / well-tuned / overdamped heading PID settings."""
    configs = [
        PIDGains(0.6, 0.0, 0.05, "Underdamped (low Kd)"),
        PIDGains(1.8, 0.05, 0.35, "Well-tuned"),
        PIDGains(4.5, 0.2, 0.02, "Aggressive / oscillatory"),
    ]
    speed = PIDGains(1.0, 0.0, 0.0, "speed")
    results = []
    for g in configs:
        sim = simulate_tracking(path_xy, g, speed)
        metrics = step_response_metrics(sim["heading_err"], sim["times"])
        results.append({"sim": sim, "metrics": metrics, "name": g.name})
    return {"configs": results, "path_xy": path_xy}


def visualise_tuning(study: dict) -> str:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    ax = axes[0, 0]
    path = study["path_xy"]
    ax.plot(path[:, 0], path[:, 1], "k--", linewidth=1.5, label="planned")
    for cfg in study["configs"]:
        traj = cfg["sim"]["traj"]
        ax.plot(traj[:, 0], traj[:, 1], label=cfg["name"], linewidth=1.5)
    ax.set_aspect("equal")
    ax.set_title("Trajectory under different PID gains")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    for cfg in study["configs"]:
        ax.plot(cfg["sim"]["times"], cfg["sim"]["heading_err"], label=cfg["name"])
    ax.set_title("Heading error vs time")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("e_theta [rad]")
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    for cfg in study["configs"]:
        ax.plot(cfg["sim"]["times"], cfg["sim"]["pos_err"], label=cfg["name"])
    ax.set_title("Position (distance-to-waypoint) error vs time")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("e_pos [m]")
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    ax.axis("off")
    lines = ["PID Tuning Metrics (heading error)\n"]
    for cfg in study["configs"]:
        m = cfg["metrics"]
        lines.append(f"{cfg['name']}:")
        lines.append(
            f"  rise={m['rise_time']}, settle={m['settling_time']:.2f}s, "
            f"OS={m['overshoot_ratio']:.2f}, SSE={m['steady_state_error']:.4f}"
        )
        lines.append("")
    lines.append(
        "Stability note: excessive Kp with low Kd produces oscillatory\n"
        "heading error (aggressive case). Adding derivative damping and\n"
        "moderate Kp yields faster settling with little overshoot."
    )
    ax.text(0.02, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=8)

    fig.suptitle("PID Path-Tracking: Effect of Gain Tuning", fontsize=12)
    fig.tight_layout()
    out = os.path.join(FIGURES, "A3_pid_tuning.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    csv_path = os.path.join(RESULTS, "A3_pid_metrics.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(
            "config,rise_time,settling_time,overshoot_ratio,"
            "steady_state_error,rms_error,max_abs_error\n"
        )
        for cfg in study["configs"]:
            m = cfg["metrics"]
            f.write(
                f"{cfg['name']},{m['rise_time']},{m['settling_time']:.4f},"
                f"{m['overshoot_ratio']:.4f},{m['steady_state_error']:.6f},"
                f"{m['rms_error']:.6f},{m['max_abs_error']:.6f}\n"
            )
    return out


def main():
    print("=" * 60)
    print("COMPONENT A.3 — Control System Design (PID)")
    print("=" * 60)
    print(
        "Kinematic model (differential drive):\n"
        "  x_dot = v cos(theta)\n"
        "  y_dot = v sin(theta)\n"
        "  theta_dot = omega\n"
    )

    # Load planned path if available, else plan quickly
    path_file = os.path.join(RESULTS, "planned_path.npy")
    env = build_occupancy_grid()
    if os.path.exists(path_file):
        path_rc = [tuple(p) for p in np.load(path_file)]
    else:
        from path_planning import compare_algorithms

        comparison = compare_algorithms(env)
        path_rc = comparison["metrics"]["A*_Euclidean"]["path"]

    path_xy = planned_path_xy_from_grid(path_rc, env)
    study = run_tuning_study(path_xy)
    out = visualise_tuning(study)
    for cfg in study["configs"]:
        m = cfg["metrics"]
        print(
            f"{cfg['name']:28s} settle={m['settling_time']:.2f}s  "
            f"OS={m['overshoot_ratio']:.2f}  SSE={m['steady_state_error']:.4f}"
        )
    print(f"Saved: {out}")

    # Persist well-tuned trajectory for integration section
    well = next(c for c in study["configs"] if "Well" in c["name"])
    np.save(os.path.join(RESULTS, "actual_trajectory.npy"), well["sim"]["traj"])
    return study


if __name__ == "__main__":
    main()
