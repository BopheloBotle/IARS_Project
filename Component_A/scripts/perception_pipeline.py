"""
Sensor Simulation and Perception Pipeline (Component A, Section 1)

- Simulates LiDAR range scans and a camera image of the occupancy environment
- Applies OpenCV image processing to detect obstacles / features
- Fuses LiDAR + camera into a unified occupancy representation
- Justifies preprocessing and fusion choices in printed markdown-style comments
"""
from __future__ import annotations

import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from shared.environment import build_occupancy_grid, grid_to_world, world_to_grid

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES = os.path.join(RESULTS, "figures")
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(FIGURES, exist_ok=True)
os.makedirs(DATA, exist_ok=True)


# ---------------------------------------------------------------------------
# Design justification (preprocessing & fusion)
# ---------------------------------------------------------------------------
JUSTIFICATION = """
PREPROCESSING & SENSOR FUSION DESIGN JUSTIFICATION
--------------------------------------------------
1. LiDAR simulation: A 360-degree polar scan (1-degree resolution) is ray-cast
   through the known occupancy grid. Range readings are clipped at max_range
   and corrupted with small Gaussian noise to model real sensor uncertainty.
   Preprocessing: median filter (kernel=3) removes isolated spike outliers
   without blurring true obstacle edges — preferred over a mean filter because
   range spikes are impulsive, not Gaussian-dominant.

2. Camera simulation: A top-down greyscale "camera" view of a local window
   around the robot is rendered from the grid. OpenCV Canny edge detection
   then finds obstacle boundaries; morphological closing fills small gaps.
   Thresholding after Gaussian blur reduces texture noise before Canny.
   This yields binary obstacle masks suitable for occupancy updates.

3. Sensor fusion: Occupancy Grid Mapping with probabilistic log-odds update.
   - LiDAR provides accurate metric ranges around the robot (good for close
     geometry) but is sparse angularly and blind to appearance cues.
   - Camera provides dense 2D obstacle evidence in the field of view but is
     sensitive to lighting/noise and lacks direct range.
   Fusion rule: each cell's log-odds is updated from BOTH modalities when
   available. LiDAR hits use a higher occupied likelihood (l_occ_lidar) than
   camera detections (l_occ_camera) because range geometry is metrically
   grounded. Camera-only cells still accumulate evidence, improving coverage
   of walls not yet swept densely by LiDAR. Free-space cells along LiDAR rays
   are decayed with l_free. Final map = sigmoid(log-odds) thresholded at 0.5.

This produces a unified environment representation that is more complete than
either sensor alone and is the map used by Dijkstra / A* in Section 2.
"""


def simulate_lidar(
    grid: np.ndarray,
    robot_rc: tuple,
    max_range: float = 8.0,
    resolution: float = 0.25,
    noise_std: float = 0.03,
) -> dict:
    """Ray-cast 360 LiDAR ranges from robot grid pose."""
    size = grid.shape[0]
    rx, ry = grid_to_world(robot_rc[0], robot_rc[1], size, resolution)
    angles = np.deg2rad(np.arange(0, 360, 1.0))
    ranges = np.full(len(angles), max_range, dtype=float)

    for i, theta in enumerate(angles):
        for d in np.arange(resolution, max_range, resolution):
            x = rx + d * np.cos(theta)
            y = ry + d * np.sin(theta)
            r, c = world_to_grid(x, y, size, resolution)
            if grid[r, c] == 1:
                ranges[i] = d + np.random.normal(0.0, noise_std)
                break

    # Median filter for spike rejection
    ranges_f = cv2.medianBlur(ranges.astype(np.float32).reshape(1, -1), 3).ravel()
    return {"angles": angles, "ranges": ranges_f, "robot_xy": (rx, ry)}


def simulate_camera_view(
    grid: np.ndarray,
    robot_rc: tuple,
    window: int = 21,
) -> dict:
    """
    Render a local greyscale camera patch and detect obstacles with OpenCV.
    """
    h, w = grid.shape
    r0, c0 = robot_rc
    half = window // 2
    patch = np.ones((window, window), dtype=np.uint8) * 255  # free = white
    for i in range(window):
        for j in range(window):
            rr, cc = r0 - half + i, c0 - half + j
            if 0 <= rr < h and 0 <= cc < w:
                if grid[rr, cc] == 1:
                    patch[i, j] = 40  # occupied = dark
            else:
                patch[i, j] = 0  # out of bounds

    # Add mild sensor noise
    noise = np.random.normal(0, 8, patch.shape).astype(np.int16)
    noisy = np.clip(patch.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Preprocess: blur -> threshold -> Canny -> morphological close
    blurred = cv2.GaussianBlur(noisy, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return {
        "raw": noisy,
        "binary": binary,
        "edges": edges,
        "obstacle_mask": closed,
        "window": window,
        "robot_rc": robot_rc,
    }


def fuse_sensors(
    grid_shape: tuple,
    lidar: dict,
    camera: dict,
    resolution: float = 0.25,
    max_range: float = 8.0,
) -> np.ndarray:
    """
    Probabilistic log-odds fusion of LiDAR rays and camera obstacle mask
    into a unified occupancy probability map in [0, 1].
    """
    size = grid_shape[0]
    log_odds = np.zeros((size, size), dtype=float)

    l_occ_lidar = np.log(0.85 / 0.15)
    l_occ_camera = np.log(0.70 / 0.30)
    l_free = np.log(0.30 / 0.70)

    rx, ry = lidar["robot_xy"]

    # LiDAR updates
    for theta, rng in zip(lidar["angles"], lidar["ranges"]):
        hit = rng < max_range - 1e-3
        steps = np.arange(resolution, min(rng, max_range), resolution)
        for d in steps[:-1] if len(steps) > 1 else []:
            x = rx + d * np.cos(theta)
            y = ry + d * np.sin(theta)
            r, c = world_to_grid(x, y, size, resolution)
            log_odds[r, c] += l_free
        if hit:
            x = rx + rng * np.cos(theta)
            y = ry + rng * np.sin(theta)
            r, c = world_to_grid(x, y, size, resolution)
            log_odds[r, c] += l_occ_lidar

    # Camera updates (local window)
    r0, c0 = camera["robot_rc"]
    half = camera["window"] // 2
    mask = camera["obstacle_mask"]
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] > 0:
                rr, cc = r0 - half + i, c0 - half + j
                if 0 <= rr < size and 0 <= cc < size:
                    log_odds[rr, cc] += l_occ_camera

    # Convert log-odds -> probability
    prob = 1.0 / (1.0 + np.exp(-np.clip(log_odds, -20, 20)))
    return prob


def run_perception(robot_rc=(5, 5)) -> dict:
    env = build_occupancy_grid()
    grid = env["grid"]
    resolution = env["resolution"]

    lidar = simulate_lidar(grid, robot_rc, resolution=resolution)
    camera = simulate_camera_view(grid, robot_rc)
    fused_prob = fuse_sensors(grid.shape, lidar, camera, resolution=resolution)
    fused_binary = (fused_prob >= 0.5).astype(np.uint8)

    # Persist sensor data for reproducibility
    np.savez(
        os.path.join(DATA, "sensor_snapshot.npz"),
        lidar_angles=lidar["angles"],
        lidar_ranges=lidar["ranges"],
        camera_raw=camera["raw"],
        camera_mask=camera["obstacle_mask"],
        fused_prob=fused_prob,
        fused_binary=fused_binary,
        ground_truth=grid,
        robot_rc=np.array(robot_rc),
    )
    return {
        "env": env,
        "lidar": lidar,
        "camera": camera,
        "fused_prob": fused_prob,
        "fused_binary": fused_binary,
    }


def visualise_perception(result: dict) -> str:
    lidar = result["lidar"]
    camera = result["camera"]
    fused = result["fused_prob"]
    gt = result["env"]["grid"]

    fig = plt.figure(figsize=(12, 8))

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(gt, cmap="Greys", origin="upper")
    ax1.set_title("Ground-truth occupancy")
    ax1.plot(result["camera"]["robot_rc"][1], result["camera"]["robot_rc"][0], "go")

    ax2 = fig.add_subplot(2, 3, 2, projection="polar")
    ax2.plot(lidar["angles"], lidar["ranges"], color="#1f77b4", linewidth=0.8)
    ax2.set_title("Simulated LiDAR scan", pad=20)
    ax2.set_theta_zero_location("E")

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(camera["raw"], cmap="gray")
    ax3.set_title("Camera patch (noisy)")

    ax4 = fig.add_subplot(2, 3, 4)
    ax4.imshow(camera["edges"], cmap="gray")
    ax4.set_title("OpenCV Canny edges")

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(camera["obstacle_mask"], cmap="gray")
    ax5.set_title("Obstacle mask (morph close)")

    ax6 = fig.add_subplot(2, 3, 6)
    im = ax6.imshow(fused, cmap="hot", origin="upper", vmin=0, vmax=1)
    ax6.set_title("Fused occupancy (LiDAR+Camera)")
    fig.colorbar(im, ax=ax6, fraction=0.046)

    fig.suptitle("Perception Pipeline: Sensors → Processing → Fusion", fontsize=12)
    fig.tight_layout()
    out = os.path.join(FIGURES, "A1_perception_pipeline.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    print("=" * 60)
    print("COMPONENT A.1 — Sensor Simulation and Perception Pipeline")
    print("=" * 60)
    print(JUSTIFICATION)
    result = run_perception()
    out = visualise_perception(result)
    occupied = int((result["fused_binary"] == 1).sum())
    print(f"Fused map occupied cells: {occupied}")
    print(f"Saved sensor data -> Component_A/data/sensor_snapshot.npz")
    print(f"Saved figure -> {out}")
    return result


if __name__ == "__main__":
    main()
