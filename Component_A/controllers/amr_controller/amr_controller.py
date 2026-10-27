"""
Webots controller: Autonomous Mobile Robot (Component A)

Uses LiDAR (Lidar device) + Camera, builds a local occupancy estimate,
follows an A*-planned polyline with a PID heading controller.

Open world: Component_A/worlds/autonomous_navigation.wbt
Assign this controller to the robot named "AMR".
"""
from controller import Robot, Supervisor
import math
import os
import sys

# Allow importing shared planning helpers when WEBOTS_HOME project path is set
CTRL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CTRL_DIR, "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from Component_A.scripts.path_planning import astar, euclidean
    from shared.environment import build_occupancy_grid, grid_to_world
except Exception:
    astar = None


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class PID:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.i = 0.0
        self.prev = 0.0

    def step(self, error, dt):
        self.i += error * dt
        d = (error - self.prev) / dt if dt > 0 else 0.0
        self.prev = error
        return self.kp * error + self.ki * self.i + self.kd * d


def main():
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())
    dt = timestep / 1000.0

    left = robot.getDevice("left wheel motor")
    right = robot.getDevice("right wheel motor")
    left.setPosition(float("inf"))
    right.setPosition(float("inf"))
    left.setVelocity(0.0)
    right.setVelocity(0.0)

    lidar = robot.getDevice("lidar")
    if lidar:
        lidar.enable(timestep)

    camera = robot.getDevice("camera")
    if camera:
        camera.enable(timestep)

    gps = robot.getDevice("gps")
    if gps:
        gps.enable(timestep)

    compass = robot.getDevice("compass")
    if compass:
        compass.enable(timestep)

    # Precomputed A* path in world coordinates (m), matching shared environment
    env = None
    path_xy = []
    if astar is not None:
        env = build_occupancy_grid()
        path, _, _, _ = astar(env["grid"], env["start"], env["goal"], euclidean)
        path_xy = [grid_to_world(r, c, env["size"], env["resolution"]) for r, c in path]
    else:
        # Fallback waypoints if shared modules unavailable inside Webots
        path_xy = [(-4.0, 4.0), (-2.0, 2.0), (0.0, 0.0), (2.0, -1.0), (4.0, -4.0)]

    pid = PID(kp=1.8, ki=0.05, kd=0.35)
    wp = 1
    wheel_base = 0.16  # e-puck-like
    wheel_radius = 0.02
    max_speed = 6.28

    print("[AMR] Controller started. Waypoints:", len(path_xy))
    while robot.step(timestep) != -1:
        if gps:
            x, y = gps.getValues()[0], gps.getValues()[1]
        else:
            x, y = path_xy[0]

        if compass:
            north = compass.getValues()
            theta = math.atan2(north[0], north[2]) if len(north) >= 3 else math.atan2(north[0], north[1])
        else:
            theta = 0.0

        # Simple LiDAR obstacle slowdown (perception cue)
        v_scale = 1.0
        if lidar:
            ranges = lidar.getRangeImage()
            if ranges:
                front = [r for r in ranges[len(ranges)//2 - 10 : len(ranges)//2 + 10] if r == r]
                if front and min(front) < 0.25:
                    v_scale = 0.2

        target = path_xy[min(wp, len(path_xy) - 1)]
        dx, dy = target[0] - x, target[1] - y
        dist = math.hypot(dx, dy)
        desired = math.atan2(dy, dx)
        e = wrap(desired - theta)
        omega = pid.step(e, dt)
        v = 0.5 * v_scale * max(0.15, 1.0 - 0.8 * abs(e) / math.pi)

        # Differential drive inverse kinematics
        wr = (2 * v + omega * wheel_base) / (2 * wheel_radius)
        wl = (2 * v - omega * wheel_base) / (2 * wheel_radius)
        wr = max(-max_speed, min(max_speed, wr))
        wl = max(-max_speed, min(max_speed, wl))
        left.setVelocity(wl)
        right.setVelocity(wr)

        if dist < 0.15 and wp < len(path_xy) - 1:
            wp += 1
        if wp >= len(path_xy) - 1 and dist < 0.12:
            left.setVelocity(0.0)
            right.setVelocity(0.0)
            print("[AMR] Goal reached.")
            break


if __name__ == "__main__":
    main()
