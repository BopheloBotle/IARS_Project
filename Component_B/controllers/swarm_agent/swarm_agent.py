"""
Webots per-agent controller (Component B).

Roles: Explorer by default; leader elected via max-id flood on ring neighbours.
Motion: simple PSO-inspired attraction to personal/global best hotspot estimate
plus LiDAR-based reactive avoidance (ML-style action selection approximated
with the same expert rules used to label the offline MLP dataset).
"""
from controller import Robot
import math
import struct


HOTSPOTS = [(2.0, 2.0), (-2.5, 1.5), (1.0, -2.5), (-1.5, -1.0)]


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def pick_action(front, left, right, goal_bearing):
    if front < 0.35:
        return 1 if left >= right else 2
    if front < 0.7:
        return 1 if left >= right else 2
    if abs(goal_bearing) < 0.35:
        return 0
    return 1 if goal_bearing > 0 else 2


def main():
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())
    args = robot.getCustomData() if hasattr(robot, "getCustomData") else ""
    # Prefer controllerArgs via sys-like API
    agent_id = 0
    try:
        import sys
        if len(sys.argv) > 1:
            agent_id = int(sys.argv[1])
    except Exception:
        agent_id = 0

    left_m = robot.getDevice("left wheel motor")
    right_m = robot.getDevice("right wheel motor")
    left_m.setPosition(float("inf"))
    right_m.setPosition(float("inf"))
    left_m.setVelocity(0.0)
    right_m.setVelocity(0.0)

    lidar = robot.getDevice("lidar")
    if lidar:
        lidar.enable(timestep)
    gps = robot.getDevice("gps")
    if gps:
        gps.enable(timestep)
    compass = robot.getDevice("compass")
    if compass:
        compass.enable(timestep)
    emitter = robot.getDevice("emitter")
    receiver = robot.getDevice("receiver")
    if receiver:
        receiver.enable(timestep)

    # PSO-like local state
    pbest = None
    pbest_score = float("inf")
    gbest = HOTSPOTS[agent_id % len(HOTSPOTS)]
    known_leader = agent_id
    max_speed = 6.28
    wheel_radius = 0.02
    wheel_base = 0.16

    def hotspot_score(x, y):
        best = 0.0
        for hx, hy in HOTSPOTS:
            d2 = (x - hx) ** 2 + (y - hy) ** 2
            best -= math.exp(-0.5 * d2)
        return best

    print(f"[agent_{agent_id}] started")
    step = 0
    while robot.step(timestep) != -1:
        step += 1
        if gps:
            x, y = gps.getValues()[0], gps.getValues()[1]
        else:
            x, y = 0.0, 0.0
        if compass:
            n = compass.getValues()
            theta = math.atan2(n[0], n[2]) if len(n) >= 3 else 0.0
        else:
            theta = 0.0

        score = hotspot_score(x, y)
        if score < pbest_score:
            pbest_score = score
            pbest = (x, y)

        # Broadcast id + position + pbest score for consensus / leader election
        if emitter and step % 5 == 0:
            msg = struct.pack("ifff", agent_id, x, y, pbest_score)
            emitter.send(msg)

        # Receive neighbours: update leader + gbest
        if receiver:
            while receiver.getQueueLength() > 0:
                data = receiver.getData()
                receiver.nextPacket()
                if len(data) >= 16:
                    oid, ox, oy, oscore = struct.unpack("ifff", bytes(data[:16]))
                    if oid > known_leader:
                        known_leader = oid
                    if oscore < pbest_score - 1e-6:
                        # neighbour found better — adopt as gbest cue
                        gbest = (ox, oy)

        target = gbest if known_leader != agent_id else (pbest if pbest else gbest)
        dx, dy = target[0] - x, target[1] - y
        desired = math.atan2(dy, dx)
        bearing = wrap(desired - theta)

        # LiDAR sectors
        front = left = right = 2.0
        if lidar:
            ranges = list(lidar.getRangeImage() or [])
            n = len(ranges)
            if n:
                def sect(a, b):
                    vals = [ranges[i] for i in range(a, b) if ranges[i] == ranges[i]]
                    return min(vals) if vals else 2.0
                front = sect(n // 2 - 10, n // 2 + 10)
                left = sect(int(0.7 * n), int(0.9 * n))
                right = sect(int(0.1 * n), int(0.3 * n))

        action = pick_action(front, left, right, bearing)
        v, omega = 0.35, 0.0
        if action == 0:
            v, omega = 0.45, 1.2 * bearing
        elif action == 1:
            v, omega = 0.15, 1.8
        elif action == 2:
            v, omega = 0.15, -1.8
        else:
            v, omega = 0.0, 1.5

        wr = (2 * v + omega * wheel_base) / (2 * wheel_radius)
        wl = (2 * v - omega * wheel_base) / (2 * wheel_radius)
        wr = max(-max_speed, min(max_speed, wr))
        wl = max(-max_speed, min(max_speed, wl))
        left_m.setVelocity(wl)
        right_m.setVelocity(wr)

        if step % 200 == 0:
            role = "LEADER" if known_leader == agent_id else "explorer"
            print(f"[agent_{agent_id}] role={role} pos=({x:.2f},{y:.2f}) gbest={gbest}")


if __name__ == "__main__":
    main()
