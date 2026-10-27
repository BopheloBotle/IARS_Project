"""
Optional Webots supervisor: logs agent broadcasts for offline Matplotlib analysis.
"""
from controller import Supervisor
import struct
import os


def main():
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    receiver = robot.getDevice("receiver")
    if receiver:
        receiver.enable(timestep)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "webots_swarm_log.csv")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("time,agent_id,x,y,score\n")

    print("[supervisor] logging swarm packets to", log_path)
    while robot.step(timestep) != -1:
        t = robot.getTime()
        if not receiver:
            continue
        while receiver.getQueueLength() > 0:
            data = receiver.getData()
            receiver.nextPacket()
            if len(data) >= 16:
                oid, x, y, score = struct.unpack("ifff", bytes(data[:16]))
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(f"{t:.3f},{oid},{x:.4f},{y:.4f},{score:.6f}\n")


if __name__ == "__main__":
    main()
