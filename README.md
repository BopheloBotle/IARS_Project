Intelligent Autonomous Robotic Systems (IARS) Project
=====================================================

Platform constraint (mandatory):
  - Webots for robot modelling, sensors, multi-robot simulation, controllers
  - Matplotlib / seaborn / OpenCV / scikit-learn for algorithms and ALL plots
  - ROS and Gazebo are NOT used and must NOT be used

Contents
--------
IARS_Project/
  Component_A/
    worlds/autonomous_navigation.wbt
    controllers/amr_controller/amr_controller.py
    scripts/   (perception, planning, PID, integration)
    data/      (sensor snapshots)
    results/   (CSV metrics + figures/)
  Component_B/
    worlds/multi_agent_swarm.wbt
    controllers/swarm_agent/swarm_agent.py
    controllers/swarm_supervisor/swarm_supervisor.py
    scripts/   (setup, consensus, PSO, ML, ethics)
    data/      (ML dataset)
    results/   (metrics + figures/)
  shared/environment.py
  Report/IARS_Project_Report.pdf
  Report/generate_report.py
  requirements.txt
  run_all.py
  README.txt

Webots version
--------------
Designed for Webots R2023b (also compatible with R2022b–R2025 if E-puck PROTO
URLs are reachable). Worlds use Cyberbotics official EXTERNPROTO URLs.

Python version
--------------
Python 3.9+ (developed/tested with Python 3.9.6)

Python dependencies
-------------------
numpy
matplotlib
seaborn
opencv-python-headless  (or opencv-python)
scikit-learn
Pillow
reportlab               (only needed to regenerate the PDF report)

Install (recommended: virtual environment):
  cd IARS_Project
  python3 -m venv .venv
  source .venv/bin/activate          # Windows: .venv\Scripts\activate
  pip install -r requirements.txt

How to run — Matplotlib / algorithm pipeline (required plots)
-------------------------------------------------------------
From the IARS_Project folder (with venv activated):

  python run_all.py

Or separately:

  python Component_A/scripts/run_component_a.py
  python Component_B/scripts/run_component_b.py

This regenerates all figures under:
  Component_A/results/figures/
  Component_B/results/figures/

How to run — Webots simulations
-------------------------------
1. Install Webots from https://cyberbotics.com/
2. Component A:
     File → Open World → Component_A/worlds/autonomous_navigation.wbt
     Ensure the controller path resolves to Component_A/controllers
     Press Play. The E-puck "AMR" runs amr_controller (LiDAR+GPS+Compass+PID).
3. Component B:
     File → Open World → Component_B/worlds/multi_agent_swarm.wbt
     Press Play. Five agents run swarm_agent; supervisor logs to
     Component_B/results/webots_swarm_log.csv

If Webots cannot find controllers, set the project directory to Component_A
or Component_B respectively (the folder that contains worlds/ and controllers/).

Regenerate PDF report
---------------------
  python3 Report/generate_report.py

Expected runtime
----------------
  run_all.py (algorithms + plots): typically 30–90 seconds on a laptop CPU
  Webots Component A demo to goal: ~1–3 minutes simulation time
  Webots Component B coverage behaviour: observe 2–5 minutes

Hardware requirements
---------------------
  - CPU: any modern dual-core+
  - RAM: 4 GB minimum (8 GB recommended with Webots)
  - GPU: not required (Webots uses CPU/OpenGL)
  - Disk: < 50 MB for this project package
  - OS: Windows 11 / macOS / Linux (no WSL/dual-boot required)

Assignment mapping
------------------
Component A.1 Perception     → Component_A/scripts/perception_pipeline.py
Component A.2 Planning       → Component_A/scripts/path_planning.py
Component A.3 PID control    → Component_A/scripts/pid_control.py
Component A.4 Integration    → Component_A/scripts/system_integration.py
Component B.1 Multi-agent    → Component_B/scripts/multi_agent_setup.py + worlds/
Component B.2 Consensus/PSO  → consensus.py + swarm_intelligence.py
Component B.3 ML navigation  → ml_navigation.py
Component B.4 Ethics         → ethics_discussion.py + Report PDF Section 4
