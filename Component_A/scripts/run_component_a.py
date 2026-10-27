"""
Run all Component A sections end-to-end and regenerate Matplotlib figures.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import perception_pipeline
import path_planning
import pid_control
import system_integration


def main():
    print("\n>>> Running Component A (Perception → Planning → Control → Integration)\n")
    perception_pipeline.main()
    path_planning.main()
    pid_control.main()
    system_integration.main()
    print("\nComponent A complete. Figures in Component_A/results/figures/")


if __name__ == "__main__":
    main()
