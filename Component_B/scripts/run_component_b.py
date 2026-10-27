"""
Run all Component B sections end-to-end and regenerate Matplotlib figures.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import multi_agent_setup
import consensus
import swarm_intelligence
import ml_navigation
import ethics_discussion


def main():
    print("\n>>> Running Component B (Setup → Consensus/PSO → ML → Ethics)\n")
    multi_agent_setup.main()
    consensus.main()
    swarm_intelligence.main()
    ml_navigation.main()
    ethics_discussion.main()
    print("\nComponent B complete. Figures in Component_B/results/figures/")


if __name__ == "__main__":
    main()
