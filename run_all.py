"""
Master runner: executes Component A and Component B Python/Matplotlib pipelines.
Webots worlds are opened separately in the Webots GUI (see README.txt).
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "Component_A", "scripts"))
sys.path.insert(0, os.path.join(ROOT, "Component_B", "scripts"))


def main():
    print("=" * 70)
    print("IARS PROJECT — Full Matplotlib / Algorithm Pipeline")
    print("=" * 70)

    import run_component_a
    import run_component_b

    run_component_a.main()
    run_component_b.main()

    print("\nAll algorithm sections finished.")
    print("Open Webots worlds for physical robot simulation (README.txt).")


if __name__ == "__main__":
    main()
