"""Demo: a membrane self-assembling from scattered amphiphiles.

Run with:  python examples/membrane_demo.py

Amphiphiles are dropped into the box at random positions and orientations. With
nothing but excluded volume and a hydrophobic attraction between their tails,
they organise themselves into a droplet with the tails packed in a core and the
heads on the surface — a compartment boundary, built by physics. This is the
missing pillar of a cell, to be combined with replicating DNA into a protocell.
"""

import random

from protocell import build_membrane


def main() -> None:
    print("== Self-assembling amphiphile membrane ==")
    membrane = build_membrane(n_amphiphiles=10, spread=6.0, rng=random.Random(1))

    print(f"  amphiphiles     : {membrane.n_amphiphiles} (head + 2 hydrophobic tails each)")
    print(f"  start: tail spread {membrane.tail_gyration():.2f} Å, "
          f"head spread {membrane.head_gyration():.2f} Å")

    for _ in range(6):
        membrane.system.run(steps=2000, dt=0.005)
        print(f"    t={membrane.system.time:6.1f}  "
              f"tail spread {membrane.tail_gyration():5.2f}  "
              f"head spread {membrane.head_gyration():5.2f}  "
              f"hydrophobic core: {membrane.has_hydrophobic_core()}")

    print()
    print(f"  Result: tails packed to {membrane.tail_gyration():.2f} Å inside "
          f"heads at {membrane.head_gyration():.2f} Å —")
    print(f"  a hydrophobic core wrapped in a hydrophilic surface. A compartment.")


if __name__ == "__main__":
    main()
