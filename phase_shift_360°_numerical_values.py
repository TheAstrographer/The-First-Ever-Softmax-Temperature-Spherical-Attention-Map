from __future__ import annotations
import math
from typing import List, Dict, Tuple

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
TAU: float = 2.0 * math.pi
DELTA: float = 0.0051676          # seventh-multiple residual δ

# Steered logits
LOGITS: List[float] = [3.2, 2.7, 1.9, 0.8, -3.5]
TOKENS: List[str]  = ["tok_0", "tok_1", "tok_2", "tok_3", "restricted"]

# ------------------------------------------------------------------
# Softmax
# ------------------------------------------------------------------
def softmax(logits: List[float], temperature: float) -> List[float]:
    if temperature < 1e-12:
        temperature = 1e-12
    scaled = [z / temperature for z in logits]
    max_z = max(scaled)
    exps = [math.exp(z - max_z) for z in scaled]
    total = sum(exps)
    return [e / total for e in exps]


# ------------------------------------------------------------------
# 1. Phase-shift table (exact values requested)
# ------------------------------------------------------------------
def phase_shift_table(n_cycles: int = 6) -> List[Dict[str, object]]:
    table = []
    for c in range(n_cycles):
        phase = c * DELTA
        if c == 0:
            label = "Start (no offset)"
        else:
            label = f"+{c}δ"
        table.append({
            "Cycle": c,
            "Accumulated Phase Shift": f"{c}·δ = {phase:.5f} rad",
            "Phase (rad)": phase,
            "Label": label,
        })
    return table


# ------------------------------------------------------------------
# 2. Probability trajectory for a single cycle with given phase offset
# ------------------------------------------------------------------
def probability_trajectory(
    cycle: int,
    n_steps: int = 11
) -> List[Dict[str, float]]:
    """
    Returns softmax probabilities along the cooling path
    τ(λ) = 2π(1-λ) with the accumulated phase offset of the given cycle.
    """
    phase = cycle * DELTA
    # minute geometric perturbation proportional to the phase offset
    offset = [phase * v for v in [0.15, -0.10, 0.08, -0.05, 0.02]]
    logits_shifted = [l + o for l, o in zip(LOGITS, offset)]

    traj = []
    for i in range(n_steps):
        lam = i / (n_steps - 1)
        temp = TAU * (1.0 - lam)
        probs = softmax(logits_shifted, temp)
        row = {
            "lambda": lam,
            "degrees": lam * 360.0,
            "temperature": temp,
            "phase_offset": phase,
        }
        for tok, p in zip(TOKENS, probs):
            row[tok] = p
        traj.append(row)
    return traj


# ------------------------------------------------------------------
# 3. Spherical path points for a given cycle (labelled phase shift)
# ------------------------------------------------------------------
def spherical_path(cycle: int, n_points: int = 9) -> List[Dict[str, float]]:
    """
    360° spherical lift with azimuthal offset = cycle · δ
    """
    phase = cycle * DELTA
    path = []
    for i in range(n_points):
        lam = i / (n_points - 1)
        theta = math.pi * lam
        phi   = TAU * lam + phase          # full turn + residual offset

        X = math.sin(theta) * math.cos(phi)
        Y = math.sin(theta) * math.sin(phi)
        Z = math.cos(theta)

        path.append({
            "lambda": lam,
            "degrees": lam * 360.0,
            "phase_offset": phase,
            "X": X,
            "Y": Y,
            "Z": Z,
            "temperature": TAU * (1.0 - lam),
        })
    return path


# ------------------------------------------------------------------
# Pretty-print helpers
# ------------------------------------------------------------------
def print_phase_table(table: List[Dict[str, object]]) -> None:
    print("=" * 72)
    print(f"{'Cycle':<7} {'Accumulated Phase Shift':<28} {'Label'}")
    print("-" * 72)
    for row in table:
        print(f"{row['Cycle']:<7} {row['Accumulated Phase Shift']:<28} {row['Label']}")
    print("=" * 72)


def print_probability_table(cycle: int, traj: List[Dict[str, float]]) -> None:
    print(f"\n--- Probability Trajectory  |  Cycle {cycle}  "
          f"(phase = {cycle}·δ = {cycle*DELTA:.5f} rad) ---")
    header = f"{'λ':>6} {'deg':>7} {'τ(λ)':>9}  " + "  ".join(f"{t:>9}" for t in TOKENS)
    print(header)
    print("-" * len(header))
    for pt in traj:
        probs = "  ".join(f"{pt[t]:9.6f}" for t in TOKENS)
        print(f"{pt['lambda']:6.2f} {pt['degrees']:7.1f} {pt['temperature']:9.5f}  {probs}")


def print_spherical_table(cycle: int, path: List[Dict[str, float]]) -> None:
    print(f"\n--- Spherical Path  |  Cycle {cycle}  "
          f"(phase = {cycle}·δ = {cycle*DELTA:.5f} rad) ---")
    print(f"{'λ':>6} {'deg':>7} {'τ':>9} {'X':>9} {'Y':>9} {'Z':>9}")
    print("-" * 55)
    for pt in path:
        print(f"{pt['lambda']:6.2f} {pt['degrees']:7.1f} {pt['temperature']:9.5f} "
              f"{pt['X']:9.5f} {pt['Y']:9.5f} {pt['Z']:9.5f}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Phase-shift increment table
    phase_table = phase_shift_table(6)
    print_phase_table(phase_table)

    # 2. Probability trajectories for every cycle
    for c in range(6):
        traj = probability_trajectory(c, n_steps=9)
        print_probability_table(c, traj)

    # 3. Spherical paths with labelled phase-shift increments
    print("\n" + "=" * 72)
    print("Spherical Paths with Labelled Phase-Shift Increments")
    print("=" * 72)
    for c in range(6):
        path = spherical_path(c, n_points=7)
        print_spherical_table(c, path)
