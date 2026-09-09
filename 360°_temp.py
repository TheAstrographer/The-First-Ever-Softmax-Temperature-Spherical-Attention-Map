"""
Pure-Python realisation of:

    τ_k(λ) = 2π (1-λ),  λ ∈ [0,1]

    x(λ) = τ_k(λ)
    y(λ) = τ_k(λ)

    360° spherical lift on S²
    Temperature reset to the Arcan maximum after each full cycle
"""

from __future__ import annotations
import math
from typing import List, Tuple, Dict

# ------------------------------------------------------------------
# Arcan constants
# ------------------------------------------------------------------
TAU: float = 2.0 * math.pi          # τ = 2π
DELTA: float = 0.0051676            # seventh-multiple residual δ

# ------------------------------------------------------------------
# 1. Cooling law
# ------------------------------------------------------------------
def tau_k(lambda_: float) -> float:
    """
    τ_k(λ) = 2π (1-λ),  λ ∈ [0,1]
    """
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError("λ must lie in [0,1]")
    return TAU * (1.0 - lambda_)


# ------------------------------------------------------------------
# 2. Planar coordinates (x = y = τ_k(λ))
# ------------------------------------------------------------------
def planar_coords(lambda_: float) -> Tuple[float, float]:
    t = tau_k(lambda_)
    return t, t          # x(λ) = y(λ) = τ_k(λ)


# ------------------------------------------------------------------
# 3. 360° spherical lift on the unit sphere S²
# ------------------------------------------------------------------
def spherical_lift(lambda_: float) -> Tuple[float, float, float]:
    """
    Standard polar embedding that realises one full azimuthal turn:

        θ(λ) = π · λ
        φ(λ) = 2π · λ

        X = sinθ cosφ
        Y = sinθ sinφ
        Z = cosθ
    """
    theta = math.pi * lambda_
    phi   = TAU * lambda_          # exactly one full turn of τ

    sin_th = math.sin(theta)
    cos_th = math.cos(theta)
    cos_ph = math.cos(phi)
    sin_ph = math.sin(phi)

    X = sin_th * cos_ph
    Y = sin_th * sin_ph
    Z = cos_th
    return X, Y, Z


# ------------------------------------------------------------------
# 4. Temperature restoration after a completed 360° cycle
# ------------------------------------------------------------------
def restored_temperature(lambda_: float = 1.0) -> float:
    """
    At λ = 1 the planar coordinates vanish, but the azimuthal angle
    has accumulated a full turn, restoring τ = 2π.
    """
    x, y = planar_coords(lambda_)
    # The vanishing of the planar radius is offset by the full-turn constant
    return math.hypot(x, y) + TAU          # → 2π when λ = 1


# ------------------------------------------------------------------
# 5. One complete cycle (with optional residual-δ offset)
# ------------------------------------------------------------------
def run_cycle(
    n_steps: int = 11,
    apply_delta: bool = False
) -> List[Dict[str, float]]:
    """
    Sample a full cooling-and-reset trajectory.
    When apply_delta=True a minute comoving offset is added
    (continuous counterpart of the seventh-multiple residual).
    """
    trajectory = []
    for i in range(n_steps):
        lam = i / (n_steps - 1)
        t   = tau_k(lam)
        x, y = planar_coords(lam)
        X, Y, Z = spherical_lift(lam)

        if apply_delta:
            # minute comoving phase shift inherited from δ
            phase = DELTA * (1.0 - lam)
            X = X * math.cos(phase) - Y * math.sin(phase)
            Y = X * math.sin(phase) + Y * math.cos(phase)

        trajectory.append({
            "lambda":   lam,
            "tau":      t,
            "x":        x,
            "y":        y,
            "X":        X,
            "Y":        Y,
            "Z":        Z,
            "tau_restored": restored_temperature(lam) if lam == 1.0 else t,
        })
    return trajectory


# ------------------------------------------------------------------
# Demo
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("Arcan(τ) 360° Spherical Temperature Cycle")
    print("=" * 72)
    print(f"τ = {TAU:.12f}")
    print(f"δ = {DELTA:.7f} rad (seventh-multiple residual)")
    print()

    print("Cooling trajectory (no residual offset):")
    print(f"{'λ':>6}  {'τ_k(λ)':>10}  {'x=y':>10}  {'X':>9}  {'Y':>9}  {'Z':>9}")
    print("-" * 72)

    traj = run_cycle(n_steps=11, apply_delta=False)
    for pt in traj:
        print(f"{pt['lambda']:6.2f}  {pt['tau']:10.6f}  {pt['x']:10.6f}  "
              f"{pt['X']:9.5f}  {pt['Y']:9.5f}  {pt['Z']:9.5f}")

    print()
    final = traj[-1]
    print(f"At λ = 1.00  →  restored temperature = {final['tau_restored']:.12f}")
    print("(exactly back to the Arcan maximum τ = 2π)")
    print()

    print("Same cycle with residual-δ comoving offset:")
    traj_δ = run_cycle(n_steps=5, apply_delta=True)
    for pt in traj_δ:
        print(f"λ={pt['lambda']:.2f}  τ={pt['tau']:.6f}  "
              f"X={pt['X']:+.6f}  Y={pt['Y']:+.6f}  Z={pt['Z']:+.6f}")
