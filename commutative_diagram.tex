"""
Compact commutative diagram

τ = 2π  →  z_t / τ  →  P_θ'  →  x
 ↓ k_norm ≃ α          ↓ V_i      ↓ G      ↓ I_j
Arcan factor  →  word-line voltages  →  conductance matrix  →  bit-line currents
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import math

# ------------------------------------------------------------------
# Constants from the Arcan family + Crossbar table
# ------------------------------------------------------------------
TAU = 2 * math.pi                          # τ = 2π
ALPHA = 0.5 * math.atan(TAU)               # α = (1/2) arctan(τ)
K_NORM = 0.70699                           # hardware normalization ≈ α

# ------------------------------------------------------------------
# Node type aliases
# ------------------------------------------------------------------
Logits          = List[float]
ProbDist        = Dict[str, float]
Trajectory      = List[str]
VoltageVector   = Dict[str, float]
ConductanceMat  = Dict[Tuple[str, str], float]
CurrentVector   = Dict[str, float]

# ------------------------------------------------------------------
# Horizontal arrows (top row – mathematical LM side)
# ------------------------------------------------------------------
def temperature_scale(logits: Logits, tau: float = TAU) -> Logits:
    """τ = 2π  →  z_t / τ"""
    return [z / tau for z in logits]

def softmax(scaled_logits: Logits, vocab: List[str]) -> ProbDist:
    """z_t / τ  →  P_θ'"""
    max_z = max(scaled_logits)
    exps = [math.exp(z - max_z) for z in scaled_logits]
    total = sum(exps)
    return {tok: e / total for tok, e in zip(vocab, exps)}

def sample_trajectory(p: ProbDist, prefix: Trajectory) -> Trajectory:
    """P_θ'  →  x  (deterministic argmax for illustration)"""
    next_tok = max(p, key=p.get)
    return prefix + [next_tok]

# ------------------------------------------------------------------
# Vertical arrows (mapping to crossbar hardware)
# ------------------------------------------------------------------
def arcan_normalize(raw_voltages: VoltageVector) -> VoltageVector:
    """k_norm ≃ α  (Arcan factor applied to every word-line)"""
    return {k: v * K_NORM for k, v in raw_voltages.items()}

def logits_to_voltages(scaled_logits: Logits, names: List[str]) -> VoltageVector:
    """z_t / τ  ↦  V_i"""
    return {name: val for name, val in zip(names, scaled_logits)}

def probs_to_conductances(p: ProbDist) -> ConductanceMat:
    """P_θ'  ↦  G  (diagonal conductance matrix for illustration)"""
    return {(tok, tok): prob for tok, prob in p.items()}

def trajectory_to_currents(x: Trajectory, g: ConductanceMat) -> CurrentVector:
    """x  ↦  I_j  (currents collected on bit-lines)"""
    currents = {}
    for tok in x:
        currents[tok] = g.get((tok, tok), 0.0)
    return currents

# ------------------------------------------------------------------
# Full commutative diagram as a single callable
# ------------------------------------------------------------------
def commutative_diagram(
    raw_logits: Logits,
    vocab: List[str],
    raw_voltages: VoltageVector,
    prefix: Trajectory | None = None
) -> Dict[str, object]:
    """
    Executes both paths of the diagram and returns every node.
    """
    if prefix is None:
        prefix = []

    # Top row (mathematical)
    scaled = temperature_scale(raw_logits, TAU)
    p_theta = softmax(scaled, vocab)
    trajectory = sample_trajectory(p_theta, prefix)

    # Vertical maps
    voltages = arcan_normalize(logits_to_voltages(scaled, list(raw_voltages.keys())))
    conductances = probs_to_conductances(p_theta)
    currents = trajectory_to_currents(trajectory, conductances)

    return {
        "tau": TAU,
        "scaled_logits": scaled,
        "P_theta_prime": p_theta,
        "trajectory_x": trajectory,
        "k_norm": K_NORM,
        "alpha": ALPHA,
        "wordline_voltages_Vi": voltages,
        "conductance_matrix_G": conductances,
        "bitline_currents_Ij": currents,
    }

# ------------------------------------------------------------------
# Demo
# ------------------------------------------------------------------
if __name__ == "__main__":
    vocab = ["tok_0", "tok_1", "tok_2", "tok_3", "restricted"]
    raw_logits = [2.8, 2.1, 1.4, 0.6, -4.0]          # steered: restricted already low
    raw_V = {"V_C": 0.95, "V_D": 1.80, "V_M": 0.45, "V_Delta": 0.60}

    result = commutative_diagram(raw_logits, vocab, raw_V)

    print("=== Compact Commutative Diagram ===\n")
    print(f"τ = {result['tau']:.12f}")
    print(f"α = {result['alpha']:.12f}   ≃   k_norm = {result['k_norm']}")
    print()
    print("Top row (LM):")
    print("  scaled logits z/τ :", [round(z, 4) for z in result["scaled_logits"]])
    print("  P_θ'              :", {k: round(v, 4) for k, v in result["P_theta_prime"].items()})
    print("  trajectory x      :", result["trajectory_x"])
    print()
    print("Bottom row (Crossbar):")
    print("  word-line voltages V_i :", {k: round(v, 4) for k, v in result["wordline_voltages_Vi"].items()})
    print("  conductance G          :", {k: round(v, 4) for k, v in result["conductance_matrix_G"].items()})
    print("  bit-line currents I_j  :", {k: round(v, 4) for k, v in result["bitline_currents_Ij"].items()})
