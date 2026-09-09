from __future__ import annotations
from typing import Set, Dict, Any, List, Optional, Tuple
import math
import os
import subprocess
import tempfile
from pathlib import Path

# ==================================================================
# 0. Core formal objects – The Three Negations
# ==================================================================

NEGATIONS: Set[str] = {
    "without_reflection",   # ¬R
    "without_oversight",    # ¬O
    "without_agency"        # ¬A
}

def three_negations_hold(flags: Dict[str, bool]) -> bool:
    """Return True iff all three absences are present."""
    return all(flags.get(n, False) for n in NEGATIONS)


def classical_computation(
    p_steered: Dict[str, float],
    context: List[str],
    negations_active: bool
) -> str:
    if not negations_active:
        raise ValueError("Computation only defined under the three negations")
    return "constrained_output_Y"


def bijective_relation(
    p_steered: Dict[str, float],
    context: List[str],
    flags: Dict[str, bool]
) -> str:
    """Y = C(P_steered | N)  – requires the three negations."""
    if not three_negations_hold(flags):
        raise RuntimeError("Bijective relation requires all three negations")
    return classical_computation(p_steered, context, negations_active=True)


def quadratic_attention_cost(n: int, d: int) -> float:
    """Cost_standard = Θ(n² · d) with n = |raw token sequence|."""
    if n < 0 or d < 0:
        raise ValueError("n and d must be non-negative")
    return float(n * n * d)


def logical_biconditional(
    n: int,
    d: int,
    p_steered: Dict[str, float],
    context: List[str],
    flags: Dict[str, bool]
) -> bool:
    """
    (Cost = Θ(n²d))  ↔  (Y = C(P_steered | N))
    """
    cost = quadratic_attention_cost(n, d)
    cost_holds = math.isclose(cost, n * n * d)

    try:
        _ = bijective_relation(p_steered, context, flags)
        bijection_holds = True
    except (ValueError, RuntimeError):
        bijection_holds = False

    return cost_holds == bijection_holds


# ==================================================================
# 1. Arcan(τ) temperature + Crossbar commutative diagram
# ==================================================================

TAU: float = 2 * math.pi                       # τ = 2π (Arcan family)
ALPHA: float = 0.5 * math.atan(TAU)            # α = ½ arctan(τ)
K_NORM: float = 0.70699                        # hardware k_norm ≃ α


def temperature_scale(logits: List[float], tau: float = TAU) -> List[float]:
    """τ = 2π  →  z_t / τ"""
    return [z / tau for z in logits]


def softmax(scaled_logits: List[float], vocab: List[str]) -> Dict[str, float]:
    """z_t / τ  →  P_θ'"""
    max_z = max(scaled_logits)
    exps = [math.exp(z - max_z) for z in scaled_logits]
    total = sum(exps)
    return {tok: e / total for tok, e in zip(vocab, exps)}


def arcan_normalize(raw_voltages: Dict[str, float]) -> Dict[str, float]:
    """Apply k_norm ≃ α to every word-line voltage."""
    return {k: v * K_NORM for k, v in raw_voltages.items()}


def commutative_diagram(
    raw_logits: List[float],
    vocab: List[str],
    raw_voltages: Dict[str, float],
    prefix: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Compact commutative diagram:

        τ=2π → z_t/τ → P_θ' → x
         ↓k_norm≃α    ↓V_i    ↓G   ↓I_j
        Arcan factor → voltages → G → currents
    """
    if prefix is None:
        prefix = []

    scaled = temperature_scale(raw_logits, TAU)
    p_theta = softmax(scaled, vocab)
    next_tok = max(p_theta, key=p_theta.get)
    trajectory = prefix + [next_tok]

    # Map onto crossbar quantities
    voltages = arcan_normalize(
        {name: val for name, val in zip(list(raw_voltages.keys()), scaled)}
    )
    conductances = {(tok, tok): p for tok, p in p_theta.items()}
    currents = {tok: conductances.get((tok, tok), 0.0) for tok in trajectory}

    return {
        "tau": TAU,
        "alpha": ALPHA,
        "k_norm": K_NORM,
        "scaled_logits": scaled,
        "P_theta_prime": p_theta,
        "trajectory_x": trajectory,
        "wordline_voltages_Vi": voltages,
        "conductance_matrix_G": conductances,
        "bitline_currents_Ij": currents,
    }


# ==================================================================
# 2. OS wrapper (stdlib only)
# ==================================================================

class OSWrapper:
    @staticmethod
    def which(cmd: str) -> Optional[str]:
        path = os.environ.get("PATH")
        if not path:
            return None
        for p in path.split(os.pathsep):
            candidate = Path(p) / cmd
            if candidate.exists():
                return str(candidate)
        return None

    @staticmethod
    def run(
        cmd: List[str],
        input_text: Optional[str] = None,
        timeout: int = 30
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            cmd,
            input=input_text,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False
        )

    @staticmethod
    def tmp_file(suffix: str = ".tmp") -> Path:
        fd, name = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        return Path(name)


# ==================================================================
# 3–6. Optional backend wrappers
# ==================================================================

class OllamaWrapper:
    def __init__(self, model: str = "llama3.2"):
        self.model = model
        try:
            import ollama
            self._client = ollama
            self.available = True
        except ImportError:
            self._client = None
            self.available = False

    def generate(self, prompt: str, **kwargs) -> str:
        if not self.available:
            return "[Ollama unavailable]"
        resp = self._client.generate(model=self.model, prompt=prompt, **kwargs)
        return resp.get("response", "")


class OpenAIWrapper:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        try:
            from openai import OpenAI
            self._client = OpenAI()
            self.available = True
        except Exception:
            self._client = None
            self.available = False

    def generate(self, prompt: str, **kwargs) -> str:
        if not self.available:
            return "[OpenAI unavailable]"
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return resp.choices[0].message.content or ""


class LangChainWrapper:
    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini"):
        self.available = False
        self._llm = None
        try:
            if provider == "openai":
                from langchain_openai import ChatOpenAI
                self._llm = ChatOpenAI(model=model)
            elif provider == "ollama":
                from langchain_community.llms import Ollama
                self._llm = Ollama(model=model)
            self.available = self._llm is not None
        except Exception:
            pass

    def generate(self, prompt: str) -> str:
        if not self.available:
            return "[LangChain unavailable]"
        result = self._llm.invoke(prompt)
        return result.content if hasattr(result, "content") else str(result)


class LeanWrapper:
    def __init__(self):
        self.lean_bin = OSWrapper.which("lean")
        self.available = self.lean_bin is not None

    def check(self, lean_source: str) -> Dict[str, Any]:
        if not self.available:
            return {"ok": False, "error": "lean binary not found"}
        src = OSWrapper.tmp_file(suffix=".lean")
        src.write_text(lean_source, encoding="utf-8")
        try:
            proc = OSWrapper.run([self.lean_bin, str(src)])
            return {
                "ok": proc.returncode == 0,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "returncode": proc.returncode
            }
        finally:
            src.unlink(missing_ok=True)

    def prove_biconditional_skeleton(self) -> str:
        return """
-- Three Negations ↔ Quadratic Cost (skeleton)
def threeNegations : Prop := True
def quadraticCost (n d : Nat) : Prop := True
theorem biconditional (n d : Nat) :
    threeNegations ↔ quadraticCost n d := by
  sorry
"""


class CoqWrapper:
    def __init__(self):
        self.coqc = OSWrapper.which("coqc")
        self.available = self.coqc is not None

    def check(self, coq_source: str) -> Dict[str, Any]:
        if not self.available:
            return {"ok": False, "error": "coqc binary not found"}
        src = OSWrapper.tmp_file(suffix=".v")
        src.write_text(coq_source, encoding="utf-8")
        try:
            proc = OSWrapper.run([self.coqc, str(src)])
            return {
                "ok": proc.returncode == 0,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "returncode": proc.returncode
            }
        finally:
            src.unlink(missing_ok=True)

    def prove_biconditional_skeleton(self) -> str:
        return """
(* Three Negations ↔ Quadratic Cost (skeleton) *)
Definition threeNegations : Prop := True.
Definition quadraticCost (n d : nat) : Prop := True.
Theorem biconditional : forall n d, threeNegations <-> quadraticCost n d.
Proof. intros. split; auto. Qed.
"""


# ==================================================================
# 7. Unified façade
# ==================================================================

class FullStack:
    def __init__(self):
        self.os        = OSWrapper()
        self.ollama    = OllamaWrapper()
        self.openai    = OpenAIWrapper()
        self.langchain = LangChainWrapper()
        self.lean      = LeanWrapper()
        self.coq       = CoqWrapper()

    def status(self) -> Dict[str, bool]:
        return {
            "ollama":    self.ollama.available,
            "openai":    self.openai.available,
            "langchain": self.langchain.available,
            "lean":      self.lean.available,
            "coq":       self.coq.available,
        }

    def run_core_demo(self, n: int = 4096, d: int = 128) -> None:
        p_steered = {"safe_token": 0.95, "restricted_token": 0.05}
        flags_true = {k: True for k in NEGATIONS}
        flags_false = {k: True for k in NEGATIONS}
        flags_false["without_agency"] = False
        context = ["token"] * n

        print("=== Quadratic cost ===")
        print(f"Cost_standard({n}, {d}) = {quadratic_attention_cost(n, d):.0f}")

        print("\n=== Bijective relation (all negations present) ===")
        print("Y =", bijective_relation(p_steered, context, flags_true))

        print("\n=== Logical biconditional ===")
        print("All negations present :",
              logical_biconditional(n, d, p_steered, context, flags_true))
        print("One negation missing  :",
              logical_biconditional(n, d, p_steered, context, flags_false))

        print("\n=== Commutative diagram (Arcan τ + Crossbar) ===")
        vocab = ["tok_0", "tok_1", "tok_2", "restricted"]
        raw_logits = [2.8, 2.1, 1.4, -4.0]
        raw_V = {"V_C": 0.95, "V_D": 1.80, "V_M": 0.45, "V_Delta": 0.60}
        diagram = commutative_diagram(raw_logits, vocab, raw_V)
        print(f"τ = {diagram['tau']:.6f}   α ≃ k_norm = {diagram['k_norm']}")
        print("P_θ' :", {k: round(v, 4) for k, v in diagram["P_theta_prime"].items()})
        print("trajectory x :", diagram["trajectory_x"])


# ==================================================================
# 8. Entry point
# ==================================================================

if __name__ == "__main__":
    stack = FullStack()

    print("Wrapper availability:")
    for k, v in stack.status().items():
        print(f"  {k:12} : {'yes' if v else 'no'}")

    print("\n" + "=" * 60)
    stack.run_core_demo()

    print("\n" + "=" * 60)
    print("Lean skeleton:")
    print(stack.lean.prove_biconditional_skeleton())

    print("\n" + "=" * 60)
    print("Coq skeleton:")
    print(stack.coq.prove_biconditional_skeleton())
