import math
import re
import hashlib
from collections import defaultdict
from typing import Dict

# ============================================================
# BitPerfectFilter (structural token filter / first-occurrence cache)
# Integrated for long-context efficiency & spam/structural suppression
# ============================================================
class BitPerfectFilter:
    UNIT_SEP = '\x1F'  # Unit Separator
    GROUP_SEP = '\x1D'  # Group Separator (full suppression)

    def __init__(self, max_units: int = 80, max_length: int = 8000):
        self.max_units = max_units
        self.max_length = max_length
        self.session_counters = defaultdict(lambda: defaultdict(int))
        self.CACHE_4: Dict[str, str] = {}  # block_hash → exact original (first occurrence only)

        # Marker support: [A] to [ZZ], [0] to [999], and easily adaptable
        self.marker_pattern = re.compile(r'\[([A-Za-z0-9_-]{1,20})\]')

    def _collapse_blocks(self, text: str) -> str:
        """Clean structural normalization with marker support"""
        if not text:
            return ""

        if len(text) > self.max_length:
            return self.GROUP_SEP

        # Marker replacement
        cleaned = self.marker_pattern.sub(self.UNIT_SEP, text)

        # Normalize whitespace
        cleaned = re.sub(r'[\s]+', ' ', cleaned).strip()

        # Structural splitting on common delimiters
        units = re.split(r'([{};\n])', cleaned)
        cleaned_units = [u.strip() for u in units if u.strip()]

        collapsed = self.UNIT_SEP.join(cleaned_units)

        # Final suppression check
        unit_count = collapsed.count(self.UNIT_SEP)
        if unit_count >= self.max_units or len(collapsed) > self.max_length:
            return self.GROUP_SEP

        if unit_count >= 2:
            return self.GROUP_SEP + collapsed + self.GROUP_SEP

        return collapsed

    def filter_message(self, text: str, session_id: str = "default"):
        original = text
        collapsed = self._collapse_blocks(text)

        block_hash = hashlib.sha256(collapsed.encode('utf-8', errors='ignore')).hexdigest()[:32]

        counter = self.session_counters[session_id]
        counter[block_hash] += 1
        is_first = counter[block_hash] == 1

        if is_first:
            self.CACHE_4[block_hash] = original

        return {
            "block_hash": block_hash,
            "original_len": len(original),
            "collapsed_len": len(collapsed),
            "is_suppressed": collapsed == self.GROUP_SEP,
            "unit_count": collapsed.count(self.UNIT_SEP) if collapsed != self.GROUP_SEP else 0,
            "is_first_occurrence": is_first,
            "occurrence_count": counter[block_hash],
            "extractable": True,
            "marker_count": len(self.marker_pattern.findall(text))
        }

    def extract_original(self, block_hash: str) -> str:
        """Bit-perfect recovery of the exact first-seen original."""
        return self.CACHE_4.get(block_hash, "")

    def extract_or_default(self, block_hash: str, default: str = "") -> str:
        return self.CACHE_4.get(block_hash, default)

    def get_cache_size(self) -> int:
        return len(self.CACHE_4)


# ============================================================
# Constants from the notes (Thinnest-Triangle)
# ============================================================
psi = 0.15033788                    # ≈ 8.61°
cos_psi = math.cos(psi)             # ≈ 0.98872053
cos2_psi = cos_psi ** 2             # ≈ 0.9775 (97.75%)
eta_psi = psi / (2 * math.pi)       # ≈ 0.02392

M = 30
B_effective_coeff = M * eta_psi     # ≈ 0.7177

# ============================================================
# Helper: discrete angles t_k = 2πk / N
# ============================================================
def get_angles(N):
    return [2 * math.pi * k / N for k in range(N)]

# ============================================================
# Probability distribution
# P_i = exp(cos(t_i) * cosψ) / Σ exp(cos(t_j) * cosψ)
# ============================================================
def compute_P(N, cos_psi):
    angles = get_angles(N)
    logits = [math.exp(math.cos(t) * cos_psi) for t in angles]
    Z = sum(logits)
    return [x / Z for x in logits]

# ============================================================
# Alternative form that appears in the notes
# A_ij related to exp(cos(t_i) * cos²ψ)
# ============================================================
def compute_A(N, cos2_psi):
    angles = get_angles(N)
    logits = [math.exp(math.cos(t) * cos2_psi) for t in angles]
    Z = sum(logits)
    return [x / Z for x in logits]

# ============================================================
# Capacity formulas
# ============================================================
def C_classical(B, S, N):
    return B * math.log2(1 + S / N)

def C_JCRIN(gamma, M, U, S, N, cos2_psi):
    B = gamma * M * U
    return B * math.log2(1 + (S * cos2_psi) / N)

def B_effective(U):
    return B_effective_coeff * U

# ============================================================
# Demo / verification + BitPerfectFilter integration example
# ============================================================
if __name__ == "__main__":
    print("=== Basic constants ===")
    print(f"ψ          = {psi:.8f} rad ≈ {math.degrees(psi):.2f}°")
    print(f"cosψ       = {cos_psi:.8f}")
    print(f"cos²ψ      = {cos2_psi:.8f}  ({cos2_psi*100:.2f}%)")
    print(f"ηψ         = {eta_psi:.5f}")
    print(f"B_eff ≈    {B_effective_coeff:.4f} * U")

    print("\n=== Probability distributions ===")
    for N in [16, 32, 64]:
        P = compute_P(N, cos_psi)
        print(f"\nP^{N} (first 5 values):")
        print([round(p, 6) for p in P[:5]], "...")

    print("\n=== A version with cos²ψ ===")
    for N in [16, 32, 64]:
        A = compute_A(N, cos2_psi)
        print(f"\nA^{N} (first 5 values):")
        print([round(a, 6) for a in A[:5]], "...")

    print("\n=== Example capacity calculation ===")
    gamma = 1.0
    U = 100.0
    S = 10.0
    N = 1.0

    c_j = C_JCRIN(gamma, M, U, S, N, cos2_psi)
    c_c = C_classical(gamma * M * U, S, N)

    print(f"C_JCRIN     = {c_j:.4f}")
    print(f"C_classical = {c_c:.4f}")
    print(f"B_effective = {B_effective(U):.4f}")

    # ----------------------------------------------------------
    # BitPerfectFilter demonstration (structural pre-filter)
    # ----------------------------------------------------------
    print("\n=== BitPerfectFilter integration demo ===")
    bpf = BitPerfectFilter(max_units=80, max_length=8000)

    # Example long structured input (simulates CSS / log / marker spam)
    test_msg = "[stylesheet-group=\"0\"]{} body{margin:0;} " * 200
    result = bpf.filter_message(test_msg, session_id="jcrin_demo")

    print(f"Original length : {result['original_len']:,}")
    print(f"Collapsed length: {result['collapsed_len']}")
    print(f"Suppressed      : {result['is_suppressed']}")
    print(f"Unit count      : {result['unit_count']}")
    print(f"Marker count    : {result['marker_count']}")
    print(f"First occurrence: {result['is_first_occurrence']}")
    print(f"Cache size      : {bpf.get_cache_size()}")
