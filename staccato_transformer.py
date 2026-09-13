import math
import random
import hashlib
import re
from collections import defaultdict
from typing import Dict, List, Union
from decimal import Decimal, getcontext

getcontext().prec = 28

# ============================================================
# 0. KERNEL DIVISION BRIDGE + STACCATO SPECIAL (from staccato_special.py)
# ============================================================
class KernelDivisionBridge:
    """Exact constants from the repository."""
    def __init__(self):
        self.psi = Decimal("0.1503378808")
        self.Re_tau = Decimal("1.4129651365")
        self.cos_psi = Decimal("0.9887205")
        self.sin_Re_tau = Decimal("0.98768834059")

        numerator = self.cos_psi * self.Re_tau
        self.K = numerator / self.sin_Re_tau
        self.k_norm = Decimal(1) / self.K          # ≈ 0.7071

    @property
    def k_norm_float(self) -> float:
        return float(self.k_norm)


class LowCadenceVMM:
    """
    Low-cadence (staccato) specialization of the VMM equation:

        I_staccato = (k_norm / 0.47) * Σ (V_i * G_i)
                   ≈ 1.5045 * Σ (V_i * G_i)
    """
    def __init__(self):
        self.bridge = KernelDivisionBridge()
        self.N_low = Decimal("0.47")               # low-regime normalizer
        self.gain = self.bridge.k_norm / self.N_low  # ≈ 1.5045

    def to_signed_int32(self, val: int) -> int:
        """Enforce true 32-bit signed arithmetic."""
        if val & 0x80000000:
            return val - 0x100000000
        return val

    def raw_dot_product(self, V: List[int], G: List[int]) -> int:
        """Σ (V_i * G_i) with signed 32-bit multiplication."""
        if len(V) != len(G):
            raise ValueError("V and G must have the same length")
        acc = 0
        for v, g in zip(V, G):
            acc += self.to_signed_int32(v) * self.to_signed_int32(g)
        return acc

    def compute(self, V: List[int], G: List[int]) -> Dict[str, Union[int, float, str]]:
        """
        Full low-cadence VMM evaluation.
        Returns raw accumulation, kernel-scaled value, and final normalized result.
        """
        raw = self.raw_dot_product(V, G)
        raw_dec = Decimal(raw)

        # kernel scaling
        kernel_scaled = raw_dec * self.bridge.k_norm

        # low-regime normalization
        I_low = kernel_scaled / self.N_low

        return {
            "raw_dot_product": raw,
            "k_norm": float(self.bridge.k_norm),
            "N_low": float(self.N_low),
            "gain": float(self.gain),
            "kernel_scaled": int(kernel_scaled.to_integral_value()),
            "I_normalized_low": float(I_low),
            "I_normalized_low_sci": f"{float(I_low):.6e}",
        }


# ============================================================
# 1. BITPERFECT FILTER
# ============================================================
class BitPerfectFilter:
    UNIT_SEP = '\x1F'
    GROUP_SEP = '\x1D'

    def __init__(self, max_units: int = 80, max_length: int = 8000):
        self.max_units = max_units
        self.max_length = max_length
        self.session_counters = defaultdict(lambda: defaultdict(int))
        self.CACHE_4: Dict[str, str] = {}
        self.marker_pattern = re.compile(r'\[([A-Za-z0-9_-]{1,20})\]')

    def _collapse_blocks(self, text: str) -> str:
        if not text:
            return ""
        if len(text) > self.max_length:
            return self.GROUP_SEP
        cleaned = self.marker_pattern.sub(self.UNIT_SEP, text)
        cleaned = re.sub(r'[\s]+', ' ', cleaned).strip()
        units = re.split(r'([{};\n])', cleaned)
        cleaned_units = [u.strip() for u in units if u.strip()]
        collapsed = self.UNIT_SEP.join(cleaned_units)
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
            "marker_count": len(self.marker_pattern.findall(text)),
            "collapsed_text": collapsed
        }

    def extract_original(self, block_hash: str) -> str:
        return self.CACHE_4.get(block_hash, "")


# ============================================================
# 2. FULL TRANSFORMER STACK
# ============================================================
def zeros_matrix(rows, cols):
    return [[0.0 for _ in range(cols)] for _ in range(rows)]

def random_matrix(rows, cols):
    scale = math.sqrt(2.0 / cols)
    return [[random.gauss(0.0, scale) for _ in range(cols)] for _ in range(rows)]

def transpose(matrix):
    return [[matrix[r][c] for r in range(len(matrix))] for c in range(len(matrix[0]))]

def matmul(A, B):
    M = len(A)
    N = len(A[0])
    N2 = len(B)
    P = len(B[0])
    assert N == N2, f"Dimension mismatch: {N} != {N2}"
    result = zeros_matrix(M, P)
    B_T = transpose(B)
    for i in range(M):
        for j in range(P):
            result[i][j] = sum(A[i][k] * B_T[j][k] for k in range(N))
    return result

def add_matrices(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def softmax_row_masked(row, mask_row=None):
    max_val = -float('inf')
    for i, val in enumerate(row):
        if mask_row and mask_row[i] == 1:
            continue
        if val > max_val:
            max_val = val
    if max_val == -float('inf'):
        max_val = 0.0
    exps = []
    for i, x in enumerate(row):
        if mask_row and mask_row[i] == 1:
            exps.append(0.0)
        else:
            exps.append(math.exp(x - max_val))
    sum_exps = sum(exps)
    if sum_exps == 0.0:
        return [1.0 / len(row) for _ in row]
    return [x / sum_exps for x in exps]

class LayerNorm:
    def __init__(self, d_model, eps=1e-5):
        self.eps = eps
        self.gamma = [1.0 for _ in range(d_model)]
        self.beta = [0.0 for _ in range(d_model)]

    def forward(self, X):
        N, D = len(X), len(X[0])
        out = zeros_matrix(N, D)
        for i in range(N):
            mean = sum(X[i]) / D
            variance = sum((x - mean) ** 2 for x in X[i]) / D
            std = math.sqrt(variance + self.eps)
            for j in range(D):
                out[i][j] = self.gamma[j] * ((X[i][j] - mean) / std) + self.beta[j]
        return out

class MultiHeadAttention:
    def __init__(self, d_model, n_heads):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = random_matrix(d_model, d_model)
        self.W_k = random_matrix(d_model, d_model)
        self.W_v = random_matrix(d_model, d_model)
        self.W_o = random_matrix(d_model, d_model)

    def forward(self, Q_in, K_in, V_in, is_causal=False):
        tgt_len = len(Q_in)
        src_len = len(K_in)
        Q_all = matmul(Q_in, transpose(self.W_q))
        K_all = matmul(K_in, transpose(self.W_k))
        V_all = matmul(V_in, transpose(self.W_v))
        head_outputs = []
        scale = math.sqrt(self.d_k)
        for h in range(self.n_heads):
            start_idx = h * self.d_k
            end_idx = start_idx + self.d_k
            Q_h = [[row[c] for c in range(start_idx, end_idx)] for row in Q_all]
            K_h = [[row[c] for c in range(start_idx, end_idx)] for row in K_all]
            V_h = [[row[c] for c in range(start_idx, end_idx)] for row in V_all]
            scores = matmul(Q_h, transpose(K_h))
            for i in range(tgt_len):
                mask_row = [1 if j > i else 0 for j in range(src_len)] if is_causal else None
                for j in range(src_len):
                    scores[i][j] /= scale
                scores[i] = softmax_row_masked(scores[i], mask_row)
            context_h = matmul(scores, V_h)
            head_outputs.append(context_h)
        concat_out = zeros_matrix(tgt_len, self.d_model)
        for i in range(tgt_len):
            col_idx = 0
            for h in range(self.n_heads):
                for j in range(self.d_k):
                    concat_out[i][col_idx] = head_outputs[h][i][j]
                    col_idx += 1
        return matmul(concat_out, transpose(self.W_o))

class PositionWiseFeedForward:
    def __init__(self, d_model, d_ff=None):
        if d_ff is None:
            d_ff = d_model * 4
        self.W1 = random_matrix(d_ff, d_model)
        self.W2 = random_matrix(d_model, d_ff)

    def forward(self, X):
        h1 = matmul(X, transpose(self.W1))
        for i in range(len(h1)):
            for j in range(len(h1[0])):
                if h1[i][j] < 0:
                    h1[i][j] = 0.0
        return matmul(h1, transpose(self.W2))

class EncoderLayer:
    def __init__(self, d_model, n_heads, d_ff=None):
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ffn = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(self, X):
        attn_out = self.mha.forward(X, X, X, is_causal=False)
        x = self.norm1.forward(add_matrices(X, attn_out))
        ffn_out = self.ffn.forward(x)
        return self.norm2.forward(add_matrices(x, ffn_out))

class DecoderLayer:
    def __init__(self, d_model, n_heads, d_ff=None):
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

    def forward(self, tgt, memory):
        self_attn_out = self.self_attn.forward(tgt, tgt, tgt, is_causal=True)
        x = self.norm1.forward(add_matrices(tgt, self_attn_out))
        cross_attn_out = self.cross_attn.forward(x, memory, memory, is_causal=False)
        x = self.norm2.forward(add_matrices(x, cross_attn_out))
        ffn_out = self.ffn.forward(x)
        return self.norm3.forward(add_matrices(x, ffn_out))

class TransformerStack:
    def __init__(self, num_encoder_layers, num_decoder_layers, d_model, n_heads, d_ff=None):
        self.d_model = d_model
        self.encoder_layers = [EncoderLayer(d_model, n_heads, d_ff) for _ in range(num_encoder_layers)]
        self.decoder_layers = [DecoderLayer(d_model, n_heads, d_ff) for _ in range(num_decoder_layers)]

    def generate_positional_encoding(self, seq_len):
        pe = zeros_matrix(seq_len, self.d_model)
        for pos in range(seq_len):
            for i in range(0, self.d_model, 2):
                div_term = math.exp(i * -(math.log(10000.0) / self.d_model))
                pe[pos][i] = math.sin(pos * div_term)
                if i + 1 < self.d_model:
                    pe[pos][i + 1] = math.cos(pos * div_term)
        return pe

    def forward(self, src_tokens, tgt_tokens):
        src_pe = self.generate_positional_encoding(len(src_tokens))
        tgt_pe = self.generate_positional_encoding(len(tgt_tokens))
        enc_input = add_matrices(src_tokens, src_pe)
        dec_input = add_matrices(tgt_tokens, tgt_pe)
        memory = enc_input
        for layer in self.encoder_layers:
            memory = layer.forward(memory)
        output = dec_input
        for layer in self.decoder_layers:
            output = layer.forward(output, memory)
        return output


# ============================================================
# 3. FULLY INTEGRATED JCRIN PRODUCTION PIPELINE
# ============================================================
class JCRINProductionPipeline:
    def __init__(self, 
                 num_encoder_layers=2, 
                 num_decoder_layers=2, 
                 d_model=64, 
                 n_heads=4):
        self.filter = BitPerfectFilter()
        self.bridge = KernelDivisionBridge()
        self.staccato = LowCadenceVMM()          # ← integrated staccato_special
        self.transformer = TransformerStack(
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            d_model=d_model,
            n_heads=n_heads
        )
        self.d_model = d_model

    def _text_to_matrix(self, text: str, seq_len: int):
        matrix = zeros_matrix(seq_len, self.d_model)
        if not text:
            return matrix
        for i in range(seq_len):
            for j in range(self.d_model):
                val = (hash(text[i % len(text)] + str(j)) % 10000) / 10000.0
                matrix[i][j] = val * 0.1
        return matrix

    def _matrix_to_int_vectors(self, matrix, dim=8):
        """Convert a slice of the transformer output into integer V and G vectors for staccato VMM."""
        flat = [int(x * 100000) for row in matrix for x in row]
        # Take two equal-length slices
        half = min(len(flat) // 2, dim)
        V = flat[:half]
        G = flat[half:half*2] if half*2 <= len(flat) else flat[:half]
        # Pad if necessary
        while len(V) < dim:
            V.append(0)
        while len(G) < dim:
            G.append(0)
        return V[:dim], G[:dim]

    def process(self, source_text: str, target_text: str = None, session_id: str = "prod"):
        print("=" * 70)
        print("JCRIN PRODUCTION PIPELINE  (with Staccato Low-Cadence VMM)")
        print("=" * 70)

        # Stage 1 – BitPerfectFilter
        filt = self.filter.filter_message(source_text, session_id=session_id)
        print(f"[1] BitPerfectFilter")
        print(f"    Original length : {filt['original_len']}")
        print(f"    Collapsed length: {filt['collapsed_len']}")
        print(f"    Suppressed      : {filt['is_suppressed']}")

        if filt['is_suppressed']:
            print("    → Fully suppressed. Pipeline halted.")
            return {"status": "suppressed", "filter": filt}

        # Stage 2 – Prepare matrices
        src_len = min(32, max(4, filt['collapsed_len'] // 8))
        tgt_len = src_len if target_text is None else min(32, max(4, len(target_text) // 8))

        src_matrix = self._text_to_matrix(filt.get("collapsed_text", source_text), src_len)
        tgt_matrix = self._text_to_matrix(target_text or source_text, tgt_len)

        # Stage 3 – Full Transformer Stack
        print(f"[2] TransformerStack  (enc={len(self.transformer.encoder_layers)}, "
              f"dec={len(self.transformer.decoder_layers)}, d_model={self.d_model})")
        output_matrix = self.transformer.forward(src_matrix, tgt_matrix)
        print(f"    Output shape    : [{len(output_matrix)}, {len(output_matrix[0])}]")

        # Stage 4 – Staccato Low-Cadence VMM
        V, G = self._matrix_to_int_vectors(output_matrix, dim=8)
        staccato_result = self.staccato.compute(V, G)
        print(f"[3] LowCadenceVMM (Staccato)")
        print(f"    raw_dot_product : {staccato_result['raw_dot_product']}")
        print(f"    kernel_scaled   : {staccato_result['kernel_scaled']}")
        print(f"    I_normalized_low: {staccato_result['I_normalized_low_sci']}")
        print(f"    gain            : {staccato_result['gain']:.6f}")

        return {
            "status": "completed",
            "filter": filt,
            "output_matrix": output_matrix,
            "staccato": staccato_result
        }


# ============================================================
# DEMONSTRATION
# ============================================================
if __name__ == "__main__":
    pipeline = JCRINProductionPipeline(
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_model=64,
        n_heads=4
    )

    print("\n--- Clean scientific input ---")
    result = pipeline.process(
        source_text="The thinnest triangle residual angle Psi generates the temperature-invariant softmax and the 30-fold cyclic lock on S2.",
        session_id="demo_clean"
    )

    print("\n--- High-structure spam (should suppress) ---")
    spam = '[stylesheet-group="0"]{} body{margin:0;} ' * 200
    pipeline.process(spam, session_id="demo_spam")
