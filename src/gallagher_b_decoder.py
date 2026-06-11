"""
Gallagher-B Hard-Decision Bit-Flipping Decoder for the WiMAX Rate-5/6 LDPC Code (N=2304).

Gallager's Algorithm B [1] is an iterative hard-decision message-passing decoder on the Tanner graph of the LDPC code.

At each iteration:
  1. SYNDROME CHECK
     Compute s[j] = XOR of all variable estimates in check j.
     s[j] = 1 means check j is UNSATISFIED (parity wrong).
  2. FLIP-VOTE COUNT
     Each unsatisfied check "votes to flip" every variable it is connected to.
     For each variable node i:
         flip_votes[i] = number of connected checks that are unsatisfied
                       = number of checks that want variable i to change.
  3. THRESHOLD FLIP
     Each variable i is flipped if:
         flip_votes[i] >= b(i)
     where b(i) is the threshold, set to the strict majority of d_v(i):
         b(i) = floor(d_v(i) / 2) + 1
     For the WiMAX 5/6 code:
         d_v = 2  ->  b = 2  (both connected checks must be unsatisfied)
         d_v = 3  ->  b = 2
         d_v = 4  ->  b = 3
  4. Repeat until syndrome = 0 or max_iter is reached.

Why this threshold?
    A variable node connected to d_v checks flips only when the MAJORITY of
    its checks are unsatisfied. This prevents correct bits (which typically
    participate in only 1 unsatisfied check) from being flipped by mistake.

Performance on WiMAX 5/6
    The code has d_v in {2, 3, 4} and d_c = 20.
    Gallager-B works well at low BSC crossover probabilities (p <= 0.002).
    Above p ~ 0.003-0.005 the decoder stalls because too many checks become
    unsatisfied and the threshold condition can no longer distinguish errors
    from correct bits reliably.
    This is an expected, documented limitation of hard-decision decoding on
    irregular LDPC codes with low variable-node degrees [2].
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# Result dataclass
@dataclass
class GallagherBResult:
    """Output of one Gallagher-B decoding run."""

    decoded_bits: np.ndarray
    """Decoded binary vector of length N."""

    converged: bool
    """True if a valid codeword (syndrome = 0) was found."""

    iterations_used: int
    """Number of iterations before stopping."""

    residual_syndrome_weight: int
    """Number of unsatisfied checks at termination. 0 means converged."""

    elapsed_seconds: float
    """Wall-clock time of the decoding call."""

    ber_vs_iter: list[float] = field(default_factory=list)
    """BER at each iteration (populated only if reference_codeword is given)."""

# Core decoder
def gallagher_b_decode(
    H: np.ndarray,
    received_bits: np.ndarray,
    max_iter: int = 50,
    reference_codeword: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> GallagherBResult:
    """
    Gallagher-B hard-decision bit-flipping LDPC decoder.

    Parameters
    ----------
    H : np.ndarray, shape (M, N)
        Binary parity-check matrix (expanded, not base matrix).
    received_bits : np.ndarray, shape (N,)
        Hard-decision bits from the channel. Must contain only 0 and 1.
    max_iter : int
        Maximum number of decoding iterations.
    reference_codeword : np.ndarray, optional
        The transmitted codeword. When provided, BER is tracked each iteration.
    verbose : bool
        Print one summary line per iteration.

    Returns
    -------
    GallagherBResult
    """
    # ------ input validation ------
    if H.ndim != 2:
        raise ValueError("H must be a 2-D array.")
    M, N = H.shape
    if received_bits.ndim != 1 or received_bits.shape[0] != N:
        raise ValueError(f"received_bits must be 1-D of length {N}.")
    if not np.all((received_bits == 0) | (received_bits == 1)):
        raise ValueError("received_bits must contain only 0 or 1.")

    # ------ pre-compute graph properties ------
    H_bool = H.astype(bool)

    # Variable-node degree: how many checks each bit participates in
    var_degrees = H_bool.sum(axis=0).astype(np.int32)   # shape (N,)

    # Flip threshold: strict majority of connected checks
    # b(i) = floor(d_v(i) / 2) + 1
    threshold = var_degrees // 2 + 1                    # shape (N,)

    if verbose:
        print("[Gallagher-B decoder]")
        print(f"  Code parameters : N={N}, M={M}, max_iter={max_iter}")
        for dv in sorted(np.unique(var_degrees)):
            count = int(np.sum(var_degrees == dv))
            b_val = int(dv // 2 + 1)
            print(f"  d_v={dv} ({count:4d} nodes)  flip threshold b={b_val}")

    # ------ initialise ------
    v = received_bits.copy().astype(np.uint8)
    ber_vs_iter: list[float] = []
    t0 = time.perf_counter()
    syndrome_weight = 0

    # ------ main loop ------
    for iteration in range(1, max_iter + 1):

        # Step 1: which checks are unsatisfied?
        row_xor = (H_bool @ v) % 2              # shape (M,)  1=unsatisfied
        syndrome_weight = int(row_xor.sum())

        # Track BER before flipping (so iteration 1 shows channel BER)
        if reference_codeword is not None:
            ber_vs_iter.append(float(np.sum(v != reference_codeword)) / N)

        if verbose:
            extra = (
                f"  BER={ber_vs_iter[-1]:.6f}"
                if reference_codeword is not None else ""
            )
            print(f"  iter={iteration:3d}  syndrome={syndrome_weight:4d}{extra}")

        # Converged check (before any flipping)
        if syndrome_weight == 0:
            elapsed = time.perf_counter() - t0
            if verbose:
                print(f"  -> Converged at iteration {iteration}")
            return GallagherBResult(
                decoded_bits=v,
                converged=True,
                iterations_used=iteration,
                residual_syndrome_weight=0,
                elapsed_seconds=elapsed,
                ber_vs_iter=ber_vs_iter,
            )

        # Step 2: count unsatisfied checks per variable
        flip_votes = H_bool.T @ row_xor.astype(np.int32)   # shape (N,)

        # Step 3: flip variables that exceed the threshold
        flip_mask = flip_votes >= threshold
        v = v ^ flip_mask.astype(np.uint8)

    # Did not converge
    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"  -> Did not converge after {max_iter} iterations")
    return GallagherBResult(
        decoded_bits=v,
        converged=False,
        iterations_used=max_iter,
        residual_syndrome_weight=syndrome_weight,
        elapsed_seconds=elapsed,
        ber_vs_iter=ber_vs_iter,
    )


# BER / FER simulation

def run_ber_simulation(
    H: np.ndarray,
    base_matrix: np.ndarray,
    Z: int,
    crossover_probs: list[float],
    num_frames: int = 500,
    max_iter: int = 50,
    seed: int = 0,
) -> dict[float, dict]:
    """
    Monte-Carlo BER/FER simulation for Gallagher-B over a BSC.

    Parameters
    ----------
    H               : Expanded parity-check matrix (M x N).
    base_matrix     : LDPC base matrix (4 x 24 for WiMAX 5/6).
    Z               : Expansion factor (96).
    crossover_probs : List of BSC crossover probabilities to test.
    num_frames      : Number of random frames per probability value.
    max_iter        : Maximum Gallagher-B iterations per frame.
    seed            : Random seed for reproducibility.

    Returns
    -------
    dict[float -> dict] with keys per probability:
        'ber'                  : post-decoding bit-error rate
        'fer'                  : frame-error rate
        'avg_iter'             : average iterations per frame
        'converge_rate'        : fraction of frames where syndrome = 0
        'ber_before_decoding'  : channel BER before decoding
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from encoder import encode_message_qc
    from channel import add_bsc_noise

    M, N = H.shape
    K = N - M
    rng = np.random.default_rng(seed)
    results: dict[float, dict] = {}

    print(f"\n{'p':>6}  {'BER_ch':>10}  {'BER_dec':>10}  {'FER':>8}  "
          f"{'avg_iter':>9}  {'conv_rate':>10}")
    print("-" * 62)

    for p in crossover_probs:
        be_before = be_after = frame_errors = total_iter = total_conv = 0

        for _ in range(num_frames):
            msg = rng.integers(0, 2, size=K, dtype=np.uint8)
            cw  = encode_message_qc(base_matrix, msg, Z)
            rx, em = add_bsc_noise(cw, p, rng=rng)
            be_before += int(em.sum())

            res = gallagher_b_decode(
                H, rx,
                max_iter=max_iter,
                reference_codeword=cw,
                verbose=False,
            )

            be = int(np.sum(res.decoded_bits != cw))
            be_after     += be
            frame_errors += (be > 0)
            total_iter   += res.iterations_used
            total_conv   += res.converged

        total_bits = num_frames * N
        results[p] = {
            "ber":                  be_after / total_bits,
            "fer":                  frame_errors / num_frames,
            "avg_iter":             total_iter / num_frames,
            "converge_rate":        total_conv / num_frames,
            "ber_before_decoding":  be_before / total_bits,
        }
        r = results[p]
        print(
            f"{p:>6.4f}  {r['ber_before_decoding']:>10.4e}  "
            f"{r['ber']:>10.4e}  {r['fer']:>8.4f}  "
            f"{r['avg_iter']:>9.1f}  {r['converge_rate']:>10.4f}"
        )

    return results