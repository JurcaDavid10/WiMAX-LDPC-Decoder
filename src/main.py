import numpy as np

from config import BMAT_PATH, Z, NB_ROWS, NB_COLS, M, N, K, BSC_CROSSOVER_PROB, BSC_LLR_MAGNITUDE, LLR_SATURATION_LIMIT
from base_matrix import load_base_matrix
from qc_matrix import expand_base_matrix
from syndrome import compute_syndrome, is_codeword
from encoder import encode_message_qc
from channel import (
    add_bsc_noise,
    llr_bsc_quantized,
    hard_decision_from_llr
)

from graph_builder import build_tanner_graph
from decoder_min_sum import FloodedMinSumDecoder

def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_status(label: str, passed: bool) -> None:
    status = "PASS" if passed else "FAIL"
    print(f"{label:<45}: {status}")


def format_vector(vec: np.ndarray, length: int = 10, decimals: int | None = None) -> str:
    sample = vec[:length]
    if decimals is not None:
        sample = np.round(sample.astype(float), decimals)
    return np.array2string(sample, separator=", ")

def run_gallagher_b_section(
    h_matrix: np.ndarray,
    base_matrix: np.ndarray,
    codeword: np.ndarray,
    received_bits: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """Section 7: Gallagher-B decoder integration."""
    from gallagher_b_decoder import gallagher_b_decode, run_ber_simulation
    from config import Z
 
    print_section("7. GALLAGHER-B HARD-DECISION BIT-FLIPPING DECODER")
 
    N = codeword.shape[0]
    channel_errors = int(np.sum(received_bits != codeword))
    print(f"Channel bit errors going into decoder : {channel_errors} / {N}")
    print(f"Channel BER                           : {channel_errors / N:.4f}")
    print()
 
    # Single-frame decode (reuse the codeword+received from section 4/5)
    result = gallagher_b_decode(
        h_matrix,
        received_bits,
        max_iter=50,
        reference_codeword=codeword,
        verbose=True,
    )
 
    errors_after = int(np.sum(result.decoded_bits != codeword))
 
    print()
    print_status(
        "Decoder converged (syndrome = 0)",
        result.converged,
    )
    print(f"{'Iterations used':<45}: {result.iterations_used}")
    print(f"{'Residual syndrome weight':<45}: {result.residual_syndrome_weight}")
    print(f"{'Bit errors before decoding':<45}: {channel_errors}")
    print(f"{'Bit errors after decoding':<45}: {errors_after}")
    print(f"{'Decoding time (s)':<45}: {result.elapsed_seconds:.4f}")
    print_status(
        "Decoded vector is a valid codeword",
        is_codeword(h_matrix, result.decoded_bits),
    )
 
    # BER curve
    print()
    print("--- BER / FER Simulation (BSC, 500 frames per p-value) ---")
    print(
        "Note: Gallagher-B on the WiMAX 5/6 code (d_v in {2,3,4}) converges\n"
        "reliably only at low crossover probabilities (p <= 0.002).\n"
        "At higher p, the decoder stalls due to low variable-node degrees.\n"
        "This is a known property of hard-decision decoding on irregular LDPC codes."
    )
    run_ber_simulation(
        H=h_matrix,
        base_matrix=base_matrix,
        Z=Z,
        crossover_probs=[0.001, 0.002, 0.003, 0.005, 0.010, 0.020],
        num_frames=100,
        max_iter=50,
        seed=42,
    )

def main() -> None:
    rng = np.random.default_rng(seed=42)

    # ------------------------------------------------------------------
    # 1) LOAD BASE MATRIX
    # ------------------------------------------------------------------
    print_section("1. BASE MATRIX LOADING")

    base_matrix = load_base_matrix(BMAT_PATH)
    base_shape_ok = base_matrix.shape == (NB_ROWS, NB_COLS)

    print(f"Base matrix file        : {BMAT_PATH.name}")
    print(f"Expected shape          : ({NB_ROWS}, {NB_COLS})")
    print(f"Loaded shape            : {base_matrix.shape}")
    print_status("Base matrix shape check", base_shape_ok)

    if not base_shape_ok:
        raise ValueError(
            f"Unexpected base matrix shape: {base_matrix.shape}. "
            f"Expected: {(NB_ROWS, NB_COLS)}"
        )

    print("First row of base matrix:")
    print(base_matrix[0])

# ------------------------------------------------------------------
# TANNER GRAPH CONSTRUCTION VALIDATION
# ------------------------------------------------------------------
    print_section("TANNER GRAPH CONSTRUCTION VALIDATION")

    (
        check_edges,
        var_edges,
        edge_to_check,
        edge_to_var
    ) = build_tanner_graph(
        base_matrix,
        Z
    )

    print("Checks :", len(check_edges))
    print("Vars   :", len(var_edges))
    print("Edges  :", len(edge_to_check))

    print("Average VN degree:",
        np.mean([len(v) for v in var_edges]))

    print("Average CN degree:",
        np.mean([len(c) for c in check_edges]))

    # ------------------------------------------------------------------
    # 2) BUILD FULL PARITY-CHECK MATRIX H
    # ------------------------------------------------------------------
    print_section("2. QC-LDPC PARITY-CHECK MATRIX EXPANSION FOR VALIDATION ONLY")

    # Full H is built only for syndrome validation, not for encoding.
    h_matrix = expand_base_matrix(base_matrix, Z)
    h_shape_ok = h_matrix.shape == (M, N)

    print(f"Expansion factor Z      : {Z}")
    print(f"Expected H shape        : ({M}, {N})")
    print(f"Computed H shape        : {h_matrix.shape}")
    print(f"H dtype                 : {h_matrix.dtype}")
    print(f"Number of ones in H     : {int(h_matrix.sum())}")
    print_status("Parity-check matrix size check", h_shape_ok)

    if not h_shape_ok:
        raise ValueError(
            f"Unexpected H shape: {h_matrix.shape}. Expected: {(M, N)}"
        )

    # ------------------------------------------------------------------
    # 3) SYNDROME CHECKER TESTS
    # ------------------------------------------------------------------
    print_section("3. SYNDROME CHECKER VALIDATION")

    zero_codeword = np.zeros(N, dtype=np.uint8)
    zero_syndrome = compute_syndrome(h_matrix, zero_codeword)
    zero_ok = is_codeword(h_matrix, zero_codeword)

    random_vector = rng.integers(0, 2, size=N, dtype=np.uint8)
    random_syndrome = compute_syndrome(h_matrix, random_vector)
    random_ok = not is_codeword(h_matrix, random_vector)

    print(f"Zero vector syndrome weight      : {int(zero_syndrome.sum())}")
    print_status("All-zero vector is a valid codeword", zero_ok)

    print(f"Random vector syndrome weight    : {int(random_syndrome.sum())}")
    print_status("Random vector is rejected", random_ok)

    # ------------------------------------------------------------------
    # 4) ENCODER TEST
    # ------------------------------------------------------------------
    print_section("4. SYSTEMATIC QC-LDPC BASE-MATRIX ENCODER TEST")

    message_bits = rng.integers(0, 2, size=K, dtype=np.uint8)

    # Encoder uses the base matrix, not the full expanded H matrix.
    codeword = encode_message_qc(base_matrix, message_bits, Z)
    codeword_length_ok = codeword.shape[0] == N
    print_status("Codeword length check", codeword_length_ok)

    # Full H is used only for validation.
    codeword_syndrome = compute_syndrome(h_matrix, codeword)

    encoder_ok = is_codeword(h_matrix, codeword)

    print(f"Message length           : {message_bits.shape[0]}")
    print(f"Codeword length          : {codeword.shape[0]}")
    print(f"Codeword syndrome weight : {int(codeword_syndrome.sum())}")
    print_status("Encoded vector is a valid codeword", encoder_ok)

    print("Message bits sample      :", format_vector(message_bits, length=20))
    print("Parity bits sample       :", format_vector(codeword[K:], length=20))

    # ------------------------------------------------------------------
    # 5) CHANNEL MODEL TEST
    # ------------------------------------------------------------------
    print_section("5. CHANNEL MODEL TEST (BSC + LLR)")

    crossover_prob = BSC_CROSSOVER_PROB

    received_bits, error_mask = add_bsc_noise(
        codeword,
        crossover_prob,
        rng=rng
    )

    llr = llr_bsc_quantized(
        received_bits,
        BSC_LLR_MAGNITUDE,
        LLR_SATURATION_LIMIT
    )
    hard_bits_llr = hard_decision_from_llr(llr)

    raw_channel_bit_errors = int(np.sum(received_bits != codeword))
    error_mask_matches_errors = raw_channel_bit_errors == int(error_mask.sum())

    llr_consistency_errors = int(np.sum(received_bits != hard_bits_llr))

    channel_lengths_ok = (
        received_bits.shape[0] == N and
        llr.shape[0] == N
    )

    llr_consistency_ok = llr_consistency_errors == 0

    llr_saturation_ok = np.all(
        (llr >= -LLR_SATURATION_LIMIT) &
        (llr <= LLR_SATURATION_LIMIT)
    )

    print(f"BSC crossover probability p : {crossover_prob:.4f}")
    print(f"Raw channel bit errors      : {raw_channel_bit_errors}")
    print(f"Error mask weight           : {int(error_mask.sum())}")
    print_status("Error mask matches bit errors", error_mask_matches_errors)
    print_status("Channel vector lengths are correct", channel_lengths_ok)
    print_status("Hard decisions agree with LLR signs", llr_consistency_ok)
    print_status("LLR values respect saturation", llr_saturation_ok)

    print("Codeword bits sample        :", format_vector(codeword, length=10))
    print("Received bits sample        :", format_vector(received_bits, length=10))
    print("Error mask sample           :", format_vector(error_mask, length=10))
    print("LLR sample                  :", format_vector(llr, length=10, decimals=4))
    print("Hard bits from LLR          :", format_vector(hard_bits_llr, length=10))

    # ------------------------------------------------------------------
    # 6) FLOODED MIN-SUM DECODER TEST
    # ------------------------------------------------------------------
    print_section("6. FLOODED MIN-SUM DECODER TEST")

    decoder = FloodedMinSumDecoder(
        base_matrix=base_matrix,
        z=Z,
        h_matrix=h_matrix,
        max_iterations=100,
        saturation_limit=31,
    )

    decoded_bits, decoder_success, decoder_iterations = decoder.decode(llr)

    decoded_syndrome = compute_syndrome(h_matrix, decoded_bits)
    remaining_bit_errors = int(np.sum(decoded_bits != codeword))

    print(f"Decoder success          : {decoder_success}")
    print(f"Iterations used          : {decoder_iterations}")
    print(f"Decoded syndrome weight  : {int(decoded_syndrome.sum())}")
    print(f"Remaining bit errors     : {remaining_bit_errors}")
    print_status("Flooded Min-Sum decoder works", decoder_success)

    # ------------------------------------------------------------------
    # 7) GALLAGHER-B DECODER
    # ------------------------------------------------------------------
    run_gallagher_b_section(
        h_matrix=h_matrix,
        base_matrix=base_matrix,
        codeword=codeword,
        received_bits=received_bits,
        rng=rng,
    )

    # ------------------------------------------------------------------
    # 8) FINAL SUMMARY
    # ------------------------------------------------------------------
    print_section("8. FINAL SUMMARY")

    overall_ok = all([
        base_shape_ok,
        h_shape_ok,
        zero_ok,
        random_ok,
        encoder_ok,
        channel_lengths_ok,
        codeword_length_ok,
        llr_consistency_ok,
        error_mask_matches_errors,
        llr_saturation_ok,
        decoder_success
    ])

    print_status("Base matrix parsed correctly", base_shape_ok)
    print_status("Parity-check matrix built correctly", h_shape_ok)
    print_status("Syndrome checker works", zero_ok and random_ok)
    print_status("Encoder works", encoder_ok)
    print_status("Channel model works",channel_lengths_ok and llr_consistency_ok and error_mask_matches_errors and llr_saturation_ok)
    print_status("Overall pipeline status", overall_ok)


if __name__ == "__main__":
    main()