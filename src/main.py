import numpy as np

from config import BMAT_PATH, Z, NB_ROWS, NB_COLS, M, N, K
from base_matrix import load_base_matrix
from qc_matrix import expand_base_matrix
from syndrome import compute_syndrome, is_codeword
from encoder import encode_message


def main() -> None:
    base_matrix = load_base_matrix(BMAT_PATH)
    print("Base matrix loaded successfully.")
    print(f"Base matrix shape: {base_matrix.shape}")
    print(base_matrix)

    if base_matrix.shape != (NB_ROWS, NB_COLS):
        raise ValueError(
            f"Unexpected base matrix shape: {base_matrix.shape}. "
            f"Expected: {(NB_ROWS, NB_COLS)}"
        )

    h_matrix = expand_base_matrix(base_matrix, Z)
    print("\nExpanded parity-check matrix H created successfully.")
    print(f"H shape: {h_matrix.shape}")

    if h_matrix.shape != (M, N):
        raise ValueError(
            f"Unexpected H shape: {h_matrix.shape}. Expected: {(M, N)}"
        )

    print("\nSanity checks passed.")
    print(f"H dtype: {h_matrix.dtype}")
    print(f"Number of ones in H: {h_matrix.sum()}")

    # Syndrome checker tests
    zero_codeword = np.zeros(N, dtype=np.uint8)
    zero_syndrome = compute_syndrome(h_matrix, zero_codeword)

    print("\nSyndrome checker tests:")
    print(f"Zero vector syndrome weight: {zero_syndrome.sum()}")
    print(f"Is zero vector a valid codeword? {is_codeword(h_matrix, zero_codeword)}")

    rng = np.random.default_rng(seed=42)
    random_vector = rng.integers(0, 2, size=N, dtype=np.uint8)
    random_syndrome = compute_syndrome(h_matrix, random_vector)

    print(f"Random vector syndrome weight: {random_syndrome.sum()}")
    print(f"Is random vector a valid codeword? {is_codeword(h_matrix, random_vector)}")

    # Encoder test
    message_bits = rng.integers(0, 2, size=K, dtype=np.uint8)
    codeword = encode_message(h_matrix, message_bits, K)
    codeword_syndrome = compute_syndrome(h_matrix, codeword)

    print("\nEncoder test:")
    print(f"Message length: {message_bits.shape[0]}")
    print(f"Codeword length: {codeword.shape[0]}")
    print(f"Codeword syndrome weight: {codeword_syndrome.sum()}")
    print(f"Is encoded vector a valid codeword? {is_codeword(h_matrix, codeword)}")

    print("\nFirst 20 message bits:")
    print(message_bits[:20])

    print("\nFirst 20 parity bits:")
    print(codeword[K:K + 20])


if __name__ == "__main__":
    main()