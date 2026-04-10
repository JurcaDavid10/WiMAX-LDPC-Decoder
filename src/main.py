from config import BMAT_PATH, Z, NB_ROWS, NB_COLS, M, N
from base_matrix import load_base_matrix
from qc_matrix import expand_base_matrix


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


if __name__ == "__main__":
    main()