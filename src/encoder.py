import numpy as np

from qc_matrix import expand_base_matrix

def split_base_matrix(
    base_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Split the QC-LDPC base matrix into [A_base | P_base].

    For a systematic encoder:
        H = [A | P]

    In base-matrix form:
        base_matrix = [A_base | P_base]

    If the base matrix has shape mb x nb, then:
        number of parity block columns = mb
        number of information block columns = nb - mb
    """
    if base_matrix.ndim != 2:
        raise ValueError("base_matrix must be a 2D array.")

    m_blocks, n_blocks = base_matrix.shape
    k_blocks = n_blocks - m_blocks

    if k_blocks <= 0:
        raise ValueError("Invalid base-matrix shape: n_blocks must be larger than m_blocks.")

    A_base = base_matrix[:, :k_blocks].copy()
    P_base = base_matrix[:, k_blocks:].copy()

    if P_base.shape != (m_blocks, m_blocks):
        raise ValueError(
            f"P_base must be square with shape ({m_blocks}, {m_blocks}), "
            f"got {P_base.shape}."
        )

    return A_base, P_base, k_blocks


def qc_base_multiply(
    base_submatrix: np.ndarray,
    vector_bits: np.ndarray,
    z: int,
) -> np.ndarray:
    """
    Multiply a QC base submatrix by a binary vector over GF(2),
    without constructing the full expanded matrix.

    Each base-matrix entry represents:
        -1  -> all-zero z x z block
        >=0 -> circulant identity block shifted by that amount

    This function computes:

        result = base_submatrix_expanded @ vector_bits mod 2

    using cyclic shifts directly.
    """
    if base_submatrix.ndim != 2:
        raise ValueError("base_submatrix must be a 2D array.")

    if vector_bits.ndim != 1:
        raise ValueError("vector_bits must be a 1D array.")

    if z <= 0:
        raise ValueError("z must be positive.")

    if not np.all((vector_bits == 0) | (vector_bits == 1)):
        raise ValueError("vector_bits must contain only 0 or 1.")

    m_blocks, n_blocks = base_submatrix.shape

    expected_length = n_blocks * z
    if vector_bits.shape[0] != expected_length:
        raise ValueError(
            f"vector_bits must have length {expected_length}, "
            f"got {vector_bits.shape[0]}."
        )

    result = np.zeros(m_blocks * z, dtype=np.uint8)

    for row_idx in range(m_blocks):
        row_accumulator = np.zeros(z, dtype=np.uint8)

        for col_idx in range(n_blocks):
            shift = int(base_submatrix[row_idx, col_idx])

            if shift == -1:
                continue

            if shift < -1:
                raise ValueError(f"Invalid base-matrix entry: {shift}")

            vector_block = vector_bits[col_idx * z:(col_idx + 1) * z]

            shifted_block = np.roll(vector_block, shift % z)

            row_accumulator ^= shifted_block.astype(np.uint8)

        result[row_idx * z:(row_idx + 1) * z] = row_accumulator

    return result


def gf2_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve A x = b over GF(2) using Gaussian elimination.

    This is used only on the parity part P, not on the full H matrix.
    """
    if A.ndim != 2:
        raise ValueError("A must be a 2D array.")

    if b.ndim != 1:
        raise ValueError("b must be a 1D array.")

    if A.shape[0] != b.shape[0]:
        raise ValueError(
            f"Dimension mismatch: A has {A.shape[0]} rows, "
            f"b has length {b.shape[0]}."
        )

    A = (A % 2).astype(np.uint8)
    b = (b % 2).astype(np.uint8)

    m, n = A.shape
    aug = np.hstack((A.copy(), b.reshape(-1, 1)))

    pivot_row = 0
    pivot_cols: list[int] = []

    for col in range(n):
        row = None

        for r in range(pivot_row, m):
            if aug[r, col] == 1:
                row = r
                break

        if row is None:
            continue

        if row != pivot_row:
            aug[[pivot_row, row]] = aug[[row, pivot_row]]

        for r in range(m):
            if r != pivot_row and aug[r, col] == 1:
                aug[r, :] ^= aug[pivot_row, :]

        pivot_cols.append(col)
        pivot_row += 1

        if pivot_row == m:
            break

    for r in range(m):
        if np.all(aug[r, :n] == 0) and aug[r, n] == 1:
            raise ValueError("System has no solution over GF(2).")

    if len(pivot_cols) < n:
        raise ValueError("System does not have a unique solution over GF(2).")

    x = np.zeros(n, dtype=np.uint8)

    for r, col in enumerate(pivot_cols):
        x[col] = aug[r, n]

    return x


def encode_message_qc(
    base_matrix: np.ndarray,
    message_bits: np.ndarray,
    z: int,
) -> np.ndarray:
    """
    Systematic QC-LDPC encoder using the base matrix directly.

    The codeword has the form:

        c = [u | p]

    The parity bits are found from:

        P p = A u  mod 2

    but A u is computed using the base matrix and cyclic shifts,
    not by multiplying with the full expanded H matrix.
    """
    if message_bits.ndim != 1:
        raise ValueError("message_bits must be a 1D array.")

    if not np.all((message_bits == 0) | (message_bits == 1)):
        raise ValueError("message_bits must contain only 0 or 1.")

    A_base, P_base, k_blocks = split_base_matrix(base_matrix)

    expected_k = k_blocks * z

    if message_bits.shape[0] != expected_k:
        raise ValueError(
            f"message_bits must have length {expected_k}, "
            f"got {message_bits.shape[0]}."
        )

    # Compute A u using only the base matrix and cyclic shifts.
    rhs = qc_base_multiply(A_base, message_bits, z)

    # Expand only the small parity part P_base, not the full H matrix.
    P = expand_base_matrix(P_base, z)

    parity_bits = gf2_solve(P, rhs)

    codeword = np.concatenate((message_bits.astype(np.uint8), parity_bits))

    return codeword.astype(np.uint8)