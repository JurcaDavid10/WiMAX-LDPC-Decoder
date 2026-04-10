import numpy as np


def split_h_matrix(H: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Split H into [A | P], where:
    - A contains the first k columns
    - P contains the remaining columns
    """
    if H.ndim != 2:
        raise ValueError("H must be a 2D array.")

    n = H.shape[1]
    if not (0 < k < n):
        raise ValueError(f"k must satisfy 0 < k < {n}, got {k}.")

    A = H[:, :k].copy()
    P = H[:, k:].copy()
    return A, P


def gf2_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve A x = b over GF(2) using Gaussian elimination.

    Parameters
    ----------
    A : np.ndarray
        Binary matrix of shape (m, n)
    b : np.ndarray
        Binary vector of shape (m,)

    Returns
    -------
    np.ndarray
        Solution vector x of shape (n,)

    Notes
    -----
    This implementation assumes the system has a unique solution.
    For your current encoder usage, P should be square and invertible.
    """
    if A.ndim != 2:
        raise ValueError("A must be a 2D array.")
    if b.ndim != 1:
        raise ValueError("b must be a 1D array.")
    if A.shape[0] != b.shape[0]:
        raise ValueError(
            f"Dimension mismatch: A has {A.shape[0]} rows, b has length {b.shape[0]}."
        )

    A = (A % 2).astype(np.uint8)
    b = (b % 2).astype(np.uint8)

    m, n = A.shape
    aug = np.hstack((A.copy(), b.reshape(-1, 1)))

    pivot_row = 0
    pivot_cols: list[int] = []

    for col in range(n):
        # Find pivot
        row = None
        for r in range(pivot_row, m):
            if aug[r, col] == 1:
                row = r
                break

        if row is None:
            continue

        # Swap pivot row into place
        if row != pivot_row:
            aug[[pivot_row, row]] = aug[[row, pivot_row]]

        # Eliminate this column from all other rows
        for r in range(m):
            if r != pivot_row and aug[r, col] == 1:
                aug[r, :] ^= aug[pivot_row, :]

        pivot_cols.append(col)
        pivot_row += 1

        if pivot_row == m:
            break

    # Check for inconsistency: [0 ... 0 | 1]
    for r in range(m):
        if np.all(aug[r, :n] == 0) and aug[r, n] == 1:
            raise ValueError("System has no solution over GF(2).")

    # Require unique solution
    if len(pivot_cols) < n:
        raise ValueError("System does not have a unique solution over GF(2).")

    x = np.zeros(n, dtype=np.uint8)
    for r, col in enumerate(pivot_cols):
        x[col] = aug[r, n]

    return x


def encode_message(H: np.ndarray, message_bits: np.ndarray, k: int) -> np.ndarray:
    """
    Systematic LDPC encoder:
        c = [u | p]

    where p is found by solving:
        P p = A u  (mod 2)
    with H = [A | P]

    Parameters
    ----------
    H : np.ndarray
        Full parity-check matrix of shape (M, N)
    message_bits : np.ndarray
        Binary vector of length k
    k : int
        Number of information bits

    Returns
    -------
    np.ndarray
        Binary codeword of length N
    """
    if message_bits.ndim != 1:
        raise ValueError("message_bits must be a 1D array.")
    if message_bits.shape[0] != k:
        raise ValueError(
            f"message_bits must have length {k}, got {message_bits.shape[0]}."
        )
    if not np.all((message_bits == 0) | (message_bits == 1)):
        raise ValueError("message_bits must contain only 0 or 1.")

    A, P = split_h_matrix(H, k)

    rhs = (A @ message_bits) % 2
    parity_bits = gf2_solve(P, rhs)

    codeword = np.concatenate((message_bits.astype(np.uint8), parity_bits))
    return codeword.astype(np.uint8)