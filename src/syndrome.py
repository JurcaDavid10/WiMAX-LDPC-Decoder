import numpy as np


def compute_syndrome(H: np.ndarray, codeword: np.ndarray) -> np.ndarray:
    """
    Compute the syndrome s = H * c^T mod 2.

    Parameters
    ----------
    H : np.ndarray
        Parity-check matrix of shape (M, N).
    codeword : np.ndarray
        Binary vector of length N.

    Returns
    -------
    np.ndarray
        Syndrome vector of length M.
    """
    if H.ndim != 2:
        raise ValueError("H must be a 2D array.")

    if codeword.ndim != 1:
        raise ValueError("codeword must be a 1D array.")

    if H.shape[1] != codeword.shape[0]:
        raise ValueError(
            f"Dimension mismatch: H has {H.shape[1]} columns, "
            f"but codeword has length {codeword.shape[0]}."
        )

    if not np.all((codeword == 0) | (codeword == 1)):
        raise ValueError("codeword must contain only binary values 0 or 1.")

    syndrome = (H @ codeword) % 2
    return syndrome.astype(np.uint8)


def is_codeword(H: np.ndarray, codeword: np.ndarray) -> bool:
    """
    Check whether a binary vector is a valid codeword.

    A valid codeword has zero syndrome.
    """
    syndrome = compute_syndrome(H, codeword)
    return np.all(syndrome == 0)