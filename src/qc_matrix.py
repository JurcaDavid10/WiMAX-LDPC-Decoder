import numpy as np


def circulant_identity(z: int, shift: int) -> np.ndarray:
    """
    Create a z x z identity matrix circularly right-shifted by 'shift'.
    """
    if z <= 0:
        raise ValueError("z must be positive.")

    shift = shift % z
    identity = np.eye(z, dtype=np.uint8)
    return np.roll(identity, shift=shift, axis=1)


def expand_base_matrix(base_matrix: np.ndarray, z: int) -> np.ndarray:
    """
    Expand the QC base matrix into the full binary parity-check matrix H.

    Rules:
    -1  -> z x z all-zero block
    >=0 -> z x z circulant identity block shifted by that amount
    """
    if base_matrix.ndim != 2:
        raise ValueError("base_matrix must be a 2D array.")

    block_rows = []

    for row in base_matrix:
        expanded_blocks = []

        for entry in row:
            if entry == -1:
                block = np.zeros((z, z), dtype=np.uint8)
            elif entry >= 0:
                block = circulant_identity(z, int(entry))
            else:
                raise ValueError(f"Invalid base-matrix entry: {entry}")

            expanded_blocks.append(block)

        block_rows.append(np.hstack(expanded_blocks))

    return np.vstack(block_rows)