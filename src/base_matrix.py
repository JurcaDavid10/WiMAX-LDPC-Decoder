from pathlib import Path
import numpy as np


def load_base_matrix(filepath: str | Path) -> np.ndarray:
    """
    Load the QC-LDPC base matrix from a .bmat file.

    The parser looks for the line 'qc_base_matrix' and then reads all
    following numeric rows until a non-numeric/comment section begins.

    Returns
    -------
    np.ndarray
        2D integer NumPy array containing the base matrix.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Base-matrix file not found: {filepath}")

    with filepath.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    start_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "qc_base_matrix":
            start_idx = i + 1
            break

    if start_idx is None:
        raise ValueError("Could not find 'qc_base_matrix' section in file.")

    matrix_rows: list[list[int]] = []

    for line in lines[start_idx:]:
        stripped = line.strip()

        if not stripped:
            continue

        if stripped.startswith("*"):
            break

        parts = stripped.split()

        try:
            row = [int(x) for x in parts]
        except ValueError:
            break

        matrix_rows.append(row)

    if not matrix_rows:
        raise ValueError("No matrix rows found after 'qc_base_matrix'.")

    row_lengths = {len(row) for row in matrix_rows}
    if len(row_lengths) != 1:
        raise ValueError("Inconsistent row lengths found in base matrix.")

    return np.array(matrix_rows, dtype=int)