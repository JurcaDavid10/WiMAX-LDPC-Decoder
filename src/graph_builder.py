import numpy as np


def build_tanner_graph(
    base_matrix: np.ndarray,
    z: int,
):
    """
    Build Tanner graph connections directly from the QC base matrix.

    Returns
    -------
    check_edges : list[list[int]]
        check_edges[c] -> edge ids connected to check node c

    var_edges : list[list[int]]
        var_edges[v] -> edge ids connected to variable node v

    edge_to_check : np.ndarray
        edge_to_check[e] -> check node index

    edge_to_var : np.ndarray
        edge_to_var[e] -> variable node index
    """

    if base_matrix.ndim != 2:
        raise ValueError("base_matrix must be 2D.")

    n_check_nodes = base_matrix.shape[0] * z
    n_variable_nodes = base_matrix.shape[1] * z

    check_edges = [[] for _ in range(n_check_nodes)]
    var_edges = [[] for _ in range(n_variable_nodes)]

    edge_to_check = []
    edge_to_var = []

    edge_id = 0

    rows, cols = base_matrix.shape

    for br in range(rows):

        for bc in range(cols):

            shift = int(base_matrix[br, bc])

            if shift < 0:
                continue

            for k in range(z):

                check_node = br * z + k

                variable_node = bc * z + (
                    (k - shift) % z
                )

                check_edges[check_node].append(edge_id)

                var_edges[variable_node].append(edge_id)

                edge_to_check.append(check_node)

                edge_to_var.append(variable_node)

                edge_id += 1

    return (
        check_edges,
        var_edges,
        np.array(edge_to_check, dtype=np.int32),
        np.array(edge_to_var, dtype=np.int32),
    )