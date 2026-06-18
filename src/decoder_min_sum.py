import numpy as np

from graph_builder import build_tanner_graph
from channel import hard_decision_from_llr
from syndrome import is_codeword


class FloodedMinSumDecoder:
    """
    Flooded Min-Sum decoder for QC-LDPC codes.

    This implementation uses edge-based message passing:
        variable node -> check node messages: msg_v2c
        check node -> variable node messages: msg_c2v

    The schedule is flooded:
        1. update all check nodes
        2. update all variable nodes
        3. make hard decision
        4. check syndrome
    """

    def __init__(
        self,
        base_matrix: np.ndarray,
        z: int,
        h_matrix: np.ndarray,
        max_iterations: int = 30,
        saturation_limit: int = 7,
    ):
        self.base_matrix = base_matrix
        self.z = z
        self.h_matrix = h_matrix
        self.max_iterations = max_iterations
        self.saturation_limit = saturation_limit

        (
            self.check_edges,
            self.var_edges,
            self.edge_to_check,
            self.edge_to_var,
        ) = build_tanner_graph(base_matrix, z)

        self.n_check_nodes = len(self.check_edges)
        self.n_variable_nodes = len(self.var_edges)
        self.n_edges = len(self.edge_to_check)

    def _clip(self, values: np.ndarray) -> np.ndarray:
        return np.clip(
            values,
            -self.saturation_limit,
            self.saturation_limit
        ).astype(np.int16)

    def decode(
        self,
        channel_llr: np.ndarray,
    ) -> tuple[np.ndarray, bool, int]:
        """
        Decode using flooded Min-Sum.

        Parameters
        ----------
        channel_llr : np.ndarray
            Input LLRs of length N.
            Positive LLR means bit 0 is more likely.
            Negative LLR means bit 1 is more likely.

        Returns
        -------
        decoded_bits : np.ndarray
            Decoded binary codeword.

        success : bool
            True if syndrome becomes zero.

        iterations_used : int
            Number of iterations used.
        """

        if channel_llr.ndim != 1:
            raise ValueError("channel_llr must be a 1D array.")

        if channel_llr.shape[0] != self.n_variable_nodes:
            raise ValueError(
                f"channel_llr must have length {self.n_variable_nodes}, "
                f"got {channel_llr.shape[0]}."
            )

        channel_llr = self._clip(channel_llr.astype(np.int16))

        # Initialize variable-to-check messages with channel LLRs.
        msg_v2c = np.zeros(self.n_edges, dtype=np.int16)
        msg_c2v = np.zeros(self.n_edges, dtype=np.int16)

        for edge in range(self.n_edges):
            v = self.edge_to_var[edge]
            msg_v2c[edge] = channel_llr[v]

        posterior_llr = channel_llr.copy()

        # Initial hard decision before iterations.
        decoded_bits = hard_decision_from_llr(posterior_llr)

        if is_codeword(self.h_matrix, decoded_bits):
            return decoded_bits, True, 0

        for iteration in range(1, self.max_iterations + 1):

            # ----------------------------------------------------------
            # 1. CHECK NODE UPDATE
            # ----------------------------------------------------------
            for check_node_edges in self.check_edges:

                if len(check_node_edges) == 0:
                    continue

                incoming = msg_v2c[check_node_edges].astype(np.int16)

                signs = np.where(incoming >= 0, 1, -1).astype(np.int16)
                magnitudes = np.abs(incoming).astype(np.int16)

                total_sign = int(np.prod(signs))

                min_index = int(np.argmin(magnitudes))
                min_value = int(magnitudes[min_index])

                if len(magnitudes) == 1:
                    second_min_value = min_value
                else:
                    temp = magnitudes.copy()
                    temp[min_index] = np.iinfo(np.int16).max
                    second_min_value = int(np.min(temp))

                for local_idx, edge in enumerate(check_node_edges):

                    edge_sign = int(signs[local_idx])

                    outgoing_sign = total_sign * edge_sign

                    if local_idx == min_index:
                        outgoing_magnitude = second_min_value
                    else:
                        outgoing_magnitude = min_value

                    msg_c2v[edge] = outgoing_sign * outgoing_magnitude

            msg_c2v = self._clip(msg_c2v)

            # ----------------------------------------------------------
            # 2. VARIABLE NODE UPDATE
            # ----------------------------------------------------------
            posterior_llr = channel_llr.astype(np.int16).copy()

            for edge in range(self.n_edges):
                v = self.edge_to_var[edge]
                posterior_llr[v] += msg_c2v[edge]

            posterior_llr = self._clip(posterior_llr)

            for variable_node_edges in self.var_edges:

                if len(variable_node_edges) == 0:
                    continue

                for edge in variable_node_edges:
                    v = self.edge_to_var[edge]
                    msg_v2c[edge] = posterior_llr[v] - msg_c2v[edge]

            msg_v2c = self._clip(msg_v2c)

            # ----------------------------------------------------------
            # 3. HARD DECISION + SYNDROME CHECK
            # ----------------------------------------------------------
            decoded_bits = hard_decision_from_llr(posterior_llr)

            if is_codeword(self.h_matrix, decoded_bits):
                return decoded_bits, True, iteration

        return decoded_bits, False, self.max_iterations