import numpy as np


def add_bsc_noise(
    bits: np.ndarray,
    crossover_prob: float,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Binary Symmetric Channel.

    Each transmitted bit is flipped independently with probability p.
    """
    if bits.ndim != 1:
        raise ValueError("bits must be a 1D array.")

    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError("bits must contain only 0 or 1.")

    if not (0.0 <= crossover_prob <= 1.0):
        raise ValueError("crossover_prob must satisfy 0 <= p <= 1.")

    if rng is None:
        rng = np.random.default_rng()

    error_mask = rng.random(size=bits.shape) < crossover_prob

    received_bits = bits.copy().astype(np.uint8)
    received_bits[error_mask] ^= 1

    return received_bits, error_mask.astype(np.uint8)


def saturate_llr_values(
    values: np.ndarray,
    saturation_limit: int,
) -> np.ndarray:
    """
    Saturate LLR values to the range:

        -saturation_limit ... +saturation_limit
    """
    if values.ndim != 1:
        raise ValueError("values must be a 1D array.")

    if saturation_limit <= 0:
        raise ValueError("saturation_limit must be positive.")

    return np.clip(
        values,
        -saturation_limit,
        saturation_limit
    ).astype(np.int16)


def llr_bsc_quantized(
    received_bits: np.ndarray,
    llr_magnitude: int,
    saturation_limit: int,
) -> np.ndarray:
    """
    Quantized saturated BSC LLRs.

    received bit 0 -> +LLR magnitude
    received bit 1 -> -LLR magnitude

    The result is saturated to:

        -saturation_limit ... +saturation_limit
    """
    if received_bits.ndim != 1:
        raise ValueError("received_bits must be a 1D array.")

    if not np.all((received_bits == 0) | (received_bits == 1)):
        raise ValueError("received_bits must contain only 0 or 1.")

    if llr_magnitude <= 0:
        raise ValueError("llr_magnitude must be positive.")

    if llr_magnitude % 2 == 0:
        raise ValueError("llr_magnitude should be odd.")

    if saturation_limit <= 0:
        raise ValueError("saturation_limit must be positive.")

    if llr_magnitude > saturation_limit:
        raise ValueError("llr_magnitude must not exceed saturation_limit.")

    raw_llr = llr_magnitude * (1 - 2 * received_bits.astype(np.int16))

    return saturate_llr_values(raw_llr, saturation_limit)


def hard_decision_from_llr(llr: np.ndarray) -> np.ndarray:
    """
    Hard decision from LLR values:
        LLR >= 0 -> 0
        LLR < 0  -> 1
    """
    if llr.ndim != 1:
        raise ValueError("llr must be a 1D array.")

    return (llr < 0).astype(np.uint8)