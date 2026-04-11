import numpy as np


def bpsk_modulate(bits: np.ndarray) -> np.ndarray:
    """
    Map bits to BPSK symbols:
        0 -> +1
        1 -> -1
    """
    if bits.ndim != 1:
        raise ValueError("bits must be a 1D array.")
    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError("bits must contain only 0 or 1.")

    return 1.0 - 2.0 * bits.astype(np.float64)


def awgn_sigma2_from_ebn0(ebn0_db: float, code_rate: float) -> float:
    """
    Compute AWGN noise variance per real dimension for BPSK over AWGN.

    With BPSK symbols normalized to unit energy:
        sigma^2 = 1 / (2 * R * Eb/N0)

    where:
        R = code rate
        Eb/N0 is in linear scale
    """
    if code_rate <= 0 or code_rate > 1:
        raise ValueError("code_rate must satisfy 0 < code_rate <= 1.")

    ebn0_linear = 10 ** (ebn0_db / 10.0)
    sigma2 = 1.0 / (2.0 * code_rate * ebn0_linear)
    return sigma2


def add_awgn_noise(
    symbols: np.ndarray,
    ebn0_db: float,
    code_rate: float,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float]:
    """
    Add AWGN noise to BPSK symbols.

    Returns
    -------
    received : np.ndarray
        Noisy received samples
    sigma2 : float
        Noise variance used
    """
    if symbols.ndim != 1:
        raise ValueError("symbols must be a 1D array.")

    if rng is None:
        rng = np.random.default_rng()

    sigma2 = awgn_sigma2_from_ebn0(ebn0_db, code_rate)
    noise = rng.normal(loc=0.0, scale=np.sqrt(sigma2), size=symbols.shape)
    received = symbols + noise
    return received, sigma2


def llr_awgn(received: np.ndarray, sigma2: float) -> np.ndarray:
    """
    Compute channel LLRs for BPSK over AWGN.

    For mapping:
        0 -> +1
        1 -> -1

    the log-likelihood ratio is:
        LLR = 2y / sigma^2
    """
    if received.ndim != 1:
        raise ValueError("received must be a 1D array.")
    if sigma2 <= 0:
        raise ValueError("sigma2 must be positive.")

    return (2.0 * received) / sigma2


def hard_decision_from_received(received: np.ndarray) -> np.ndarray:
    """
    Hard-decision slicing of received BPSK samples:
        y >= 0 -> 0
        y < 0  -> 1
    """
    if received.ndim != 1:
        raise ValueError("received must be a 1D array.")

    return (received < 0).astype(np.uint8)


def hard_decision_from_llr(llr: np.ndarray) -> np.ndarray:
    """
    Hard decision from LLR values:
        LLR >= 0 -> 0
        LLR < 0  -> 1
    """
    if llr.ndim != 1:
        raise ValueError("llr must be a 1D array.")

    return (llr < 0).astype(np.uint8)