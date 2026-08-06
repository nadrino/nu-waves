"""Correlation witnesses for the single-excitation flavor-mode encoding."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FlavorModeCorrelations:
    """Flavor-mode quantities derived from a flavor density matrix.

    ``populations[..., alpha]`` is the probability for flavor mode ``alpha``.
    ``linear_entropy[..., alpha]`` quantifies entanglement between that mode
    and all remaining flavor modes for a pure single-neutrino state.  The
    pairwise concurrence matrix is defined after tracing out the other modes.
    """

    populations: np.ndarray
    linear_entropy: np.ndarray
    one_mode_entropy: np.ndarray
    pairwise_concurrence: np.ndarray
    trace: np.ndarray


def _binary_entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(clipped > 0.0, clipped * np.log2(clipped), 0.0)
        complement = 1.0 - clipped
        terms += np.where(complement > 0.0, complement * np.log2(complement), 0.0)
    return -terms


def flavor_mode_correlations(rho) -> FlavorModeCorrelations:
    """Calculate correlations in the flavor-mode single-excitation encoding.

    ``rho`` is a flavor density matrix with shape ``(..., n_flavors,
    n_flavors)``.  The mapping is ``|nu_e> -> |100...>``, etc.; no explicit
    ``2**n_flavors`` density matrix is allocated.  This keeps the calculation
    cheap while yielding the same one-mode and pairwise reduced states.
    """
    rho = np.asarray(rho, dtype=np.complex128)
    if rho.ndim < 2 or rho.shape[-1] != rho.shape[-2]:
        raise ValueError("rho must have shape (..., n_flavors, n_flavors).")

    populations = np.real(np.diagonal(rho, axis1=-2, axis2=-1))
    trace = np.sum(populations, axis=-1)
    if not np.allclose(trace, 1.0, atol=1e-10):
        raise ValueError("rho must have unit trace in the flavor subspace.")

    # For a single excitation, rho_alpha = diag(1-p_alpha, p_alpha).
    linear_entropy = 4.0 * populations * (1.0 - populations)
    one_mode_entropy = _binary_entropy(populations)
    pairwise_concurrence = 2.0 * np.abs(rho)
    diagonal = np.arange(rho.shape[-1])
    pairwise_concurrence[..., diagonal, diagonal] = 0.0

    return FlavorModeCorrelations(
        populations=populations,
        linear_entropy=linear_entropy,
        one_mode_entropy=one_mode_entropy,
        pairwise_concurrence=pairwise_concurrence,
        trace=trace,
    )
