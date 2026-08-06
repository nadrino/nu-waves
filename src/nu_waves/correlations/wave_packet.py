"""Vacuum wave-packet decoherence channel."""

from dataclasses import dataclass
import numpy as np

from nu_waves.globals.backend import Backend


METER_TO_EVINV = 5.067730716e6


@dataclass(frozen=True, slots=True)
class GaussianWavePacket:
    """Gaussian mass-wave-packet decoherence in vacuum.

    Parameters
    ----------
    sigma_x_m:
        Spatial width of the produced neutrino wave packet, in metres.

    Notes
    -----
    Off-diagonal mass terms are multiplied by
    ``exp(-(L / L_coh_ij)**2)``, where
    ``L_coh_ij = 4 sqrt(2) E**2 sigma_x / |Delta m2_ij|``.  The input state
    has already accumulated its oscillation phases; this channel only removes
    coherence.  It is therefore exact for the vacuum propagation implemented
    by ``vacuum.Hamiltonian``.  Applying it after a matter evolution is an
    approximation and is rejected deliberately.
    """

    sigma_x_m: float

    def __post_init__(self):
        if self.sigma_x_m <= 0.0:
            raise ValueError("sigma_x_m must be strictly positive.")

    def apply(self, rho, L_eV_inv, E_eV, hamiltonian):
        from nu_waves.hamiltonian.vacuum import Hamiltonian as VacuumHamiltonian

        if not isinstance(hamiltonian, VacuumHamiltonian):
            raise ValueError("GaussianWavePacket currently supports vacuum propagation only.")

        xp = Backend.xp()
        rho = xp.asarray(rho, dtype=Backend.complex_dtype())
        L = xp.asarray(L_eV_inv, dtype=Backend.real_dtype()).reshape(-1)
        E = xp.asarray(E_eV, dtype=Backend.real_dtype()).reshape(-1)

        U = hamiltonian.mixing.build_mixing_matrix()
        if hamiltonian._antineutrino:
            U = xp.conjugate(U)
        Uc = xp.conjugate(U)

        # Row-vector convention used by WaveFunction: b = a @ U.
        rho_mass = xp.matmul(xp.matrix_transpose(U), xp.matmul(rho, Uc))
        m2 = xp.asarray(hamiltonian.spectrum.get_m2(), dtype=Backend.real_dtype())
        dm2 = xp.abs(m2[:, None] - m2[None, :])
        sigma_x = self.sigma_x_m * METER_TO_EVINV
        denominator = dm2[None, :, :]
        l_coh = 4.0 * np.sqrt(2.0) * E[:, None, None] ** 2 * sigma_x
        damping = xp.where(
            denominator > 0.0,
            xp.exp(-((L[:, None, None] * denominator / l_coh) ** 2)),
            xp.ones_like(denominator),
        )
        rho_mass = rho_mass * damping[:, None, :, :]

        Ud = xp.conjugate(xp.matrix_transpose(U))
        return xp.matmul(xp.matrix_transpose(Ud), xp.matmul(rho_mass, xp.conjugate(Ud)))
