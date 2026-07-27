from nu_waves.hamiltonian.base import HamiltonianBase
from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.state.wave_function import WaveFunction, Basis
from nu_waves.globals.backend import Backend
from nu_waves.hamiltonian.executors import GroupedEventExecutor
from nu_waves.utils.units import GEV_TO_EV, KM_TO_EVINV


class VacuumExecutor(GroupedEventExecutor):
    def probabilityCompiled(self, compiled_batch):
        xp = Backend.xp()
        out = xp.empty((compiled_batch.n_events,), dtype=Backend.real_dtype())

        U = self.hamiltonian.mixing.build_mixing_matrix()
        m2 = xp.asarray(self.hamiltonian.spectrum.get_m2(), dtype=Backend.real_dtype())

        for group in compiled_batch.groups:
            Ueff = xp.conjugate(U) if group.isAntiNu else U
            coeff = Ueff[group.flavor_emit, :] * xp.conjugate(Ueff[group.flavor_det, :])

            L = xp.asarray(group.L_km, dtype=Backend.real_dtype()) * KM_TO_EVINV
            E = xp.asarray(group.E_GeV, dtype=Backend.real_dtype()) * GEV_TO_EV
            phases = 0.5 * (L / E)[:, None] * m2[None, :]
            amp = xp.sum(coeff[None, :] * xp.exp((-1j) * phases), axis=1)
            prob = xp.abs(amp) ** 2

            indices = xp.asarray(group.indices)
            out[indices] = prob

        return Backend.from_device(out)


class Hamiltonian(HamiltonianBase):
    def __init__(self, mixing: Mixing, spectrum: Spectrum, antineutrino: bool):
        super().__init__(mixing=mixing, spectrum=spectrum, antineutrino=antineutrino)

    def makeExecutor(self, oscillator):
        return VacuumExecutor(oscillator=oscillator)

    def propagate_state(self, psi: WaveFunction, L, E):
        xp = Backend().xp()

        # make sure they are matching the backend (no copy if already matching)
        E = xp.asarray(E)
        L = xp.asarray(L)

        U = self._mixing.build_mixing_matrix()
        Ud = xp.conjugate(U.T)

        if self._antineutrino:
            U = xp.conjugate(U)
            Ud = xp.conjugate(Ud)

        # Rotate psi to mass basis if needed
        if psi.current_basis == Basis.FLAVOR:
            psi.to_basis(
                target_basis=Basis.MASS,
                eigen_vectors=U
            )

        if psi.current_basis != Basis.MASS:
            raise ValueError(f"VacuumHamiltonian: unsupported input basis for psi: {psi.current_basis}")

        # phases φ_i = 1/2 * Δm_i^2[eV^2] * L[eV-1] / E[eV]
        phases = 0.5 * (L / E)[:, None] * self._spectrum.get_m2()[None, :]     # (nE, nF)
        D = xp.exp(-1j * phases)[:, None, :]                    # (nE, 1, nF)

        # Diagonal propagation in mass basis
        psi.values = psi.values * D                             # (nE, nFe, nF)

        # convention
        psi.to_basis(
            target_basis=Basis.FLAVOR,
            eigen_vectors=Ud
        )

    def get_barger_propagator(self, L, E):
        xp = Backend().xp()

        U = self._mixing.build_mixing_matrix()
        Ud = xp.conjugate(U.T)

        phases = 1.267 * (L / E)[:, None] * self._spectrum.get_m2()[None, :]
        D = xp.exp(-1j * phases)

        S = xp.matmul(U[None, :, :], xp.diag_embed(D))
        S = xp.matmul(S, Ud[None, :, :])
        return S
