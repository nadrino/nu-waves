from nu_waves.hamiltonian.base import HamiltonianBase
from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.state.wave_function import WaveFunction, Basis
from nu_waves.globals.backend import Backend
from nu_waves.hamiltonian.executors import GroupedEventExecutor
from nu_waves.utils.units import GEV_TO_EV, KM_TO_EVINV


class VacuumExecutor(GroupedEventExecutor):
    def prepareModel(self, model):
        return model

    def amplitude(self, group, length, energy, model):
        from nu_waves.hamiltonian.Evolution import vacuumAmplitude
        xp = Backend.xp()
        matrix, masses, dMatrix, dMasses = model
        if group.isAntiNu:
            matrix = xp.conjugate(matrix)
            if dMatrix is not None:
                dMatrix = xp.conjugate(dMatrix)
        return vacuumAmplitude(length, energy, matrix, masses, group.flavor_emit,
                               group.flavor_det, dMatrix, dMasses)

    def probabilityCompiled(self, compiled_batch):
        from nu_waves.hamiltonian.executors import evaluateCompiled
        return evaluateCompiled(self, compiled_batch)

    def probabilityWithGradientCompiled(self, compiledBatch, parameterBasis):
        from nu_waves.hamiltonian.executors import evaluateCompiled
        return evaluateCompiled(self, compiledBatch, parameterBasis)


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
        if self._antineutrino:
            U = xp.conjugate(U)

        # Rotate psi to mass basis if needed
        if psi.current_basis == Basis.FLAVOR:
            psi.to_basis(
                target_basis=Basis.MASS,
                eigen_vectors=xp.conjugate(U)
            )

        if psi.current_basis != Basis.MASS:
            raise ValueError(f"VacuumHamiltonian: unsupported input basis for psi: {psi.current_basis}")

        # phases φ_i = 1/2 * Δm_i^2[eV^2] * L[eV-1] / E[eV]
        from nu_waves.hamiltonian.Evolution import vacuumPhases
        D = vacuumPhases(L, E, self._spectrum.get_m2())[:, None, :]

        # Diagonal propagation in mass basis
        psi.values = psi.values * D                             # (nE, nFe, nF)

        # Ket coefficients are stored as rows: mass -> flavor uses U.T.
        psi.to_basis(
            target_basis=Basis.FLAVOR,
            eigen_vectors=U.T
        )

    def get_barger_propagator(self, L, E):
        xp = Backend().xp()
        L = xp.asarray(L)
        E = xp.asarray(E)

        U = self._mixing.build_mixing_matrix()
        if self._antineutrino:
            U = xp.conjugate(U)
        Ud = xp.conjugate(U.T)

        # L and E are in natural units, as in propagate_state.
        from nu_waves.hamiltonian.Evolution import vacuumPhases
        D = vacuumPhases(L, E, self._spectrum.get_m2())

        S = (U[None, :, :] * D[:, None, :]) @ Ud
        return S
