from nu_waves.hamiltonian.base import HamiltonianBase
from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.state.wave_function import WaveFunction, Basis
from nu_waves.globals.backend import Backend
from nu_waves.hamiltonian.executors import GroupedEventExecutor
from nu_waves.utils.units import GEV_TO_EV, KM_TO_EVINV


class VacuumExecutor(GroupedEventExecutor):
    def prepareModel(self, model, parameterBasis=None):
        return model

    def amplitude(self, group, length, energy, model):
        xp = Backend.xp()
        matrix, masses, dMatrix, dMasses = model
        if group.isAntiNu:
            matrix = xp.conjugate(matrix)
            if dMatrix is not None:
                dMatrix = xp.conjugate(dMatrix)
        return self.hamiltonian.vacuumAmplitude(length, energy, matrix, masses, group.flavor_emit,
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
        D = self.vacuumPhases(L, E, self._spectrum.get_m2())[:, None, :]

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
        D = self.vacuumPhases(L, E, self._spectrum.get_m2())

        S = (U[None, :, :] * D[:, None, :]) @ Ud
        return S

    @staticmethod
    def vacuumPhases(length, energy, masses):
        """Mass-basis phases shared by state and event propagation."""
        return Backend.xp().exp(-0.5j * (length / energy)[:, None] * masses[None, :])

    @staticmethod
    def vacuumAmplitude(length, energy, matrix, masses, emit, detect, dMatrix=None, dMasses=None, isAntiNu=None):
        """Evaluate a flavor amplitude and optional parameter tangents."""
        xp = Backend.xp()
        phases = Hamiltonian.vacuumPhases(length, energy, masses)
        coefficients = xp.conjugate(matrix[emit]) * matrix[detect]
        if isAntiNu is not None:
            coefficients = xp.where(isAntiNu[:, None], xp.conjugate(coefficients), coefficients)
        amplitude = xp.sum(phases * coefficients, axis=1)
        if dMatrix is None:
            return amplitude, None
        tangents = []
        for column in range(dMatrix.shape[0]):
            dc = xp.conjugate(dMatrix[column, emit]) * matrix[detect] + xp.conjugate(matrix[emit]) * dMatrix[column, detect]
            if isAntiNu is not None:
                dc = xp.where(isAntiNu[:, None], xp.conjugate(dc), dc)
            tangent = xp.sum(phases * dc, axis=1)
            tangent = tangent - 0.5j * (length / energy) * xp.sum(phases * (coefficients * dMasses[column]), axis=1)
            tangents.append(tangent)
        return amplitude, tangents

    @staticmethod
    def vacuumMatrix(matrix, masses, dMatrix=None, dMasses=None):
        """Flavor-basis mass-squared operator, also used by matter Hamiltonians."""
        xp = Backend.xp()
        adjoint = xp.matrix_transpose(xp.conjugate(matrix))
        value = (matrix * masses) @ adjoint
        if dMatrix is None:
            return value, None
        tangent = ((dMatrix * masses) @ adjoint + (matrix * dMasses[:, None, :]) @ adjoint
                   + (matrix * masses) @ xp.matrix_transpose(xp.conjugate(dMatrix)))
        return value, tangent
