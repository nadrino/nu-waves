from nu_waves.hamiltonian.base import HamiltonianBase
from nu_waves.hamiltonian.vacuum import Hamiltonian as VacuumHamiltonian
from nu_waves.models.spectrum import Spectrum
from nu_waves.state.wave_function import WaveFunction, Basis
from nu_waves.globals.backend import Backend
from nu_waves.models.mixing import Mixing
from nu_waves.utils.units import VCOEFF_EV, KM_TO_EVINV, GEV_TO_EV
from nu_waves.hamiltonian.executors import GroupedEventExecutor

from dataclasses import dataclass
import numpy as np
import time


@dataclass
class MatterLayer:
    rho_in_g_per_cm3: float     # matter density
    Ye: float                   # avg number of electrons per nucleon (0.5 by default)
    weight: float               # either absolute length [km] or fraction [0..1]


@dataclass
class MatterProfile:
    layers: list[MatterLayer]
    slicing: str  # "fraction" or "absolute"

    @staticmethod
    def from_fractions(rho_gcm3, Ye, fractions) -> "MatterProfile":
        # Profile metadata stays on CPU; resolve_dL creates device arrays.
        Ls = [MatterLayer(r, y, float(w)) for r, y, w in zip(rho_gcm3, Ye, fractions)]
        return MatterProfile(Ls, "fraction")

    @staticmethod
    def from_segments(rho_gcm3, Ye, lengths_km) -> "MatterProfile":
        Ls = [MatterLayer(r, y, L) for r, y, L in zip(rho_gcm3, Ye, lengths_km)]
        return MatterProfile(Ls, "absolute")

    def resolve_dL(self, L_in_eV_inv):
        """
        Map total baselines (array-like) to per-layer ΔL_k arrays.
        Returns list of arrays, one per layer, each shaped like L_total_km_array.
        - fraction layers scale with L_total
        - absolute layers ignore L_total (assumed consistent with physics setup)
        """
        Ltot = Backend.xp().asarray(L_in_eV_inv, Backend.real_dtype())
        dLs = list()
        if self.slicing == "fraction":
            for layer in self.layers:
                dLs.append(layer.weight * Ltot)
        elif self.slicing == "absolute":
            for layer in self.layers:
                dLs.append(Backend.xp().full_like(Ltot, layer.weight * KM_TO_EVINV, dtype=Backend.real_dtype()))
        else:
            raise NotImplementedError("slicing must be 'fraction' or 'absolute'")

        return dLs


@dataclass(frozen=True, slots=True)
class PreparedMatterEventGroup:
    indices: any
    L: any
    inv2E: any
    flavor_emit: int
    flavor_det: int
    isAntiNu: bool


@dataclass(frozen=True, slots=True)
class CompiledConstantMatterBatch:
    n_events: int
    groups: tuple[PreparedMatterEventGroup, ...]


class ConstantMatterOptimizedExecutor(GroupedEventExecutor):
    def __init__(self, oscillator):
        super().__init__(oscillator=oscillator)
        self._preparedCache = {}

    def prepareModel(self, model, parameterBasis=None):
        vacuum, tangent = VacuumHamiltonian.vacuumMatrix(*model)
        potentialTangent = None
        if parameterBasis is not None:
            rho, fraction = self.hamiltonian.constantDensity
            potentialTangent = self.hamiltonian.potentialTangents(rho, fraction, parameterBasis)
        return vacuum, tangent, potentialTangent

    def amplitude(self, group, length, energy, model):
        vacuum, tangent, potentialTangent = model
        rho, fraction = self.hamiltonian.constantDensity
        propagator, derivatives = self.hamiltonian.matterEvolution(
            length, 0.5 / energy, vacuum, rho, fraction,
            group.isAntiNu, tangent, potentialTangent)
        a, b = group.flavor_emit, group.flavor_det
        return propagator[:, b, a], (None if derivatives is None else [d[:, b, a] for d in derivatives])

    def probabilityCompiled(self, compiled_batch):
        from nu_waves.hamiltonian.executors import evaluateCompiled
        return evaluateCompiled(self, compiled_batch)

    def probabilityWithGradientCompiled(self, compiledBatch, parameterBasis):
        from nu_waves.hamiltonian.executors import evaluateCompiled
        return evaluateCompiled(self, compiledBatch, parameterBasis)

    def _prepareCompiledBatch(self, compiled_batch):
        if isinstance(compiled_batch, CompiledConstantMatterBatch):
            return compiled_batch

        cache_key = (
            id(compiled_batch),
            getattr(Backend.xp(), "__name__", Backend.xp().__class__.__name__),
            str(getattr(Backend.xp(), "device", None)),
        )
        cached = self._preparedCache.get(cache_key)
        if cached is not None and cached[0] is compiled_batch:
            return cached[1]

        xp = Backend.xp()
        groups = []
        for group in compiled_batch.groups:
            E = xp.asarray(group.E_GeV, dtype=Backend.real_dtype()) * GEV_TO_EV
            groups.append(PreparedMatterEventGroup(
                indices=xp.asarray(group.indices),
                L=xp.asarray(group.L_km, dtype=Backend.real_dtype()) * KM_TO_EVINV,
                inv2E=0.5 / E,
                flavor_emit=group.flavor_emit,
                flavor_det=group.flavor_det,
                isAntiNu=group.isAntiNu,
            ))

        prepared = CompiledConstantMatterBatch(
            n_events=compiled_batch.n_events,
            groups=tuple(groups),
        )
        # Keep the source alive: Python can otherwise reuse its id for a
        # different batch, returning stale energies and baselines.
        self._preparedCache[cache_key] = (compiled_batch, prepared)
        return prepared


class ConstantMatterExecutor(ConstantMatterOptimizedExecutor):
    pass


class LayeredMatterExecutor(ConstantMatterOptimizedExecutor):
    def prepareModel(self, model, parameterBasis=None):
        vacuum, tangent = VacuumHamiltonian.vacuumMatrix(*model)
        potentials = None
        if parameterBasis is not None:
            potentials = [self.hamiltonian.potentialTangents(
                layer.rho_in_g_per_cm3, layer.Ye, parameterBasis, index)
                for index, layer in enumerate(self.hamiltonian._matter_profile.layers)]
        return vacuum, tangent, potentials

    def amplitude(self, group, length, energy, model):
        vacuum, tangent, potentials = model
        propagator, derivatives = self.hamiltonian.profileEvolution(
            length, 0.5 / energy, vacuum, group.isAntiNu, tangent, potentials)
        a, b = group.flavor_emit, group.flavor_det
        return propagator[:, b, a], (None if derivatives is None else [d[:, b, a] for d in derivatives])


class Hamiltonian(HamiltonianBase):
    """
        Matter Hamiltonian with Barger layer product.


    """
    def __init__(self, mixing: Mixing, spectrum: Spectrum, antineutrino: bool):
        super().__init__(mixing=mixing, spectrum=spectrum, antineutrino=antineutrino)
        self.enableConstantMatterBatchOptimization = False
        self.enableEighProfiling = False
        self.resetProfiling()
        self._constant_profile = None
        self._matter_profile = None
        self.set_constant_density(rho_in_g_per_cm3=0)

    def makeExecutor(self, oscillator):
        if self._matter_profile is None and self.enableConstantMatterBatchOptimization:
            return ConstantMatterOptimizedExecutor(oscillator=oscillator)
        if self._matter_profile is None:
            return ConstantMatterExecutor(oscillator=oscillator)
        return LayeredMatterExecutor(oscillator=oscillator)

    @property
    def constantDensity(self):
        return self._constant_profile

    def resetProfiling(self):
        self._profiling = {
            "eighCalls": 0,
            "eighEvents": 0,
            "eighTimeSeconds": 0.0,
        }

    def getProfiling(self):
        return dict(self._profiling)

    def _eigh(self, H):
        xp = Backend.xp()
        if not self.enableEighProfiling:
            return xp.linalg.eigh(H)

        t0 = time.perf_counter()
        eigen_values, eigen_vectors = xp.linalg.eigh(H)
        elapsed = time.perf_counter() - t0
        self._profiling["eighCalls"] += 1
        self._profiling["eighEvents"] += int(H.shape[0])
        self._profiling["eighTimeSeconds"] += elapsed
        return eigen_values, eigen_vectors

    def set_constant_density(self, rho_in_g_per_cm3: float, Ye: float = 0.5):
        self._constant_profile = (rho_in_g_per_cm3, Ye)

    def set_matter_profile(self, matter_profile: MatterProfile):
        self._matter_profile = matter_profile

    def get_barger_propagator(self, L, E):
        xp = Backend.xp()

        # make sure they are matching the backend (no copy if already matching)
        E = xp.asarray(E)
        L = xp.asarray(L)

        U = self._mixing.build_mixing_matrix()
        masses = xp.asarray(self.spectrum.get_m2(), dtype=Backend.real_dtype())
        vacuum, _ = VacuumHamiltonian.vacuumMatrix(U, masses)
        inv2E = 0.5 / E

        if self._matter_profile is not None:
            S = self.profileEvolution(L, inv2E, vacuum, self._antineutrino)[0]
        else:
            rho, Ye = self._constant_profile
            S = self.matterEvolution(L, inv2E, vacuum, rho, Ye, self._antineutrino)[0]

        return S

    def potentialTangents(self, rho, fraction, parameterBasis, layerIndex=None):
        """Neutrino potential derivatives for a constant medium or one layer."""
        from nu_waves.propagation.gradient import MatterDensity, ElectronFraction
        xp = Backend.xp()
        tangents = xp.zeros((len(parameterBasis), self.n_neutrinos, self.n_neutrinos),
                            dtype=Backend.complex_dtype())
        for column, parameter in enumerate(parameterBasis):
            if isinstance(parameter, (MatterDensity, ElectronFraction)) and parameter.layerIndex == layerIndex:
                value = fraction if isinstance(parameter, MatterDensity) else rho
                tangents[column, 0, 0] = VCOEFF_EV * value
        return tangents

    def profileEvolution(self, length, inv2Energy, vacuum, antineutrino=False,
                         vacuumTangent=None, potentialTangents=None):
        """Compose layers in traversal order and differentiate their product.

        Layer widths remain fixed. The recurrence uses the previous propagator
        in both terms: d(U S) = dU S + U dS. Probability-only calls allocate no
        tangent matrices and use the same per-layer evolution kernel.
        """
        xp = Backend.xp()
        shape = (length.shape[0], self.n_neutrinos, self.n_neutrinos)
        propagator = xp.copy(xp.broadcast_to(xp.eye(self.n_neutrinos,
                              dtype=Backend.complex_dtype()), shape))
        derivatives = None if vacuumTangent is None else [
            xp.zeros(shape, dtype=Backend.complex_dtype()) for _ in range(vacuumTangent.shape[0])]
        widths = self._matter_profile.resolve_dL(length)
        for index, layer in enumerate(self._matter_profile.layers):
            potential = None if potentialTangents is None else potentialTangents[index]
            step, tangents = self.matterEvolution(widths[index], inv2Energy, vacuum,
                layer.rho_in_g_per_cm3, layer.Ye, antineutrino, vacuumTangent, potential)
            if derivatives is not None:
                derivatives = [tangent @ propagator + step @ previous
                               for tangent, previous in zip(tangents, derivatives)]
            propagator = step @ propagator
        return propagator, derivatives

    def matterEvolution(self, length, inv2Energy, vacuum, rho, electronFraction,
                        antineutrino=False, vacuumTangent=None, potentialTangent=None):
        """Return propagator and optional tangents, sharing the eigendecomposition.

        The Frechet kernel uses normalized sinc and stays finite at degeneracy.
        Backend.linalg.eigh owns device-specific fallbacks (including MPS).
        vacuumTangent contains mass-squared operator derivatives; optional
        potentialTangent contains neutrino potential derivatives in eV. The
        latter receives the antineutrino sign here, without energy scaling.
        """
        xp = Backend.xp()
        sign = -1 if antineutrino else 1
        if antineutrino:
            vacuum = xp.conjugate(vacuum)
        projector = xp.zeros(vacuum.shape, dtype=Backend.complex_dtype())
        projector[0, 0] = sign * VCOEFF_EV * rho * electronFraction
        h = vacuum[None, ...] * inv2Energy[:, None, None] + projector[None, ...]
        values, vectors = self._eigh(h)
        adjoint = xp.matrix_transpose(xp.conjugate(vectors))
        phaseValues = values * length[:, None]
        phases = xp.exp(-1j * phaseValues)
        propagator = (vectors * phases[:, None, :]) @ adjoint
        if vacuumTangent is None:
            return propagator, None
        if antineutrino:
            vacuumTangent = xp.conjugate(vacuumTangent)
        if vacuumTangent.shape[0] == 0:
            return propagator, []
        midpoints = 0.5 * (phaseValues[:, :, None] + phaseValues[:, None, :])
        halfGaps = 0.5 * (phaseValues[:, :, None] - phaseValues[:, None, :])
        kernel = -1j * length[:, None, None] * xp.exp(-1j * midpoints) * xp.sinc(halfGaps / 3.141592653589793)
        tangents = []
        for column in range(vacuumTangent.shape[0]):
            hTangent = vacuumTangent[column] * inv2Energy[:, None, None]
            if potentialTangent is not None:
                hTangent = hTangent + sign * potentialTangent[column]
            spectral = adjoint @ hTangent @ vectors
            tangents.append(vectors @ (kernel * spectral) @ adjoint)
        return propagator, tangents
