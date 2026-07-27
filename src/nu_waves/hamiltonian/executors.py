from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from nu_waves.globals.backend import Backend
from nu_waves.utils.units import GEV_TO_EV, KM_TO_EVINV


@dataclass
class ProbabilityExecutor(ABC):
    oscillator: any

    @property
    def hamiltonian(self):
        return self.oscillator.hamiltonian

    @abstractmethod
    def probabilityCompiled(self, compiled_batch):
        ...


class GroupedEventExecutor(ProbabilityExecutor):
    def probabilityCompiled(self, compiled_batch):
        out = np.empty(compiled_batch.n_events, dtype=float)
        original_antineutrino = self.hamiltonian._antineutrino

        try:
            for group in compiled_batch.groups:
                self.hamiltonian.set_antineutrino(group.isAntiNu)
                probs = self.oscillator._probability_legacy(
                    L_km=group.L_km,
                    E_GeV=group.E_GeV,
                    flavor_emit=group.flavor_emit,
                    flavor_det=group.flavor_det,
                )
                out[group.indices] = np.asarray(probs, dtype=float).reshape(-1)
        finally:
            self.hamiltonian.set_antineutrino(original_antineutrino)

        return out


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


class ConstantMatterExecutor(GroupedEventExecutor):
    pass


class LayeredMatterExecutor(GroupedEventExecutor):
    pass
