from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class ProbabilityExecutor(ABC):
    oscillator: any

    @property
    def hamiltonian(self):
        return self.oscillator.hamiltonian

    @abstractmethod
    def probability_compiled(self, compiled_batch):
        ...


class GroupedEventExecutor(ProbabilityExecutor):
    def probability_compiled(self, compiled_batch):
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
    pass


class ConstantMatterExecutor(GroupedEventExecutor):
    pass


class LayeredMatterExecutor(GroupedEventExecutor):
    pass
