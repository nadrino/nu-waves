from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


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
        raise NotImplementedError
