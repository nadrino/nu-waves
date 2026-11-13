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

    def probabilityWithGradientCompiled(self, compiledBatch, parameterBasis):
        raise NotImplementedError("This executor does not support analytic gradients.")


def evaluateCompiled(executor, compiledBatch, parameterBasis=None):
    """Common event evaluation; no tangent allocations for probability alone."""
    from nu_waves.globals.backend import Backend
    from nu_waves.propagation.gradient import prepareModel, probabilityGradient
    from nu_waves.utils.units import KM_TO_EVINV, GEV_TO_EV

    xp = Backend.xp()
    model = prepareModel(executor.hamiltonian, parameterBasis)
    prepared = executor.prepareModel(model)
    if hasattr(executor, "_prepareCompiledBatch"):
        compiledBatch = executor._prepareCompiledBatch(compiledBatch)
    out = xp.empty((compiledBatch.n_events,), dtype=Backend.real_dtype())
    gradients = None
    if parameterBasis is not None:
        gradients = xp.empty((compiledBatch.n_events, model[2].shape[0]), dtype=Backend.real_dtype())
    for group in compiledBatch.groups:
        for start in range(0, len(group.indices), 65536):
            sl = slice(start, start + 65536)
            indices = xp.asarray(group.indices[sl])
            if hasattr(group, "inv2E"):
                length = group.L[sl]
                energy = 0.5 / group.inv2E[sl]
            else:
                length = xp.asarray(group.L_km[sl], dtype=Backend.real_dtype()) * KM_TO_EVINV
                energy = xp.asarray(group.E_GeV[sl], dtype=Backend.real_dtype()) * GEV_TO_EV
            amplitude, tangents = executor.amplitude(group, length, energy, prepared)
            out[indices] = xp.abs(amplitude) ** 2
            if gradients is not None:
                for column, tangent in enumerate(tangents):
                    gradients[indices, column] = probabilityGradient(amplitude, tangent)
    if gradients is None:
        return Backend.from_device(out)
    return Backend.from_device(out), Backend.from_device(gradients)
