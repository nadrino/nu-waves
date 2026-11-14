"""Caller-owned event/output storage for Oscillator's probability interface."""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np

from nu_waves.globals.backend import Backend
from nu_waves.utils.units import KM_TO_EVINV, GEV_TO_EV


def evaluateBuffers(oscillator, events, output, parameterBasis, gradientOutput):
    """Evaluate without scalar device reads, host copies or global backend changes.

    Output shape is (N,); Jacobian shape is (N, len(parameterBasis)). Input
    categories may use floating storage (as in GUNDAM); flavors must contain
    exact zero-based integers and isAntiNu uses the threshold 0.5. The caller
    validates values and guarantees disjoint input/output storage and lifetime.
    Intermediate allocations and device-to-device copies are permitted.
    """
    from nu_waves.hamiltonian.vacuum import Hamiltonian as VacuumHamiltonian
    from nu_waves.hamiltonian.matter import Hamiltonian as MatterHamiltonian
    from nu_waves.propagation.gradient import (
        buildSpectrumGradient, validateParameterBasis, probabilityGradient,
    )

    basis = None if parameterBasis is None else tuple(parameterBasis)
    if gradientOutput is not None and basis is None:
        raise ValueError("gradientOutputBuffer requires parameterBasis.")
    numpyMode = isinstance(output, np.ndarray)
    if numpyMode:
        torch = None
        guard = nullcontext()
        device = None
    else:
        import torch
        if not isinstance(output, torch.Tensor):
            raise TypeError("outputBuffer must be a NumPy array or Torch tensor.")
        guard = torch.no_grad()
        device = output.device
    if output.ndim != 1:
        raise ValueError("outputBuffer must have shape (N,).")

    def checkStorage(value, name, shape=None, real=False, allowScalar=False):
        if allowScalar and isinstance(value, (int, float, bool, np.number)):
            return
        if numpyMode:
            if not isinstance(value, np.ndarray):
                raise TypeError(f"{name} must be a NumPy array, matching outputBuffer.")
        elif not isinstance(value, torch.Tensor) or value.device != device:
            raise ValueError(f"{name} must be a Torch tensor on {device}.")
        if shape is not None and value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}.")
        if real and value.dtype != output.dtype:
            raise TypeError(f"{name} must have the same dtype as outputBuffer.")

    checkStorage(output, "outputBuffer", real=True)
    if numpyMode and not output.flags.writeable:
        raise ValueError("outputBuffer must be writable.")
    for name in ("L_km", "E_GeV", "flavor_emit", "flavor_det"):
        checkStorage(getattr(events, name), name, output.shape, real=name in ("L_km", "E_GeV"))
    if events.isAntiNu is not None:
        checkStorage(events.isAntiNu, "isAntiNu", output.shape)
    if gradientOutput is not None:
        checkStorage(gradientOutput, "gradientOutputBuffer", output.shape + (len(basis),), real=True)
        if numpyMode and not gradientOutput.flags.writeable:
            raise ValueError("gradientOutputBuffer must be writable.")
    hamiltonian = oscillator.hamiltonian
    for mapping in (hamiltonian.mixing.mixing_angles, hamiltonian.mixing.dirac_phases,
                    hamiltonian.spectrum._dm2_dict or {}):
        for value in mapping.values():
            if value is not None:
                checkStorage(value, "model parameter", allowScalar=True)
    matter = isinstance(hamiltonian, MatterHamiltonian)
    if not matter and type(hamiltonian) is not VacuumHamiltonian:
        raise NotImplementedError("Buffer evaluation supports vacuum and matter Hamiltonians.")
    if matter and device is not None and device.type == "mps":
        raise NotImplementedError("Matter buffer evaluation on MPS requires a device-native eigensolver; CPU fallback is disabled.")

    with guard, Backend.useBuffer(output):
        xp = Backend.xp()
        if basis is not None:
            validateParameterBasis(hamiltonian, basis)
            matrix, dMatrix = hamiltonian.mixing.build_mixing_matrix(parameterBasis=basis)
            _, dMasses = buildSpectrumGradient(hamiltonian.spectrum, basis)
        else:
            matrix = hamiltonian.mixing.build_mixing_matrix()
            dMatrix = dMasses = None
        masses = hamiltonian.spectrum.relativeMasses()
        model = matrix, masses, dMatrix, dMasses
        gradients = gradientOutput
        if basis is not None and gradients is None:
            gradients = xp.empty(output.shape + (len(basis),), dtype=Backend.real_dtype())
        # Flavor conversion stays on the device, including GUNDAM's float inputs.
        emit = xp.asarray(events.flavor_emit, dtype=xp.int64)
        detect = xp.asarray(events.flavor_det, dtype=xp.int64)
        anti = events.isAntiNu
        if anti is None:
            anti = xp.full(output.shape, hamiltonian._antineutrino, dtype=bool if numpyMode else xp.bool)
        else:
            anti = anti > 0.5
        executor = hamiltonian.makeExecutor(oscillator)
        prepared = executor.prepareModel(model, basis) if matter else model
        for start in range(0, output.shape[0], 65536):
            sl = slice(start, start + 65536)
            length = events.L_km[sl] * KM_TO_EVINV
            energy = events.E_GeV[sl] * GEV_TO_EV
            if not matter:
                amplitude, tangents = hamiltonian.vacuumAmplitude(
                    length, energy, *model[:2], emit[sl], detect[sl],
                    dMatrix, dMasses, isAntiNu=anti[sl])
                output[sl] = xp.abs(amplitude) ** 2
                if gradients is not None:
                    for column, tangent in enumerate(tangents):
                        gradients[sl, column] = probabilityGradient(amplitude, tangent)
            else:
                # Fixed channel loops avoid device mask reads or CPU grouping.
                for sign in (False, True):
                    for a in range(hamiltonian.n_neutrinos):
                        for b in range(hamiltonian.n_neutrinos):
                            group = SimpleNamespace(isAntiNu=sign, flavor_emit=a, flavor_det=b)
                            amplitude, tangents = executor.amplitude(group, length, energy, prepared)
                            mask = (anti[sl] == sign) & (emit[sl] == a) & (detect[sl] == b)
                            output[sl] = xp.where(mask, xp.abs(amplitude) ** 2, output[sl])
                            if gradients is not None:
                                for column, tangent in enumerate(tangents):
                                    gradients[sl, column] = xp.where(mask, probabilityGradient(amplitude, tangent), gradients[sl, column])
        return output if basis is None else (output, gradients)
