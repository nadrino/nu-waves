"""Caller-owned event/output storage for Oscillator's probability interface."""

from contextlib import nullcontext
from types import SimpleNamespace
import sys

import numpy as np

from nu_waves.globals.backend import Backend
from nu_waves.utils.units import KM_TO_EVINV, GEV_TO_EV


def findDeviceInput(oscillator, *values):
    """Inspect metadata only; Torch remains optional for host callers."""
    torch = sys.modules.get('torch')
    if torch is None:
        return None
    model = oscillator.hamiltonian
    profile = getattr(model, '_matter_profile', None)
    layers = () if profile is None else tuple(
        (layer.rho_in_g_per_cm3, layer.Ye, layer.weight) for layer in profile.layers)
    values = (*values, model.mixing.mixing_angles, model.mixing.dirac_phases,
              model.spectrum._dm2_dict or {}, getattr(model, '_constant_profile', None), layers)

    def visit(value):
        if isinstance(value, torch.Tensor):
            return value if value.device.type != 'cpu' else None
        if isinstance(value, dict):
            value = tuple(value.values())
        if isinstance(value, (tuple, list)):
            for entry in value:
                found = visit(entry)
                if found is not None:
                    return found
        # Event batches are inspected without iterating their array values.
        if hasattr(value, 'L_km') and hasattr(value, 'E_GeV'):
            for name in ('L_km', 'E_GeV', 'flavor_emit', 'flavor_det', 'isAntiNu'):
                found = visit(getattr(value, name))
                if found is not None:
                    return found
        return None

    return visit(values)


def evaluateDeviceArguments(oscillator, length, energy, emit, detect, reference, basis):
    """Preserve the positional API's energy/channel axes on the input device."""
    import torch
    from nu_waves.propagation.oscillator import NeutrinoEventBatch

    def realArray(value, name):
        if isinstance(value, (int, float, np.number)):
            return torch.as_tensor(value, device=reference.device, dtype=reference.dtype).reshape(-1)
        if not isinstance(value, torch.Tensor) or value.device != reference.device:
            raise ValueError(f'{name} is on host or another device; all array inputs must be on {reference.device}.')
        return value.reshape(-1)

    length, energy = realArray(length, 'L_km'), realArray(energy, 'E_GeV')
    length, energy = torch.broadcast_tensors(length, energy)

    def channels(value, name):
        if value is None:
            return torch.arange(oscillator.hamiltonian.n_neutrinos, device=reference.device)
        if isinstance(value, int) or (isinstance(value, list) and all(isinstance(v, int) for v in value)):
            return torch.as_tensor(value, device=reference.device, dtype=torch.int64).reshape(-1)
        return realArray(value, name).to(dtype=torch.int64)

    emit, detect = channels(emit, 'flavor_emit'), channels(detect, 'flavor_det')
    shape = (length.numel(), emit.numel(), detect.numel())
    expand = lambda value: value.expand(shape).reshape(-1)
    events = NeutrinoEventBatch(expand(length[:, None, None]), expand(energy[:, None, None]),
                                expand(emit[None, :, None]), expand(detect[None, None, :]))
    result = evaluateBuffers(oscillator, events, None, basis, None)
    squeezed = tuple(size for size in shape if size != 1)
    if basis is None:
        return result.reshape(squeezed)
    probabilities, jacobian = result
    return probabilities.reshape(squeezed), jacobian.reshape(squeezed + (jacobian.shape[-1],))


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
    if output is None:
        import torch
        reference = findDeviceInput(oscillator, events)
        if reference is None:
            raise ValueError('Automatic buffer allocation requires device inputs.')
        # Precision follows real event data, not integer flavor/category buffers.
        realInputs = [value for value in (events.E_GeV, events.L_km)
                      if isinstance(value, torch.Tensor) and value.device == reference.device]
        realReference = realInputs[0] if realInputs else reference
        count = next((value.shape[0] for value in (events.E_GeV, events.L_km,
                     events.flavor_emit, events.flavor_det, events.isAntiNu)
                      if hasattr(value, 'ndim') and value.ndim == 1), 1)
        output = torch.empty(count, device=reference.device, dtype=realReference.dtype)
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
        if allowScalar and isinstance(value, (int, float, bool, np.number, np.bool_)):
            return
        if numpyMode:
            if not isinstance(value, np.ndarray):
                raise TypeError(f"{name} must be a NumPy array, matching outputBuffer.")
        elif not isinstance(value, torch.Tensor) or value.device != device:
            location = str(value.device) if isinstance(value, torch.Tensor) else 'host/non-Torch storage'
            raise ValueError(f"{name} uses {location}, expected a Torch tensor on {device}. "
                             "All array inputs and outputs must share the same device; no implicit transfer is performed.")
        if shape is not None and value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}.")
        if real and value.dtype != output.dtype:
            raise TypeError(f"{name} must have the same dtype as outputBuffer.")

    checkStorage(output, "outputBuffer", real=True)
    if numpyMode and not output.flags.writeable:
        raise ValueError("outputBuffer must be writable.")
    for name in ("L_km", "E_GeV", "flavor_emit", "flavor_det"):
        checkStorage(getattr(events, name), name, output.shape, real=name in ("L_km", "E_GeV"), allowScalar=True)
    if events.isAntiNu is not None:
        checkStorage(events.isAntiNu, "isAntiNu", output.shape, allowScalar=True)
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
    if isinstance(hamiltonian, MatterHamiltonian):
        if hamiltonian._matter_profile is None:
            for value in hamiltonian.constantDensity:
                checkStorage(value, 'matter parameter', allowScalar=True)
        else:
            for layer in hamiltonian._matter_profile.layers:
                for value in (layer.rho_in_g_per_cm3, layer.Ye, layer.weight):
                    checkStorage(value, 'matter layer parameter', allowScalar=True)
    matter = isinstance(hamiltonian, MatterHamiltonian)
    if not matter and type(hamiltonian) is not VacuumHamiltonian:
        raise NotImplementedError("Buffer evaluation supports vacuum and matter Hamiltonians.")
    if matter and device is not None and device.type == "mps":
        raise NotImplementedError("Matter buffer evaluation on MPS requires a device-native eigensolver; CPU fallback is disabled.")

    with guard, Backend.useBuffer(output):
        xp = Backend.xp()
        # Python/NumPy scalar constants are runtime context, not host data arrays.
        def eventArray(value, dtype):
            if isinstance(value, (int, float, bool, np.number, np.bool_)):
                return xp.full(output.shape, value, dtype=dtype)
            return xp.asarray(value, dtype=dtype)
        lengthInput = eventArray(events.L_km, Backend.real_dtype())
        energyInput = eventArray(events.E_GeV, Backend.real_dtype())
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
        emit = eventArray(events.flavor_emit, xp.int64)
        detect = eventArray(events.flavor_det, xp.int64)
        anti = events.isAntiNu
        if anti is None:
            anti = xp.full(output.shape, hamiltonian._antineutrino, dtype=bool if numpyMode else xp.bool)
        else:
            anti = eventArray(anti, Backend.real_dtype()) > 0.5
        executor = hamiltonian.makeExecutor(oscillator)
        prepared = executor.prepareModel(model, basis) if matter else model
        for start in range(0, output.shape[0], 65536):
            sl = slice(start, start + 65536)
            length = lengthInput[sl] * KM_TO_EVINV
            energy = energyInput[sl] * GEV_TO_EV
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
