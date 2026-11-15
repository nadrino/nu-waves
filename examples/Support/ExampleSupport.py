"""Shared paths and explicit NumPy adiabatic approximations for the examples."""
from itertools import permutations
import os
from pathlib import Path
import sys
import tempfile
import urllib.request
import warnings

import numpy as np

exampleDirectory = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(exampleDirectory.parent / 'src'))
dataDirectory = exampleDirectory / 'data'
figureDirectory = exampleDirectory / 'figures'
figureDirectory.mkdir(exist_ok=True)
quick = '--quick' in sys.argv or os.environ.get('NU_WAVES_EXAMPLE_QUICK') == '1'


def sampleCount(full, reduced=32):
    return min(full, reduced) if quick else full


def probabilityOnDevice(oscillator, lengthKm, energyGeV, device):
    """Full channel matrix via explicit resident event and output buffers."""
    import torch
    from nu_waves.propagation.oscillator import NeutrinoEventBatch
    n = oscillator.hamiltonian.n_neutrinos
    dtype = torch.float32 if device == 'mps' else torch.float64
    energy = torch.as_tensor(energyGeV, device=device, dtype=dtype).reshape(-1)
    count = len(energy)
    energy = energy[:, None, None].expand(count, n, n).reshape(-1)
    labels = torch.arange(n, device=device)
    emit = labels[None, :, None].expand(count, n, n).reshape(-1)
    detect = labels[None, None, :].expand(count, n, n).reshape(-1)
    events = NeutrinoEventBatch(torch.full_like(energy, lengthKm), energy, emit, detect)
    weights = torch.empty_like(energy)
    oscillator.probability(events, outputBuffer=weights)
    return weights.reshape(count, n, n).cpu().numpy()


def loadSolarProfile():
    from nu_waves.matter.solar import load_bs05_agsop, SolarProfile
    path = Path(os.environ.get('NU_WAVES_SOLAR_TABLE', dataDirectory / 'ssm' / 'bs05_agsop.dat')).expanduser()
    try:
        return load_bs05_agsop(str(path))
    except (OSError, ValueError):
        if quick and 'NU_WAVES_SOLAR_TABLE' not in os.environ:
            warnings.warn('Quick smoke test uses the analytic toy SolarProfile; no BS05 predictions are produced.')
            return SolarProfile()
        if 'NU_WAVES_SOLAR_TABLE' in os.environ:
            raise
    path.parent.mkdir(parents=True, exist_ok=True)
    # Validate downloads before replacing the cache, including HTTP error pages.
    try:
        with tempfile.TemporaryDirectory(dir=path.parent) as temporary:
            download = Path(temporary) / path.name
            urllib.request.urlretrieve('https://www.sns.ias.edu/~jnb/SNdata/Export/BS2005/bs05_agsop.dat', download)
            result = load_bs05_agsop(str(download))
            download.replace(path)
            return result
    except (OSError, ValueError) as error:
        raise RuntimeError('Cannot load BS05 data. Set NU_WAVES_SOLAR_TABLE to a valid bs05_agsop.dat file, '
                           'or use --quick for an analytic toy-profile smoke test.') from error


def solarHamiltonians(oscillator, energyGeV, profile, radiiKm, antineutrino=False):
    from nu_waves.utils.units import VCOEFF_EV, GEV_TO_EV
    model = oscillator.hamiltonian
    mixing = model.mixing.build_mixing_matrix()
    if antineutrino:
        mixing = mixing.conj()
    masses = model.spectrum.relativeMasses()
    vacuum = (mixing * masses[None, :]) @ mixing.conj().T
    radiiKm = np.atleast_1d(radiiKm)
    matrices = np.broadcast_to(vacuum / (2 * energyGeV * GEV_TO_EV),
                               (len(radiiKm),) + vacuum.shape).copy()
    matrices[:, 0, 0] += ((-1 if antineutrino else 1) * VCOEFF_EV *
                          profile.rho_gcm3(radiiKm) * profile.Ye(radiiKm))
    return matrices


def adiabaticMassFractions(oscillator, E_GeV, profile, r_emit_km, s_km, alpha=0, antineutrino=False):
    """Phase-averaged vacuum-basis fractions under adiabatic evolution.

    Follow instantaneous matter eigenstates by maximum adjacent overlap.
    No non-adiabatic jumps or coherent relative phases are modeled here.
    """
    radii = r_emit_km + np.asarray(s_km)
    _, vectors = np.linalg.eigh(solarHamiltonians(oscillator, E_GeV, profile, radii, antineutrino))
    n = vectors.shape[-1]
    orders = list(permutations(range(n)))
    for index in range(1, len(vectors)):
        overlaps = np.abs(vectors[index - 1].conj().T @ vectors[index]) ** 2
        order = max(orders, key=lambda order: sum(overlaps[i, order[i]] for i in range(n)))
        vectors[index] = vectors[index][:, order]
    weights = np.abs(vectors[0, alpha, :]) ** 2
    mixing = oscillator.hamiltonian.mixing.build_mixing_matrix()
    if antineutrino:
        mixing = mixing.conj()
    projection = mixing.conj().T[None, :, :] @ vectors
    fractions = (np.abs(projection) ** 2) @ weights
    np.testing.assert_allclose(fractions.sum(axis=-1), 1., atol=1e-12)
    return fractions


def initialMassComposition(oscillator, alpha=0, basis='vacuum', E_GeV=None, profile=None, r_emit_km=0.):
    if basis == 'vacuum':
        return np.abs(oscillator.hamiltonian.mixing.build_mixing_matrix()[alpha]) ** 2
    if basis == 'matter':
        _, vectors = np.linalg.eigh(solarHamiltonians(oscillator, E_GeV, profile, [r_emit_km]))
        return np.abs(vectors[0, alpha]) ** 2
    if basis == 'vacuum_from_matter':
        return adiabaticMassFractions(oscillator, E_GeV, profile, r_emit_km, [0.], alpha)[0]
    raise ValueError(f'Unknown basis: {basis}')


def exitMassFractionsAveraged(oscillator, profile, energies, radii, weights, n_steps_per_ray=512):
    return np.array([sum(weight * adiabaticMassFractions(
        oscillator, energy, profile, radius,
        np.linspace(0., profile.R_sun_km - radius, n_steps_per_ray))[-1]
        for radius, weight in zip(radii, weights)) for energy in energies])
