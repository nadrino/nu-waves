"""Parameter references and analytic model tangents; no event propagation."""

from dataclasses import dataclass
import numpy as np

from nu_waves.globals.backend import Backend


@dataclass(frozen=True)
class MixingAngle:
    i: int
    j: int


@dataclass(frozen=True)
class DiracPhase:
    i: int
    j: int


@dataclass(frozen=True)
class MassSquaredDifference:
    i: int
    j: int


@dataclass(frozen=True)
class MatterDensity:
    """Constant density in g/cm³, holding electron fraction fixed."""


@dataclass(frozen=True)
class ElectronFraction:
    """Dimensionless electron fraction, holding density fixed."""


def buildMixingGradient(mixing, parameterBasis):
    """Compatibility wrapper around the model's common rotation product."""
    return mixing.build_mixing_matrix(parameterBasis=parameterBasis)


def buildSpectrumGradient(spectrum, parameterBasis):
    """Resolve independent mass differences relative to state 1.

    Connectivity follows declared edges, including exactly zero differences.
    A common mass offset is omitted because it cancels in probabilities.
    """
    n = spectrum.n_neutrinos
    adjacency = [[] for _ in range(n)]
    for pair, value in (spectrum._dm2_dict or {}).items():
        if value is None:
            continue
        i, j = pair[0] - 1, pair[1] - 1
        tangent = np.array([float(isinstance(p, MassSquaredDifference) and
                                  (p.i, p.j) == pair) for p in parameterBasis])
        adjacency[j].append((i, value, tangent))
        adjacency[i].append((j, -value, -tangent))
    masses = np.zeros(n)
    derivatives = np.zeros((len(parameterBasis), n))
    visited = {0}
    queue = [0]
    for j in queue:
        for i, value, tangent in adjacency[j]:
            if i in visited:
                continue
            masses[i] = masses[j] + value
            derivatives[:, i] = derivatives[:, j] + tangent
            visited.add(i)
            queue.append(i)
    if len(visited) != n:
        raise ValueError("Mass differences must connect all mass states.")
    xp = Backend.xp()
    return xp.asarray(masses, dtype=Backend.real_dtype()), xp.asarray(derivatives, dtype=Backend.real_dtype())


def validateParameterBasis(hamiltonian, parameterBasis):
    """Validate ordered physical parameter references against the model."""
    mixing, spectrum = hamiltonian.mixing, hamiltonian.spectrum
    for p in parameterBasis:
        if type(p) in (MatterDensity, ElectronFraction):
            from nu_waves.hamiltonian.matter import Hamiltonian
            if not isinstance(hamiltonian, Hamiltonian) or hamiltonian._matter_profile is not None:
                raise ValueError("Matter parameters require a constant-matter Hamiltonian.")
            continue
        if type(p) not in (MixingAngle, DiracPhase, MassSquaredDifference):
            raise TypeError(f"Unsupported gradient parameter: {p!r}")
        pair = (p.i, p.j)
        if isinstance(p, MassSquaredDifference):
            valid = (spectrum._dm2_dict or {}).get(pair) is not None
        else:
            valid = pair in mixing.mixing_angles
        if not valid:
            raise ValueError(f"Parameter is not defined in this model: {p!r}")


def prepareModel(hamiltonian, parameterBasis=None):
    """Build model values once, allocating tangents only when requested."""
    xp = Backend.xp()
    masses = xp.asarray(hamiltonian.spectrum.get_m2(), dtype=Backend.real_dtype())
    if parameterBasis is None:
        return hamiltonian.mixing.build_mixing_matrix(), masses, None, None
    parameterBasis = tuple(parameterBasis)
    validateParameterBasis(hamiltonian, parameterBasis)
    matrix, dMatrix = hamiltonian.mixing.build_mixing_matrix(parameterBasis=parameterBasis)
    _, dMasses = buildSpectrumGradient(hamiltonian.spectrum, parameterBasis)
    return matrix, masses, dMatrix, dMasses


def probabilityGradient(amplitude, tangent):
    """Physical derivative of |amplitude| squared on the active backend."""
    xp = Backend.xp()
    return 2 * xp.real(xp.conjugate(amplitude) * tangent)
