"""Analytic vacuum and constant-matter probability Jacobians.

This initial implementation uses NumPy float64 on the CPU independently of
the configured backend. The ordinary probability path is unaffected.
"""

from dataclasses import dataclass
import numpy as np

from nu_waves.utils.units import GEV_TO_EV, KM_TO_EVINV, VCOEFF_EV


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


def buildMixingGradient(mixing, parameterBasis):
    """Differentiate the same ordered rotation product as Mixing."""
    n = mixing.n_neutrinos
    pairs = list(mixing.mixing_angles)
    ordered = [p for p in [(2, 3), (1, 3), (1, 2)] if n >= 3 and p in pairs]
    ordered += [p for p in pairs if p not in ordered]
    matrix = np.eye(n, dtype=complex)
    derivatives = np.zeros((len(parameterBasis), n, n), dtype=complex)
    for pair in ordered:
        i, j = pair[0] - 1, pair[1] - 1
        theta = mixing.mixing_angles[pair]
        delta = mixing.dirac_phases.get(pair, 0.0)
        s, c = np.sin(theta), np.cos(theta)
        phase = np.exp(-1j * delta)
        rotation = np.eye(n, dtype=complex)
        rotation[i, i] = rotation[j, j] = c
        rotation[i, j], rotation[j, i] = s * phase, -s * phase.conjugate()
        derivatives = derivatives @ rotation
        for column, parameter in enumerate(parameterBasis):
            if (parameter.i, parameter.j) != pair:
                continue
            tangent = np.zeros((n, n), dtype=complex)
            if isinstance(parameter, MixingAngle):
                tangent[i, i] = tangent[j, j] = -s
                tangent[i, j], tangent[j, i] = c * phase, -c * phase.conjugate()
            elif isinstance(parameter, DiracPhase):
                tangent[i, j] = -1j * s * phase
                tangent[j, i] = -1j * s * phase.conjugate()
            derivatives[column] += matrix @ tangent
        matrix = matrix @ rotation
    return matrix, derivatives


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
    return masses, derivatives


def validateParameterBasis(hamiltonian, parameterBasis):
    """Validate ordered physical parameter references against the model."""
    mixing, spectrum = hamiltonian.mixing, hamiltonian.spectrum
    for p in parameterBasis:
        if type(p) not in (MixingAngle, DiracPhase, MassSquaredDifference):
            raise TypeError(f"Unsupported gradient parameter: {p!r}")
        pair = (p.i, p.j)
        if isinstance(p, MassSquaredDifference):
            valid = (spectrum._dm2_dict or {}).get(pair) is not None
        else:
            valid = pair in mixing.mixing_angles
        if not valid:
            raise ValueError(f"Parameter is not defined in this model: {p!r}")


def vacuumProbabilityWithGradient(hamiltonian, compiledEvents, parameterBasis):
    """Return (probabilities, Jacobian), with shapes (events,), (events, params).

    Angles and phases are in radians; mass differences are in eV squared.
    Only declared independent mass differences may be requested. An empty
    basis is valid. Unknown parameter references raise ValueError.
    """
    parameterBasis = tuple(parameterBasis)
    validateParameterBasis(hamiltonian, parameterBasis)
    mixing, spectrum = hamiltonian.mixing, hamiltonian.spectrum
    matrix, dMatrix = buildMixingGradient(mixing, parameterBasis)
    masses, dMasses = buildSpectrumGradient(spectrum, parameterBasis)
    probabilities = np.empty(compiledEvents.n_events)
    gradients = np.empty((compiledEvents.n_events, len(parameterBasis)))
    for group in compiledEvents.groups:
        u = matrix.conjugate() if group.isAntiNu else matrix
        du = dMatrix.conjugate() if group.isAntiNu else dMatrix
        a, b = group.flavor_emit, group.flavor_det
        coefficients = u[a].conjugate() * u[b]
        dCoefficients = du[:, a].conjugate() * u[b] + u[a].conjugate() * du[:, b]
        # Bound intermediates independently of the total event count.
        for start in range(0, len(group.indices), 65536):
            sl = slice(start, start + 65536)
            indices = group.indices[sl]
            scale = 0.5 * (group.L_km[sl] * KM_TO_EVINV) / (group.E_GeV[sl] * GEV_TO_EV)
            phases = np.exp(-1j * scale[:, None] * masses)
            amplitude = phases @ coefficients
            probabilities[indices] = np.abs(amplitude) ** 2
            for column in range(len(parameterBasis)):
                dAmplitude = phases @ dCoefficients[column]
                dAmplitude += -1j * scale * (phases @ (coefficients * dMasses[column]))
                gradients[indices, column] = 2 * np.real(amplitude.conjugate() * dAmplitude)
    return probabilities, gradients


def constantMatterProbabilityWithGradient(hamiltonian, compiledEvents, parameterBasis):
    """Differentiate exp(-i L H) at fixed density, electron fraction, L and E.

    The spectral Frechet derivative reuses one eigendecomposition for every
    requested parameter. Its sinc form remains finite at repeated eigenvalues:
    F_ij = -i L exp(-i L (lambda_i + lambda_j)/2)
                  sinc(L (lambda_i - lambda_j)/2), with sinc(x) = sin(x)/x.
    No eigenvector derivatives or finite differences are used. All intermediate
    matrices are NumPy complex128 on CPU, independently of the backend.
    """
    if hamiltonian._matter_profile is not None:
        raise NotImplementedError("Analytic matter gradients require constant density.")
    parameterBasis = tuple(parameterBasis)
    validateParameterBasis(hamiltonian, parameterBasis)
    u, du = buildMixingGradient(hamiltonian.mixing, parameterBasis)
    masses, dMasses = buildSpectrumGradient(hamiltonian.spectrum, parameterBasis)
    ud = u.conjugate().T
    vacuumMatrix = (u * masses) @ ud
    dVacuum = ((du * masses) @ ud + (u * dMasses[:, None, :]) @ ud
               + (u * masses) @ du.conjugate().transpose(0, 2, 1))
    rho, electronFraction = hamiltonian.constantDensity
    potential = VCOEFF_EV * rho * electronFraction
    probabilities = np.empty(compiledEvents.n_events)
    gradients = np.empty((compiledEvents.n_events, len(parameterBasis)))
    for group in compiledEvents.groups:
        hVac = vacuumMatrix.conjugate() if group.isAntiNu else vacuumMatrix
        dhVac = dVacuum.conjugate() if group.isAntiNu else dVacuum
        sign = -1 if group.isAntiNu else 1
        for start in range(0, len(group.indices), 65536):
            sl = slice(start, start + 65536)
            indices = group.indices[sl]
            length = group.L_km[sl] * KM_TO_EVINV
            inv2Energy = 0.5 / (group.E_GeV[sl] * GEV_TO_EV)
            h = inv2Energy[:, None, None] * hVac
            h[:, 0, 0] += sign * potential
            eigenvalues, q = np.linalg.eigh(h)
            qd = q.conjugate().transpose(0, 2, 1)
            phases = np.exp(-1j * length[:, None] * eigenvalues)
            a, b = group.flavor_emit, group.flavor_det
            amplitude = np.sum(q[:, b, :] * phases * q[:, a, :].conjugate(), axis=1)
            probabilities[indices] = np.abs(amplitude) ** 2
            if not parameterBasis:
                continue
            phaseValues = length[:, None] * eigenvalues
            midpoints = 0.5 * (phaseValues[:, :, None] + phaseValues[:, None, :])
            halfGaps = 0.5 * (phaseValues[:, :, None] - phaseValues[:, None, :])
            # NumPy's sinc is normalized: sinc(x) = sin(pi*x)/(pi*x).
            kernel = -1j * length[:, None, None] * np.exp(-1j * midpoints) * np.sinc(halfGaps / np.pi)
            for column in range(len(parameterBasis)):
                tangent = inv2Energy[:, None, None] * dhVac[column]
                spectralTangent = qd @ tangent @ q
                dAmplitude = np.einsum('bi,bij,bj->b', q[:, b, :],
                                       kernel * spectralTangent, q[:, a, :].conjugate())
                gradients[indices, column] = 2 * np.real(amplitude.conjugate() * dAmplitude)
    return probabilities, gradients
