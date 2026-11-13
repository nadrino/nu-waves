"""Shared propagation kernels on the active backend, with optional tangents."""
from nu_waves.globals.backend import Backend
from nu_waves.utils.units import VCOEFF_EV


def vacuumPhases(length, energy, masses):
    return Backend.xp().exp(-0.5j * (length / energy)[:, None] * masses[None, :])


def vacuumAmplitude(length, energy, matrix, masses, emit, detect, dMatrix=None, dMasses=None):
    xp = Backend.xp()
    phases = vacuumPhases(length, energy, masses)
    coefficients = xp.conjugate(matrix[emit]) * matrix[detect]
    amplitude = xp.sum(phases * coefficients[None, :], axis=1)
    if dMatrix is None:
        return amplitude, None
    tangents = []
    for column in range(dMatrix.shape[0]):
        dc = xp.conjugate(dMatrix[column, emit]) * matrix[detect] + xp.conjugate(matrix[emit]) * dMatrix[column, detect]
        tangent = xp.sum(phases * dc[None, :], axis=1)
        tangent = tangent - 0.5j * (length / energy) * xp.sum(phases * (coefficients * dMasses[column])[None, :], axis=1)
        tangents.append(tangent)
    return amplitude, tangents


def vacuumMatrix(matrix, masses, dMatrix=None, dMasses=None):
    xp = Backend.xp()
    adjoint = xp.matrix_transpose(xp.conjugate(matrix))
    value = (matrix * masses) @ adjoint
    if dMatrix is None:
        return value, None
    tangent = ((dMatrix * masses) @ adjoint + (matrix * dMasses[:, None, :]) @ adjoint
               + (matrix * masses) @ xp.matrix_transpose(xp.conjugate(dMatrix)))
    return value, tangent


def matterEvolution(hamiltonian, length, inv2Energy, vacuum, rho, electronFraction,
                    antineutrino=False, vacuumTangent=None):
    """Return propagator and optional tangents, sharing the eigendecomposition.

    The Frechet kernel uses normalized sinc and stays finite at degeneracy.
    Backend.linalg.eigh owns device-specific fallbacks (including MPS).
    """
    xp = Backend.xp()
    sign = -1 if antineutrino else 1
    if antineutrino:
        vacuum = xp.conjugate(vacuum)
    projector = xp.zeros(vacuum.shape, dtype=Backend.complex_dtype())
    projector[0, 0] = sign * VCOEFF_EV * rho * electronFraction
    h = vacuum[None, ...] * inv2Energy[:, None, None] + projector[None, ...]
    values, vectors = hamiltonian._eigh(h)
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
        spectral = adjoint @ (vacuumTangent[column] * inv2Energy[:, None, None]) @ vectors
        tangents.append(vectors @ (kernel * spectral) @ adjoint)
    return propagator, tangents
