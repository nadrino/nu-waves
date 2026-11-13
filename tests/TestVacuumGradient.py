import numpy as np
import unittest

from nu_waves.hamiltonian import vacuum, matter
from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.propagation.oscillator import Oscillator, NeutrinoEventBatch
from nu_waves.propagation.gradient import MixingAngle, DiracPhase, MassSquaredDifference


basis = [MixingAngle(1, 2), MixingAngle(1, 3), MixingAngle(2, 3),
         DiracPhase(1, 3), MassSquaredDifference(2, 1), MassSquaredDifference(3, 2)]
values = np.array([0.59, 0.15, 0.81, -1.6, 7.5e-5, 2.5e-3])


def makeOscillator(parameters, n=3):
    angles = {(1, 2): parameters[0], (1, 3): parameters[1], (2, 3): parameters[2]}
    phases = {(1, 3): parameters[3]}
    differences = {(2, 1): parameters[4], (3, 2): parameters[5]}
    if n == 4:
        angles[(1, 4)] = parameters[6]
        phases[(1, 4)] = parameters[7]
        differences[(4, 1)] = parameters[8]
    return Oscillator(vacuum.Hamiltonian(
        Mixing(n, angles, phases), Spectrum(n, differences), False))


def makeEvents(n=3):
    count = 2 * n * n
    return NeutrinoEventBatch(
        L_km=np.linspace(0, 1300, count), E_GeV=np.linspace(0.2, 3, count),
        flavor_emit=np.tile(np.repeat(np.arange(n), n), 2),
        flavor_det=np.tile(np.arange(n), 2 * n),
        isAntiNu=np.repeat([False, True], n * n))


def testFiniteDifferences(inverted, n):
    parameters = values.copy()
    if inverted:
        parameters[-1] *= -1
    if n == 4:
        parameters = np.concatenate([parameters, [0.2, 0.3, 0.1]])
    oscillator = makeOscillator(parameters, n)
    events = makeEvents(n)
    compiled = oscillator.compileEvents(events)
    requested = basis + ([MixingAngle(1, 4), DiracPhase(1, 4), MassSquaredDifference(4, 1)] if n == 4 else [])
    probabilities, gradients = oscillator.probabilityWithGradient(compiled, requested)
    np.testing.assert_allclose(probabilities, oscillator.probability(compiled), atol=2e-13)
    for column in range(len(requested)):
        for factor in [1, 0.25]:
            step = (1e-9 if isinstance(requested[column], MassSquaredDifference) else 1e-6) * factor
            plus, minus = parameters.copy(), parameters.copy()
            plus[column] += step
            minus[column] -= step
            numerical = (makeOscillator(plus, n).probability(compiled) -
                         makeOscillator(minus, n).probability(compiled)) / (2 * step)
            np.testing.assert_allclose(gradients[:, column], numerical, rtol=2e-5, atol=2e-6)
    _, subset = oscillator.probabilityWithGradient(events, [basis[5], basis[0], basis[5]])
    np.testing.assert_allclose(subset, gradients[:, [5, 0, 5]])
    assert oscillator.probabilityWithGradient(events, [])[1].shape == (2 * n * n, 0)


def testConservationAndZeroBaseline():
    oscillator = makeOscillator(values)
    events = NeutrinoEventBatch(np.full(6, 295.), np.full(6, .6),
                               np.ones(6, dtype=int), np.tile(np.arange(3), 2),
                               np.repeat([False, True], 3))
    probabilities, gradients = oscillator.probabilityWithGradient(events, basis)
    np.testing.assert_allclose(probabilities.reshape(2, 3).sum(axis=1), 1, atol=1e-14)
    np.testing.assert_allclose(gradients.reshape(2, 3, 6).sum(axis=1), 0, atol=2e-12)
    zero = NeutrinoEventBatch(np.zeros(6), events.E_GeV, events.flavor_emit,
                             events.flavor_det, events.isAntiNu)
    np.testing.assert_allclose(oscillator.probabilityWithGradient(zero, basis)[1], 0, atol=1e-14)


def testValidation():
    oscillator = makeOscillator(values)
    for parameter in [MixingAngle(1, 4), MassSquaredDifference(3, 1)]:
        with unittest.TestCase().assertRaises(ValueError):
            oscillator.probabilityWithGradient(makeEvents(), [parameter])
    with unittest.TestCase().assertRaises(TypeError):
        oscillator.probabilityWithGradient(makeEvents(), ['theta_12'])
    oscillator.hamiltonian = matter.Hamiltonian(oscillator.hamiltonian.mixing,
                                              oscillator.hamiltonian.spectrum, False)
    oscillator.hamiltonian.set_matter_profile(matter.MatterProfile.from_fractions([2.6], [.5], [1.]))
    with unittest.TestCase().assertRaises(NotImplementedError):
        oscillator.probabilityWithGradient(makeEvents(), basis)


class TestVacuumGradient(unittest.TestCase):
    def testFiniteDifferences(self):
        for inverted in [False, True]:
            for n in [3, 4]:
                with self.subTest(inverted=inverted, n=n):
                    testFiniteDifferences(inverted, n)

    def testConservation(self):
        testConservationAndZeroBaseline()

    def testInvalidParameters(self):
        testValidation()
