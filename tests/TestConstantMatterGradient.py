"""Independent finite-difference and physical checks of matter Jacobians."""
import unittest
import numpy as np

from nu_waves.hamiltonian import matter, vacuum
from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.propagation.oscillator import Oscillator, NeutrinoEventBatch
from nu_waves.propagation.gradient import MixingAngle, DiracPhase, MassSquaredDifference


class TestConstantMatterGradient(unittest.TestCase):
    def makeOscillator(self, values, n=3, rho=2.6, useVacuum=False):
        angles = {(1, 2): values[0], (1, 3): values[1], (2, 3): values[2]}
        phases = {(1, 3): values[3]}
        differences = {(2, 1): values[4], (3, 2): values[5]}
        if n == 4:
            angles[(1, 4)], phases[(1, 4)], differences[(4, 1)] = values[6:9]
        h = (vacuum if useVacuum else matter).Hamiltonian(
            Mixing(n, angles, phases), Spectrum(n, differences), False)
        if not useVacuum:
            h.set_constant_density(rho, .5)
        return Oscillator(h)

    def makeEvents(self, n=3, zeroLength=False):
        count = 2 * n * n
        return NeutrinoEventBatch(
            np.zeros(count) if zeroLength else np.linspace(295, 1300, count),
            np.linspace(.3, 3., count), np.tile(np.repeat(np.arange(n), n), 2),
            np.tile(np.arange(n), 2*n), np.repeat([False, True], n*n))

    def setUp(self):
        self.values = np.array([.59, .15, .81, -1.6, 7.5e-5, 2.5e-3, .2, .3, .01])
        self.basis = [MixingAngle(1, 2), MixingAngle(1, 3), MixingAngle(2, 3),
                      DiracPhase(1, 3), MassSquaredDifference(2, 1), MassSquaredDifference(3, 2)]

    def testFiniteDifferences(self):
        for n in [3, 4]:
            requested = self.basis + ([MixingAngle(1, 4), DiracPhase(1, 4), MassSquaredDifference(4, 1)] if n == 4 else [])
            for inverted in [False, True]:
                with self.subTest(n=n, inverted=inverted):
                    values = self.values.copy()
                    if inverted:
                        values[5] *= -1
                    osc = self.makeOscillator(values, n)
                    events = self.makeEvents(n)
                    compiled = osc.compileEvents(events)
                    probability, gradient = osc.probabilityWithGradient(compiled, requested)
                    for optimized in [False, True]:
                        osc.hamiltonian.enableConstantMatterBatchOptimization = optimized
                        np.testing.assert_allclose(probability, osc.probability(compiled), atol=3e-13)
                    for column, parameter in enumerate(requested):
                        for factor in [1., .25]:
                            step = factor * (1e-9 if isinstance(parameter, MassSquaredDifference) else 1e-6)
                            plus, minus = values.copy(), values.copy()
                            plus[column] += step
                            minus[column] -= step
                            numerical = (self.makeOscillator(plus, n).probability(compiled) -
                                         self.makeOscillator(minus, n).probability(compiled)) / (2*step)
                            np.testing.assert_allclose(gradient[:, column], numerical, rtol=3e-5, atol=3e-6)
                    _, subset = osc.probabilityWithGradient(events, [requested[5], requested[0], requested[5]])
                    np.testing.assert_allclose(subset, gradient[:, [5, 0, 5]], atol=1e-12)
                    pEmpty, gEmpty = osc.probabilityWithGradient(events, [])
                    np.testing.assert_allclose(pEmpty, probability, atol=1e-14)
                    self.assertEqual(gEmpty.shape, (2*n*n, 0))

    def testVacuumLimit(self):
        events = self.makeEvents()
        expected = self.makeOscillator(self.values, useVacuum=True).probabilityWithGradient(events, self.basis)
        actual = self.makeOscillator(self.values, rho=0).probabilityWithGradient(events, self.basis)
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, b, atol=3e-11, rtol=1e-11)

    def testConservationAndZeroLength(self):
        osc = self.makeOscillator(self.values)
        events = NeutrinoEventBatch(np.full(6, 295.), np.full(6, .6), np.ones(6, dtype=int),
                                   np.tile(np.arange(3), 2), np.repeat([False, True], 3))
        p, g = osc.probabilityWithGradient(events, self.basis)
        np.testing.assert_allclose(p.reshape(2, 3).sum(axis=1), 1, atol=1e-14)
        np.testing.assert_allclose(g.reshape(2, 3, 6).sum(axis=1), 0, atol=2e-12)
        zero = self.makeEvents(zeroLength=True)
        p, g = osc.probabilityWithGradient(zero, self.basis)
        np.testing.assert_allclose(p, (np.asarray(zero.flavor_emit) == zero.flavor_det), atol=1e-14)
        np.testing.assert_allclose(g, 0, atol=1e-14)

    def testDegenerateEigenvalues(self):
        # States 1 and 3 coincide in vacuum. With zero mixing, probabilities
        # are stationary even though Hamiltonian tangents need not be zero.
        values = self.values.copy()
        values[:3] = 0
        values[5] = -values[4]
        for rho in [0, 2.6]:
            osc = self.makeOscillator(values, rho=rho)
            p, g = osc.probabilityWithGradient(self.makeEvents(), self.basis)
            self.assertTrue(np.isfinite(p).all())
            np.testing.assert_allclose(g, 0, atol=1e-12)

    def testRejectLayeredMatter(self):
        osc = self.makeOscillator(self.values)
        osc.hamiltonian.set_matter_profile(matter.MatterProfile.from_fractions([2.6, 3.], [.5, .5], [.5, .5]))
        with self.assertRaises(NotImplementedError):
            osc.probabilityWithGradient(self.makeEvents(), self.basis)
