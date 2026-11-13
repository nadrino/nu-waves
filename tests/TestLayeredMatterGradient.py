"""Layer products, independent differences and backend checks."""
import importlib.util
import unittest
from unittest.mock import patch
import numpy as np

from nu_waves.globals.backend import Backend
from nu_waves.hamiltonian import matter
from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.propagation.oscillator import Oscillator, NeutrinoEventBatch
from nu_waves.propagation.gradient import MixingAngle, DiracPhase, MassSquaredDifference, MatterDensity, ElectronFraction


class TestLayeredMatterGradient(unittest.TestCase):
    basis = [MixingAngle(2, 3), DiracPhase(1, 3), MassSquaredDifference(3, 2),
             MatterDensity(0), ElectronFraction(1), MatterDensity(1)]
    values = np.array([.81, -1.6, 2.5e-3, 2.6, .47, 4.1])

    def tearDown(self):
        Backend.set_api(np)

    def makeOscillator(self, values=None, absolute=False, constant=False, n=3):
        v = self.values if values is None else values
        angles = {(1, 2): .59, (1, 3): .15, (2, 3): v[0]}
        masses = {(2, 1): 7.5e-5, (3, 2): v[2]}
        if n == 4:
            angles[(1, 4)], masses[(4, 1)] = .2, .01
        h = matter.Hamiltonian(Mixing(n, angles, {(1, 3): v[1]}), Spectrum(n, masses), False)
        if constant:
            h.set_constant_density(v[3], .5)
        else:
            profile = (matter.MatterProfile.from_segments([v[3], v[5]], [.5, v[4]], [118., 177.])
                       if absolute else matter.MatterProfile.from_fractions([v[3], v[5]], [.5, v[4]], [.4, .6]))
            h.set_matter_profile(profile)
        return Oscillator(h)

    def makeEvents(self, n=3):
        return NeutrinoEventBatch(np.full(2*n, 295.), np.full(2*n, .6), np.ones(2*n, dtype=int),
                                 np.tile(np.arange(n), 2), np.repeat([False, True], n))

    def testFiniteDifferences(self):
        for absolute in [False, True]:
            for n in [3, 4]:
                for inverted in [False, True]:
                    values = self.values.copy()
                    if inverted:
                        values[2] *= -1
                    osc = self.makeOscillator(values, absolute, n=n)
                    events = self.makeEvents(n)
                    compiled = osc.compileEvents(events)
                    p, g = osc.probability(compiled, parameterBasis=self.basis)
                    osc.useExecutor = False
                    np.testing.assert_allclose(p, osc.probability(compiled), atol=2e-13)
                    np.testing.assert_allclose(p.reshape(2, n).sum(axis=1), 1, atol=2e-14)
                    np.testing.assert_allclose(g.reshape(2, n, -1).sum(axis=1), 0, atol=1e-11)
                    for column in range(len(self.basis)):
                        step = 1e-7 if column == 2 else 1e-3
                        def shifted(offset):
                            v = values.copy()
                            v[column] += offset * step
                            return self.makeOscillator(v, absolute, n=n).probability(compiled)
                        numerical = (shifted(-2) - 8*shifted(-1) + 8*shifted(1) - shifted(2)) / (12*step)
                        np.testing.assert_allclose(g[:, column], numerical, rtol=3e-5, atol=3e-6)
                    _, subset = osc.probability(events, parameterBasis=[self.basis[4], self.basis[0], self.basis[4]])
                    np.testing.assert_allclose(subset, g[:, [4, 0, 4]], atol=1e-12)
                    self.assertEqual(osc.probability(events, parameterBasis=[])[1].shape, (2*n, 0))

    def testConstantLimit(self):
        values = self.values.copy()
        values[4], values[5] = .5, values[3]
        p, g = self.makeOscillator(values).probability(self.makeEvents(), parameterBasis=self.basis)
        expectedP, expectedG = self.makeOscillator(values, constant=True).probability(
            self.makeEvents(), parameterBasis=self.basis[:3] + [MatterDensity()])
        np.testing.assert_allclose(p, expectedP, atol=1e-14)
        np.testing.assert_allclose(g[:, :3], expectedG[:, :3], atol=1e-12)
        np.testing.assert_allclose(g[:, 3] + g[:, 5], expectedG[:, 3], atol=1e-14)

    def testOrderZeroLengthsAndValidation(self):
        osc = self.makeOscillator()
        events = self.makeEvents()
        forward = osc.probability(events)
        osc.hamiltonian._matter_profile.layers.reverse()
        self.assertGreater(np.max(np.abs(forward - osc.probability(events))), 1e-6)
        for layer in osc.hamiltonian._matter_profile.layers:
            layer.weight = 0.
        p, g = osc.probability(events, parameterBasis=self.basis)
        np.testing.assert_allclose(p, np.tile([0., 1., 0.], 2), atol=1e-14)
        np.testing.assert_allclose(g, 0, atol=1e-13)
        for reference in [MatterDensity(), MatterDensity(-1), ElectronFraction(2), MatterDensity(True)]:
            with self.assertRaises(ValueError):
                osc.probability(events, parameterBasis=[reference])
        with self.assertRaises(ValueError):
            self.makeOscillator(constant=True).probability(events, parameterBasis=[MatterDensity(0)])

    def testSharedLayerDecomposition(self):
        osc = self.makeOscillator()
        events = NeutrinoEventBatch([295., 295.], [.5, .6], [1, 1], [0, 0])
        with patch.object(osc.hamiltonian, '_eigh', wraps=osc.hamiltonian._eigh) as eigh:
            osc.probability(events, parameterBasis=self.basis)
            self.assertEqual(eigh.call_count, 2)
        with patch.object(osc.hamiltonian, 'potentialTangents', side_effect=AssertionError('No derivatives requested')):
            osc.probability(events)

    def testAbsoluteSegmentsIgnoreTotalBaseline(self):
        events = self.makeEvents()
        changed = NeutrinoEventBatch(np.linspace(100., 900., 6), events.E_GeV,
                                    events.flavor_emit, events.flavor_det, events.isAntiNu)
        osc = self.makeOscillator(absolute=True)
        expected = osc.probability(events, parameterBasis=self.basis)
        actual = osc.probability(changed, parameterBasis=self.basis)
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, b, atol=1e-13)
        fractional = self.makeOscillator()
        self.assertGreater(np.max(np.abs(fractional.probability(events) - fractional.probability(changed))), .01)

    def checkTorch(self, device):
        import torch
        events = self.makeEvents()
        for absolute in [False, True]:
            Backend.set_api(np)
            reference = self.makeOscillator(absolute=absolute).probability(events, parameterBasis=self.basis)
            Backend.set_api(torch, device=device)
            osc = self.makeOscillator(absolute=absolute)
            p, g = osc.probability(events, parameterBasis=self.basis)
            tolerance = dict(rtol=2e-3, atol=3e-3) if device == 'mps' else dict(rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(p, reference[0], **tolerance)
            np.testing.assert_allclose(g, reference[1], **tolerance)
            np.testing.assert_allclose(p, osc.probability(events), **tolerance)
            np.testing.assert_allclose(g[:, 3:], reference[1][:, 3:], rtol=2e-3, atol=3e-6)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Torch is not installed')
    def testTorchCpu(self):
        self.checkTorch('cpu')

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Torch is not installed')
    def testTorchMps(self):
        import torch
        if not torch.backends.mps.is_available():
            self.skipTest('MPS is not available')
        self.checkTorch('mps')

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Torch is not installed')
    def testTorchCuda(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('CUDA is not available')
        self.checkTorch('cuda')
