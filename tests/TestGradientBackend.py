"""Shared API, backend parity and probability-only execution checks."""
import importlib.util
import unittest
from unittest.mock import patch
import numpy as np

from nu_waves.globals.backend import Backend
from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.hamiltonian import matter, vacuum
from nu_waves.propagation.oscillator import Oscillator, NeutrinoEventBatch
from nu_waves.propagation.gradient import MixingAngle, DiracPhase, MassSquaredDifference, MatterDensity, ElectronFraction


class TestGradientBackend(unittest.TestCase):
    basis = [MixingAngle(2, 3), DiracPhase(1, 3), MassSquaredDifference(3, 2)]

    def tearDown(self):
        Backend.set_api(np)

    def makeOscillator(self, useMatter=False):
        h = (matter if useMatter else vacuum).Hamiltonian(
            Mixing(3, {(1, 2): .59, (1, 3): .15, (2, 3): .81}, {(1, 3): -1.6}),
            Spectrum(3, {(2, 1): 7.5e-5, (3, 2): 2.5e-3}), False)
        if useMatter:
            h.set_constant_density(2.6, .5)
        return Oscillator(h)

    def testShapesAndCompatibility(self):
        for useMatter in [False, True]:
            osc = self.makeOscillator(useMatter)
            for emit, detect in [(None, None), (1, None), (1, 0), ([0, 1], [1, 2])]:
                for energy in [.6, [.4, .6]]:
                    kwargs = dict(L_km=295., E_GeV=energy, flavor_emit=emit, flavor_det=detect)
                    expected = osc.probability(**kwargs)
                    for basis in [[], self.basis]:
                        p, g = osc.probability(**kwargs, parameterBasis=basis)
                        np.testing.assert_allclose(p, expected, atol=1e-13)
                        self.assertEqual(g.shape, p.shape + (len(basis),))
                        pOld, gOld = osc.probabilityWithGradient(**kwargs, parameterBasis=basis)
                        np.testing.assert_allclose(pOld, p)
                        np.testing.assert_allclose(gOld, g)

    def testNoGradientWorkForProbability(self):
        for useMatter in [False, True]:
            osc = self.makeOscillator(useMatter)
            events = NeutrinoEventBatch([295.], [.6], [1], [0])
            with patch('nu_waves.propagation.gradient.validateParameterBasis', side_effect=AssertionError('Unexpected gradient work')):
                osc.probability(events)

    def testOneEigendecompositionPerGroup(self):
        osc = self.makeOscillator(True)
        events = NeutrinoEventBatch([295., 295.], [.4, .6], [1, 1], [0, 0])
        with patch.object(osc.hamiltonian, '_eigh', wraps=osc.hamiltonian._eigh) as eigh:
            osc.probability(events, parameterBasis=self.basis)
            self.assertEqual(eigh.call_count, 1)

    def checkTorch(self, device):
        import torch
        events = NeutrinoEventBatch([295., 1300., 295.], [.6, 1.2, .7], [1, 1, 0], [0, 1, 2], [False, True, False])
        for useMatter in [False, True]:
            Backend.set_api(np)
            basis = self.basis + ([MatterDensity(), ElectronFraction()] if useMatter else [])
            reference = self.makeOscillator(useMatter).probability(events, parameterBasis=basis)
            Backend.set_api(torch, device=device)
            osc = self.makeOscillator(useMatter)
            p, g = osc.probability(events, parameterBasis=basis)
            tolerance = dict(rtol=3e-4, atol=3e-3) if device == 'mps' else dict(rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(p, reference[0], **tolerance)
            np.testing.assert_allclose(g, reference[1], **tolerance)
            np.testing.assert_allclose(p, osc.probability(events), **tolerance)
            # Propagation intermediates must stay on the configured device.
            from nu_waves.propagation.gradient import prepareModel
            model = osc._getExecutor().prepareModel(prepareModel(osc.hamiltonian, basis), basis)
            for value in model:
                if value is not None:
                    self.assertEqual(value.device.type, device)
            # Single-precision MPS needs larger steps than CPU double precision.
            # Fourth-order differences keep truncation error small at those steps.
            step = 2e-2 if device == 'mps' else 1e-3
            def shiftedProbability(offset):
                shifted = self.makeOscillator(useMatter)
                shifted.hamiltonian.mixing.mixing_angles[(2, 3)] += offset * step
                return shifted.probability(events)
            numerical = (shiftedProbability(-2) - 8*shiftedProbability(-1)
                         + 8*shiftedProbability(1) - shiftedProbability(2)) / (12*step)
            np.testing.assert_allclose(g[:, 0], numerical, rtol=3e-3, atol=3e-4)
            if useMatter:
                np.testing.assert_allclose(g[:, 3:], reference[1][:, 3:], rtol=2e-3, atol=2e-6)

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

    def testBackendPrecisionReset(self):
        Backend._real_dtype, Backend._complex_dtype = 'float32', 'complex64'
        Backend.set_api(np)
        self.assertEqual(Backend.real_dtype(), np.float64)
        self.assertEqual(Backend.complex_dtype(), np.complex128)
