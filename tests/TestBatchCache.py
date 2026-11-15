"""Explicit batch cache lifetime, parameter changes, gradients and output ownership."""
from dataclasses import replace
import importlib.util
import unittest
from unittest.mock import patch

import numpy as np

from TestProbabilityBuffers import makeOscillator, VALUES, BASIS
from nu_waves.propagation.oscillator import NeutrinoEventBatch
from nu_waves.propagation.gradient import MatterDensity


class TestBatchCache(unittest.TestCase):
    def checkCache(self, device, withMatter):
        parameters = np.array(VALUES)
        values = [np.full(54, 295.), np.linspace(.2, 5., 54),
                  np.arange(54) % 3, (np.arange(54) // 3) % 3, (np.arange(54) // 9) % 2 == 0]
        output = np.empty(54)
        if device != 'numpy':
            import torch
            parameters = torch.tensor(parameters, device=device)
            values = [torch.as_tensor(value, device=device) for value in values]
            output = torch.empty(54, device=device, dtype=torch.float64)
        toHost = lambda value: value.copy() if device == 'numpy' else value.cpu().numpy().copy()
        events = NeutrinoEventBatch(*values, cacheProbability=True)

        def evaluate(basis=None):
            # A fresh oscillator simulates the worker's normal lifecycle.
            oscillator = makeOscillator(parameters, withMatter)
            return oscillator.probability(events, outputBuffer=output, parameterBasis=basis)

        expected = toHost(evaluate())
        output[:] = -7
        with patch('nu_waves.hamiltonian.matter.Hamiltonian._eigh', side_effect=AssertionError('cache miss')), \
             patch('nu_waves.hamiltonian.vacuum.Hamiltonian.vacuumAmplitude', side_effect=AssertionError('cache miss')):
            self.assertIs(evaluate(), output)
        np.testing.assert_allclose(toHost(output), expected)
        # External writes into a shared parameter buffer must invalidate hits.
        parameters[3] = .8
        changed = toHost(evaluate())
        self.assertGreater(np.max(np.abs(changed - expected)), 1e-5)
        np.testing.assert_allclose(changed, toHost(evaluate()))
        # Gradients cannot reuse a probability-only cache entry.
        basis = BASIS + ((MatterDensity(),) if withMatter else ())
        _, gradient = evaluate(basis)
        expectedGradient = toHost(gradient)
        gradient[:] = -99
        _, gradient = evaluate(basis)
        np.testing.assert_allclose(toHost(gradient), expectedGradient)
        if withMatter:
            oscillator = makeOscillator(parameters, True)
            oscillator.hamiltonian.set_constant_density(4.)
            dense = toHost(oscillator.probability(events, outputBuffer=output, parameterBasis=basis)[0])
            self.assertGreater(np.max(np.abs(dense - changed)), 1e-5)
        # Caller-owned event arrays are checked only by explicit invalidation.
        evaluate()
        events.E_GeV[:] *= 1.1
        events.invalidateCache()
        edited = toHost(evaluate())
        fresh = replace(events, cacheProbability=False)
        oscillator = makeOscillator(parameters, withMatter)
        np.testing.assert_allclose(edited, toHost(oscillator.probability(fresh, outputBuffer=output)))
        self.assertGreater(np.max(np.abs(edited - changed)), 1e-5)

    def testNumpy(self):
        for matter in (False, True):
            self.checkCache('numpy', matter)

    def testTorchCpu(self):
        if not importlib.util.find_spec('torch'):
            self.skipTest('Torch is not installed')
        for matter in (False, True):
            self.checkCache('cpu', matter)

    def testCuda(self):
        if not importlib.util.find_spec('torch'):
            self.skipTest('Torch is not installed')
        import torch
        if not torch.cuda.is_available():
            self.skipTest('CUDA unavailable')
        for matter in (False, True):
            self.checkCache('cuda', matter)
