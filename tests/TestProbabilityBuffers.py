"""Probability buffers: device ownership, physical gradients and parameter updates."""
import importlib.util
import unittest
from unittest.mock import patch

import numpy as np

from nu_waves.globals.backend import Backend
from nu_waves.hamiltonian import vacuum, matter
from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.propagation.oscillator import Oscillator, NeutrinoEventBatch
from nu_waves.propagation.gradient import MixingAngle, DiracPhase, MassSquaredDifference


VALUES = [.59, .15, .81, -1.6, 7.5e-5, 2.5e-3]
BASIS = (MixingAngle(1, 2), MixingAngle(1, 3), MixingAngle(2, 3),
         DiracPhase(1, 3), MassSquaredDifference(2, 1), MassSquaredDifference(3, 2))


def makeOscillator(values, withMatter=False):
    if isinstance(values, np.ndarray):
        values = [values[i:i + 1].reshape(()) for i in range(6)]
    model = matter if withMatter else vacuum
    hamiltonian = model.Hamiltonian(
        Mixing(3, {(1, 2): values[0], (1, 3): values[1], (2, 3): values[2]}, {(1, 3): values[3]}),
        Spectrum(3, {(2, 1): values[4], (3, 2): values[5]}), False)
    if withMatter:
        hamiltonian.set_constant_density(2.6, Ye=.5)
    return Oscillator(hamiltonian)


class TestProbabilityBuffers(unittest.TestCase):
    def checkDevicePolicy(self, device):
        import torch
        Backend.set_api(np)
        dtype = torch.float32 if device == 'mps' else torch.float64
        energy = torch.linspace(.3, 3., 9, device=device, dtype=dtype)
        # Constant baseline, channel, CP flag and model parameters are context.
        events = NeutrinoEventBatch(295., energy, 1, 0, False)
        oscillator = makeOscillator(VALUES)
        referenceEvents = NeutrinoEventBatch(np.full(9, 295.), energy.cpu().numpy(),
                                             np.ones(9, dtype=int), np.zeros(9, dtype=int))
        expected, expectedGradient = oscillator.probability(referenceEvents, parameterBasis=BASIS)
        with patch.object(Backend, 'from_device', side_effect=AssertionError('Host read')), \
             patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('Host read')), \
             patch.object(torch.Tensor, 'item', side_effect=AssertionError('Scalar read')):
            output, gradients = oscillator.probability(events, parameterBasis=BASIS)
            weights = oscillator.probability(events)
            suppliedGradient = torch.empty_like(gradients)
            _, returnedGradient = oscillator.probability(events, parameterBasis=BASIS,
                                                         gradientOutputBuffer=suppliedGradient)
            positional, positionalGradient = oscillator.probability(295., energy, 1, 0, parameterBasis=BASIS)
            channels = oscillator.probability(295., energy, [0, 1], None)
        self.assertIs(returnedGradient, suppliedGradient)
        for value in (output, gradients, weights, positional, positionalGradient, channels):
            self.assertEqual(value.device, energy.device)
        self.assertEqual(channels.shape, (9, 2, 3))
        np.testing.assert_allclose(output.cpu().numpy(), expected, atol=3e-5, rtol=0)
        np.testing.assert_allclose(positional.cpu().numpy(), expected, atol=3e-5, rtol=0)
        np.testing.assert_allclose(gradients.cpu().numpy(), expectedGradient, atol=.04, rtol=1e-4)
        np.testing.assert_allclose(positionalGradient.cpu().numpy(), expectedGradient, atol=.04, rtol=1e-4)
        self.assertIs(Backend.xp(), np)
        # Host arrays and even CPU scalar tensors remain data, not constants.
        for hostValue in (np.full(9, 295.), torch.full((9,), 295.), torch.tensor(295.)):
            with self.assertRaisesRegex((ValueError, TypeError), 'L_km'):
                oscillator.probability(NeutrinoEventBatch(hostValue, energy, 1, 0))
            with self.assertRaisesRegex(ValueError, 'L_km'):
                oscillator.probability(hostValue, energy, 1, 0)
        with self.assertRaises((ValueError, TypeError)):
            oscillator.probability(events, outputBuffer=np.empty(9))
        with self.assertRaises((ValueError, TypeError)):
            oscillator.probability(events, parameterBasis=BASIS, gradientOutputBuffer=np.empty((9, 6)))
        oscillator.hamiltonian.mixing.mixing_angles[(1, 2)] = np.array(.59)
        with self.assertRaisesRegex(ValueError, 'model parameter'):
            oscillator.probability(events)
        with self.assertRaisesRegex(ValueError, 'host storage'):
            makeOscillator(VALUES).compileEvents(events)
        # A device parameter with host events must also trigger the policy.
        oscillator = makeOscillator(VALUES)
        oscillator.hamiltonian.mixing.mixing_angles[(1, 2)] = energy[0]
        with self.assertRaisesRegex(ValueError, 'L_km'):
            oscillator.probability(referenceEvents)

    def checkBuffers(self, device=None, withMatter=False, count=109):
        Backend.set_api(np)
        index = np.arange(count)
        arrays = [np.full(count, 295.), np.linspace(.2, 5., count),
                  (index % 3).astype(float), ((index // 3) % 3).astype(float),
                  ((index // 9) % 2).astype(float)]
        referenceEvents = NeutrinoEventBatch(*arrays)
        if device:
            import torch
            dtype = torch.float32 if device == 'mps' else torch.float64
            convert = lambda value: torch.as_tensor(value, device=device, dtype=dtype)
            storage = torch.full((count * 2 + 4,), -7., device=device, dtype=dtype)
            gradientStorage = torch.full((count + 2, 6), -9., device=device, dtype=dtype)
        else:
            convert = np.asarray
            storage = np.full(count * 2 + 4, -7.)
            gradientStorage = np.full((count + 2, 6), -9.)
        output = storage[2:-2:2]
        gradient = gradientStorage[1:-1]
        parameters = convert(VALUES).copy() if device is None else convert(VALUES).clone()
        oscillator = makeOscillator(parameters, withMatter)
        events = NeutrinoEventBatch(*(convert(value) for value in arrays))
        for cp in (-1.6, 1.2):
            parameters[3] = cp
            parameters[4] = 7.5e-5 if cp < 0 else 7.6e-5
            parameters[5] = 2.5e-3 if cp < 0 else -2.4e-3
            values = list(VALUES)
            values[3] = cp
            values[4] = 7.5e-5 if cp < 0 else 7.6e-5
            values[5] = 2.5e-3 if cp < 0 else -2.4e-3
            expected, jacobian = makeOscillator(values, withMatter).probability(referenceEvents, parameterBasis=BASIS)
            with patch.object(Backend, 'from_device', side_effect=AssertionError('Host read')):
                if device:
                    with patch.object(torch.Tensor, 'item', side_effect=AssertionError('Scalar read')), \
                         patch.object(torch.Tensor, 'cpu', side_effect=AssertionError('Host read')), \
                         patch.object(torch.Tensor, 'numpy', side_effect=AssertionError('Host read')):
                        result, derivatives = oscillator.probability(events, parameterBasis=BASIS,
                            outputBuffer=output, gradientOutputBuffer=gradient)
                else:
                    result, derivatives = oscillator.probability(events, parameterBasis=BASIS,
                        outputBuffer=output, gradientOutputBuffer=gradient)
            self.assertIs(result, output)
            self.assertIs(derivatives, gradient)
            self.assertIs(Backend.xp(), np)
            toHost = (lambda x: x.cpu().numpy()) if device else np.asarray
            np.testing.assert_allclose(toHost(output), expected, atol=3e-5 if device == 'mps' else 1e-12, rtol=0)
            np.testing.assert_allclose(toHost(gradient), jacobian, atol=.04 if device == 'mps' else 1e-8, rtol=1e-4 if device == 'mps' else 1e-10)
            np.testing.assert_array_equal(toHost(storage)[::2][[0, -1]], [-7., -7.])
            np.testing.assert_array_equal(toHost(storage)[1::2], -7.)
            np.testing.assert_array_equal(toHost(gradientStorage)[[0, -1]], -9.)
            weightsOnly = oscillator.probability(events, outputBuffer=output)
            self.assertIs(weightsOnly, output)
            allocated = oscillator.probability(events, outputBuffer=output, parameterBasis=BASIS)[1]
            self.assertEqual(allocated.shape, gradient.shape)

    def testNumpy(self):
        self.checkBuffers()

    def testNumpyMatter(self):
        self.checkBuffers(withMatter=True, count=9)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Torch is not installed')
    def testTorchCpu(self):
        self.checkBuffers('cpu')

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Torch is not installed')
    def testTorchCpuMatter(self):
        self.checkBuffers('cpu', withMatter=True, count=9)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Torch is not installed')
    def testMps(self):
        import torch
        if not torch.backends.mps.is_available():
            self.skipTest('MPS unavailable')
        self.checkBuffers('mps', count=65539)
        self.checkDevicePolicy('mps')

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Torch is not installed')
    def testCuda(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest('CUDA unavailable')
        self.checkBuffers('cuda')
        self.checkDevicePolicy('cuda')

    def testInvalidBuffers(self):
        oscillator = makeOscillator(VALUES)
        events = NeutrinoEventBatch(*(np.ones(2) for _ in range(5)))
        with self.assertRaises(ValueError):
            oscillator.probability(events, outputBuffer=np.empty(3))
        with self.assertRaises(TypeError):
            oscillator.probability(events, outputBuffer=np.empty(2, dtype=np.int64))
        with self.assertRaises(ValueError):
            oscillator.probability(events, outputBuffer=np.empty(2), gradientOutputBuffer=np.empty((2, 6)))
        with self.assertRaises(ValueError):
            oscillator.probability(events, outputBuffer=np.empty(2), parameterBasis=BASIS, gradientOutputBuffer=np.empty((6, 2)))

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Torch is not installed')
    def testMixedStorage(self):
        import torch
        events = NeutrinoEventBatch(*(np.ones(2) for _ in range(5)))
        with self.assertRaises(ValueError):
            makeOscillator(VALUES).probability(events, outputBuffer=torch.empty(2))
        events = NeutrinoEventBatch(*(torch.ones(2, dtype=torch.float64) for _ in range(5)))
        with self.assertRaises(TypeError):
            makeOscillator(VALUES).probability(events, outputBuffer=torch.empty(2, dtype=torch.float32))


if __name__ == '__main__':
    unittest.main()
