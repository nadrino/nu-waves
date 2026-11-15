"""Check fitter update patterns and live parameter views independently of ROOT I/O."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import BenchmarkGundam as benchmark
from LoadGundamEvents import configureNuWaves, makeEventBatch


class TestBenchmarkGundam(unittest.TestCase):
    def testUpdatePatterns(self):
        calls = []

        def makeFake(parameters, *args):
            def probability(events, outputBuffer):
                calls.append(parameters.copy())
                outputBuffer[:] = .5
            return SimpleNamespace(probability=probability)

        args = SimpleNamespace(device="numpy", dtype="float64", density=2.6,
                               electronFraction=.5, modelLifetime="reuse", warmup=1, repeats=2)
        with patch.object(benchmark, "makeOscillator", side_effect=makeFake), patch("builtins.print"):
            results = benchmark.runBenchmark(SimpleNamespace(E_GeV=np.ones(2)), args)
        self.assertEqual(len(results), 23)
        for index, result in enumerate(results):
            proposals = calls[index * 3:(index + 1) * 3]
            expected = result["changedParameters"]
            for first, second in zip(proposals, proposals[1:]):
                changed = [benchmark.parameterNames[i] for i in np.flatnonzero(first != second)]
                self.assertEqual(changed, expected)
            self.assertEqual(len(result["samplesSeconds"]), 2)

    def testLiveMatterParameters(self):
        configureNuWaves(benchmark.Path(__file__).resolve().parents[1])
        arrays = dict(E_GeV=np.linspace(.2, 3., 12), flavor_emit=np.tile([0, 1], 6),
                      flavor_det=np.tile([1, 0], 6), isAntiNu=np.tile([True, False], 6))
        for device in ("numpy", "cpu"):
            if device == "cpu":
                import torch
                parameters = torch.tensor(benchmark.nominalValues, dtype=torch.float64)
                output = torch.empty(12, dtype=torch.float64)
            else:
                parameters = benchmark.nominalValues.copy()
                output = np.empty(12)
            events = makeEventBatch(arrays, 295., device)
            oscillator = benchmark.makeOscillator(parameters, 2.6, .5)
            previous = None
            for sign in (-1, 1):
                values = benchmark.nominalValues + sign * benchmark.parameterSteps
                parameters[:] = torch.tensor(values) if device == "cpu" else values
                oscillator.probability(events, outputBuffer=output)
                actual = output.numpy().copy() if device == "cpu" else output.copy()
                fresh = benchmark.makeOscillator(parameters, 2.6, .5)
                fresh.probability(events, outputBuffer=output)
                np.testing.assert_allclose(actual, output, atol=1e-12)
                if previous is not None:
                    self.assertGreater(float(np.max(np.abs(actual - previous))), 1e-5)
                previous = actual


if __name__ == "__main__":
    unittest.main()
