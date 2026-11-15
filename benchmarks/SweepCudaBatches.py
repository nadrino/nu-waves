#!/usr/bin/env python3
"""Compare eigensolver batch sizes on a single resident matter event batch."""
from contextlib import contextmanager
import json
import math
from pathlib import Path
import platform
import random
import time
from unittest.mock import patch

import numpy as np

from BenchmarkGundam import makeOscillator, nominalValues, parameterSteps
from LoadGundamEvents import configureNuWaves, loadEvents, makeEventBatch, makeParser, resolveInputs, validateArgs


@contextmanager
def solverBatchSize(size, cpuSmoke=False):
    from nu_waves.backends.torch_backend import TorchBackend

    factory = TorchBackend._make_linalg_namespace

    def makeNamespace(backend):
        namespace = factory(backend)
        # CPU normally bypasses splitting; force it only in local smoke tests.
        namespace.eigh = lambda matrices: backend.eighBatched(matrices, size)
        return namespace

    with patch.object(TorchBackend, 'cudaEighBatchSize', size):
        if cpuSmoke:
            with patch.object(TorchBackend, '_make_linalg_namespace', makeNamespace):
                yield
        else:
            yield


def runSweep(events, args, saveReport):
    import torch

    device = events.E_GeV.device
    cuda = device.type == 'cuda'
    synchronize = (lambda: torch.cuda.synchronize(device)) if cuda else (lambda: None)
    parameters = torch.tensor(nominalValues, dtype=events.E_GeV.dtype, device=device)
    proposals = [torch.tensor(nominalValues + sign * parameterSteps,
                              dtype=parameters.dtype, device=device) for sign in (-1, 1)]
    weights = torch.empty_like(events.E_GeV)
    iteration = 0

    def evaluate():
        nonlocal iteration
        parameters.copy_(proposals[iteration % 2])
        iteration += 1
        # Match the worker: all PMNS parameters change and the model is rebuilt.
        makeOscillator(parameters, args.density, args.electronFraction).probability(events, outputBuffer=weights)

    # Two proposal references at the existing default size, outside the timer.
    references = []
    with solverBatchSize(16384, not cuda):
        for _ in range(2):
            evaluate()
            references.append(weights.detach().cpu().numpy().copy())
    if any(not np.isfinite(reference).all() or (reference < -1e-5).any() or
           (reference > 1 + 1e-5).any() for reference in references):
        raise RuntimeError('Invalid reference probabilities')
    measurements = []
    rng = random.Random(args.seed)
    for roundIndex in range(args.rounds):
        order = list(args.batchSizes)
        rng.shuffle(order)
        print(f'Round {roundIndex + 1}/{args.rounds}: {order}', flush=True)
        for size in order:
            with solverBatchSize(size, not cuda):
                # Validate both proposals, including all channels/signs present.
                tolerance = 1e-12 if args.dtype == 'float64' else 3e-5
                for _ in range(2):
                    proposalIndex = iteration % 2
                    evaluate()
                    np.testing.assert_allclose(weights.detach().cpu().numpy(), references[proposalIndex],
                                               atol=tolerance, rtol=tolerance)
                for _ in range(args.warmup):
                    evaluate()
                synchronize()
                if cuda:
                    torch.cuda.reset_peak_memory_stats(device)
                samples = []
                for _ in range(args.repeats):
                    synchronize()
                    start = time.perf_counter()
                    evaluate()
                    synchronize()
                    samples.append((time.perf_counter() - start) * 1000)
                measurement = dict(round=roundIndex + 1, batchSize=size, samplesMs=samples,
                                   medianMs=float(np.median(samples)))
                if cuda:
                    measurement.update(peakAllocatedBytes=torch.cuda.max_memory_allocated(device),
                                       peakReservedBytes=torch.cuda.max_memory_reserved(device))
                measurements.append(measurement)
                print(f'  {size:6d}: {measurement["medianMs"]:9.3f} ms', flush=True)
                saveReport(measurements, [])
    results = []
    for size in args.batchSizes:
        samples = [sample for row in measurements if row['batchSize'] == size for sample in row['samplesMs']]
        median = float(np.median(samples))
        count = len(events.E_GeV)
        calls = sum(math.ceil(min(65536, count - start) / size) for start in range(0, count, 65536))
        results.append(dict(batchSize=size, medianMs=median, minMs=min(samples),
                            p95Ms=float(np.percentile(samples, 95)), eventsPerSecond=count / (median / 1000),
                            expectedSolverCalls=calls))
    results.sort(key=lambda row: row['medianMs'])
    saveReport(measurements, results)
    print('Ranking (full evaluations, without profiler):', flush=True)
    for row in results:
        print(f'{row["batchSize"]:6d}: {row["medianMs"]:9.3f} ms; {row["eventsPerSecond"]:12.0f} events/s; '
              f'{row["expectedSolverCalls"]} solver calls', flush=True)


def main():
    parser = makeParser()
    parser.description = __doc__
    parser.set_defaults(device='cuda')
    parser.add_argument('--batch-sizes', dest='batchSizes', type=int, nargs='+',
                        default=[4096, 8192, 16384, 32768, 49152, 61440])
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--density', type=float, default=2.6)
    parser.add_argument('--electron-fraction', dest='electronFraction', type=float, default=.5)
    parser.add_argument('--report', default='CudaBatchSweep.json')
    args = parser.parse_args()
    validateArgs(parser, args)
    if args.output:
        parser.error('--output is not used; export inputs using LoadGundamEvents.py')
    if min(args.rounds, args.repeats, args.warmup) < 1:
        parser.error('rounds, repeats and warmup must be positive')
    if not args.batchSizes or any(size < 1 or size > 61440 for size in args.batchSizes):
        parser.error('batch sizes must lie in [1, 61440] to avoid the observed large-batch solver failure')
    if len(set(args.batchSizes)) != len(args.batchSizes):
        parser.error('batch sizes must be unique')
    if not np.isfinite(args.density) or args.density < 0 or not np.isfinite(args.electronFraction) or not 0 <= args.electronFraction <= 1:
        parser.error('invalid density or electron fraction')
    if args.device != 'cpu' and not args.device.startswith('cuda'):
        parser.error('use cuda[:index], or cpu for local smoke testing')
    configureNuWaves(args.nuWavesPath)
    import torch
    import nu_waves

    print('Loading ROOT files once and transferring events...', flush=True)
    sources = resolveInputs(args)
    arrays, stats = loadEvents(sources, maxEvents=args.maxEvents, stepSize=args.stepSize, dtype=args.dtype)
    events = makeEventBatch(arrays, args.baselineKm, args.device, args.dtype)
    del arrays
    metadata = dict(configuration=vars(args), counts=stats, inputs=sources, probabilityCache=False,
                    gradients=False, modelLifetime='rebuild', nuWavesPath=nu_waves.__file__,
                    python=platform.python_version(), torch=torch.__version__, cudaVersion=torch.version.cuda,
                    threads=torch.get_num_threads(), referenceBatchSize=16384)
    if events.E_GeV.device.type == 'cuda':
        metadata['gpu'] = torch.cuda.get_device_name(events.E_GeV.device)
    else:
        print('CPU smoke test: forced solver splitting; these timings do not predict CUDA performance.', flush=True)

    def saveReport(measurements, results):
        Path(args.report).write_text(json.dumps(dict(metadata=metadata, measurements=measurements,
                                                   results=results), indent=2) + '\n')

    runSweep(events, args, saveReport)


if __name__ == '__main__':
    main()
