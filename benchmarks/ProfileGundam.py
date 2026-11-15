#!/usr/bin/env python3
"""Profile resident vacuum or matter batches; ROOT I/O and warmup are excluded."""
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from functools import wraps
import json
from pathlib import Path
import platform
import time
from unittest.mock import patch

import numpy as np

from BenchmarkGundam import makeOscillator, nominalValues, parameterSteps
from LoadGundamEvents import configureNuWaves, loadEvents, makeEventBatch, makeParser, resolveInputs, validateArgs


@contextmanager
def annotateStages():
    """Install profiling scopes temporarily, without changing production code."""
    import torch
    from nu_waves.hamiltonian.matter import Hamiltonian, ConstantMatterOptimizedExecutor
    from nu_waves.hamiltonian.vacuum import Hamiltonian as VacuumHamiltonian
    from nu_waves.models.mixing import Mixing
    from nu_waves.models.spectrum import Spectrum
    from nu_waves.propagation import BufferProbability

    def wrap(function, name):
        @wraps(function)
        def annotated(*args, **kwargs):
            with torch.profiler.record_function('nuWaves::' + name):
                return function(*args, **kwargs)
        return annotated

    targets = [(Mixing, 'build_mixing_matrix', 'mixingMatrix'),
               (Spectrum, 'relativeMasses', 'massSpectrum'),
               (ConstantMatterOptimizedExecutor, 'prepareModel', 'prepareMatterModel'),
               (Hamiltonian, 'matterEvolution', 'matterEvolution'),
               (Hamiltonian, 'matterAmplitude', 'matterAmplitude'),
               (Hamiltonian, '_eigh', 'eigendecomposition'),
               (VacuumHamiltonian, 'vacuumAmplitude', 'vacuumAmplitude'),
               (VacuumHamiltonian, 'vacuumPhases', 'vacuumPhases'),
               (BufferProbability, 'modelSnapshot', 'cacheModelSnapshot')]
    with ExitStack() as stack:
        for owner, attribute, label in targets:
            wrapped = wrap(getattr(owner, attribute), label)
            if isinstance(owner.__dict__.get(attribute), staticmethod):
                wrapped = staticmethod(wrapped)
            stack.enter_context(patch.object(owner, attribute, wrapped))
        yield


def profileCase(events, args, case, outputDirectory):
    import torch

    device = events.E_GeV.device
    synchronize = (lambda: torch.cuda.synchronize(device)) if device.type == 'cuda' else (lambda: None)
    parameters = torch.tensor(nominalValues, device=device, dtype=events.E_GeV.dtype)
    weights = torch.empty_like(events.E_GeV)
    changed = {'all': list(range(6)), 'unchanged': [], 'mass': [5], 'mixing': [2]}[case]
    proposals = []
    for sign in (-1, 1):
        values = nominalValues.copy()
        values[changed] += sign * parameterSteps[changed]
        proposals.append(torch.tensor(values, device=device, dtype=parameters.dtype))
    oscillator = makeOscillator(parameters, args.density, args.electronFraction, args.hamiltonian)
    iteration = 0

    def evaluate(annotated=False):
        nonlocal iteration, oscillator
        # Alternate continuously across warmup, baseline and profile captures.
        proposal = proposals[iteration % 2]
        iteration += 1
        if annotated:
            with torch.profiler.record_function('nuWaves::parameterUpdate'):
                parameters.copy_(proposal)
            if args.modelLifetime == 'rebuild':
                with torch.profiler.record_function('nuWaves::constructOscillator'):
                    oscillator = makeOscillator(parameters, args.density, args.electronFraction, args.hamiltonian)
            with torch.profiler.record_function('nuWaves::probability'):
                oscillator.probability(events, outputBuffer=weights)
        else:
            parameters.copy_(proposal)
            if args.modelLifetime == 'rebuild':
                oscillator = makeOscillator(parameters, args.density, args.electronFraction, args.hamiltonian)
            oscillator.probability(events, outputBuffer=weights)

    events.invalidateCache()
    print(f'{case}: warmup', flush=True)
    for _ in range(args.warmup):
        evaluate()
    samples = []
    for _ in range(args.repeats):
        synchronize()
        start = time.perf_counter()
        evaluate()
        synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        torch.cuda.reset_peak_memory_stats(device)
    synchronize()
    print(f'{case}: capture ({args.iterations} evaluations)', flush=True)
    with annotateStages(), torch.profiler.profile(
            activities=activities, record_shapes=args.recordShapes,
            profile_memory=args.profileMemory, with_stack=False) as profiler:
        for index in range(args.iterations):
            with torch.profiler.record_function(f'nuWaves::evaluation_{index}'):
                evaluate(annotated=True)
            with torch.profiler.record_function('nuWaves::waitForCompletion'):
                synchronize()
            profiler.step()
    averages = profiler.key_averages()
    cpuTable = averages.table(sort_by='self_cpu_time_total', row_limit=60)
    gpuTable = averages.table(sort_by='self_device_time_total', row_limit=60)
    (outputDirectory / f'{case}.txt').write_text(
        'CPU self times\n' + cpuTable + '\nDevice self times\n' + gpuTable + '\n')
    profiler.export_chrome_trace(str(outputDirectory / f'{case}.trace.json'))
    rows = [dict(name=op.key, calls=op.count, selfCpuUs=op.self_cpu_time_total,
                 cpuUs=op.cpu_time_total, selfDeviceUs=op.self_device_time_total,
                 deviceUs=op.device_time_total) for op in averages]
    cudaEvents = sum(str(event.device_type).endswith('CUDA') for event in profiler.events())
    result = dict(case=case, baselineMedianMs=float(np.median(samples)), baselineSamplesMs=samples,
                  profiledEvaluations=args.iterations, cudaEvents=cudaEvents, operations=rows)
    if device.type == 'cuda':
        result.update(peakAllocatedBytes=torch.cuda.max_memory_allocated(device),
                      peakReservedBytes=torch.cuda.max_memory_reserved(device))
    host = weights.detach().cpu().numpy()
    if not np.isfinite(host).all() or (host < -1e-5).any() or (host > 1 + 1e-5).any():
        raise RuntimeError('Invalid probabilities after profiling')
    print(f'{case}: baseline median {result["baselineMedianMs"]:.3f} ms; CUDA events={cudaEvents}', flush=True)
    if device.type == 'cuda' and not cudaEvents:
        print('WARNING: no CUDA kernel events captured. Check the CUPTI/profiler messages; CPU times alone do not locate GPU bottlenecks.', flush=True)
    return result


def main():
    parser = makeParser()
    parser.description = __doc__
    parser.set_defaults(device='cuda')
    parser.add_argument('--profile-dir', dest='profileDir')
    parser.add_argument('--hamiltonian', choices=['matter', 'vacuum'], default='matter')
    parser.add_argument('--cases', nargs='+', choices=['all', 'unchanged', 'mass', 'mixing'], default=['all', 'unchanged'])
    parser.add_argument('--iterations', type=int, default=3, help='Captured evaluations per case')
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=10, help='Unprofiled baseline evaluations per case')
    parser.add_argument('--density', type=float, default=2.6)
    parser.add_argument('--electron-fraction', dest='electronFraction', type=float, default=.5)
    parser.add_argument('--model-lifetime', dest='modelLifetime', choices=['rebuild', 'reuse'], default='rebuild')
    parser.add_argument('--cache-probability', dest='cacheProbability', action='store_true')
    parser.add_argument('--record-shapes', dest='recordShapes', action='store_true')
    parser.add_argument('--profile-memory', dest='profileMemory', action='store_true')
    args = parser.parse_args()
    if args.profileDir is None:
        args.profileDir = 'VacuumProfile' if args.hamiltonian == 'vacuum' else 'MatterProfile'
    validateArgs(parser, args)
    if args.output:
        parser.error('--output is not used here; use LoadGundamEvents.py for NPZ exports')
    if min(args.iterations, args.warmup, args.repeats) < 1:
        parser.error('iterations, warmup and repeats must be positive')
    if not np.isfinite(args.density) or args.density < 0 or not np.isfinite(args.electronFraction) or not 0 <= args.electronFraction <= 1:
        parser.error('invalid density or electron fraction')
    if args.device != 'cpu' and not args.device.startswith('cuda'):
        parser.error('use cuda[:index], or cpu for a CPU smoke test')
    configureNuWaves(args.nuWavesPath)
    import torch
    import nu_waves

    outputDirectory = Path(args.profileDir).expanduser()
    outputDirectory.mkdir(parents=True, exist_ok=False)
    print('Loading ROOT inputs once, then transferring the resident batch...', flush=True)
    sources = resolveInputs(args)
    arrays, counts = loadEvents(sources, maxEvents=args.maxEvents, stepSize=args.stepSize, dtype=args.dtype)
    events = replace(makeEventBatch(arrays, args.baselineKm, args.device, args.dtype),
                     cacheProbability=args.cacheProbability)
    del arrays
    metadata = dict(configuration=vars(args), counts=counts, inputs=sources,
                    nuWavesPath=nu_waves.__file__, python=platform.python_version(), torch=torch.__version__,
                    cudaVersion=torch.version.cuda, threads=torch.get_num_threads(), gradients=False)
    if events.E_GeV.device.type == 'cuda':
        metadata['gpu'] = torch.cuda.get_device_name(events.E_GeV.device)
    results = []
    for case in args.cases:
        results.append(profileCase(events, args, case, outputDirectory))
        (outputDirectory / 'summary.json').write_text(json.dumps(dict(metadata=metadata, results=results), indent=2) + '\n')
    print(f'Profile artifacts written to {outputDirectory.resolve()}', flush=True)


if __name__ == '__main__':
    main()
