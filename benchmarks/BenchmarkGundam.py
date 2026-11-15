#!/usr/bin/env python3
"""Benchmark constant-density matter probabilities under fitter parameter updates."""

from itertools import combinations
import json
from pathlib import Path
import platform
import time

import numpy as np

from LoadGundamEvents import configureNuWaves, loadEvents, makeEventBatch, makeParser, resolveInputs, validateArgs


parameterNames = ("theta_12", "theta_13", "theta_23", "delta_cp", "delta_m2_21", "delta_m2_32")
nominalValues = np.array([.5872523687443223, .14819001778459273, .8134128187551903,
                          -1.601, 7.53e-5, 2.509e-3])
parameterSteps = np.array([.005, .002, .01, .05, 1e-6, 2e-5])


def makeOscillator(parameters, density, electronFraction):
    from nu_waves.hamiltonian import matter
    from nu_waves.models.mixing import Mixing
    from nu_waves.models.spectrum import Spectrum
    from nu_waves.propagation.oscillator import Oscillator

    # Keep live scalar views into the parameter buffer, including with NumPy.
    p = [parameters[i:i + 1].reshape(()) for i in range(6)]
    hamiltonian = matter.Hamiltonian(
        Mixing(3, {(1, 2): p[0], (1, 3): p[1], (2, 3): p[2]}, {(1, 3): p[3]}),
        Spectrum(3, {(2, 1): p[4], (3, 2): p[5]}), False)
    hamiltonian.set_constant_density(density, Ye=electronFraction)
    hamiltonian.enableConstantMatterBatchOptimization = True
    return Oscillator(hamiltonian)


def runBenchmark(events, args):
    """Time update + optional model construction + probability + device completion.

    Parameter proposals and event transfers are prepared outside the timer.
    No probability cache or gradients are used, even for unchanged parameters.
    """
    torch = None
    if args.device == "numpy":
        convert = lambda value: np.asarray(value, dtype=args.dtype).copy()
        synchronize = lambda: None
        toHost = np.asarray
    else:
        import torch

        device = events.E_GeV.device
        convert = lambda value: torch.tensor(value, device=device, dtype=events.E_GeV.dtype)
        synchronize = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
        toHost = lambda value: value.detach().cpu().numpy()
    count = len(events.E_GeV)
    parameters = convert(nominalValues)
    output = convert(np.zeros(count))
    cases = [("all", tuple(range(6))), ("unchanged", ())]
    cases += [("single:" + parameterNames[i], (i,)) for i in range(6)]
    cases += [("pair:" + parameterNames[i] + "+" + parameterNames[j], (i, j))
              for i, j in combinations(range(6), 2)]
    results = []
    for name, changed in cases:
        parameters[...] = convert(nominalValues)
        oscillator = makeOscillator(parameters, args.density, args.electronFraction)
        # Alternate proposals around nominal: no drift and every changing call
        # changes exactly the named coordinates, including after warmup.
        proposals = []
        for sign in (-1, 1):
            proposal = nominalValues.copy()
            proposal[list(changed)] += sign * parameterSteps[list(changed)]
            proposals.append(convert(proposal))
        samples = []
        firstOutput = None
        for iteration in range(args.warmup + args.repeats):
            proposal = proposals[iteration % 2]
            synchronize()
            start = time.perf_counter()
            parameters[...] = proposal
            if args.modelLifetime == "rebuild":
                oscillator = makeOscillator(parameters, args.density, args.electronFraction)
            oscillator.probability(events, outputBuffer=output)
            synchronize()
            elapsed = time.perf_counter() - start
            if iteration >= args.warmup:
                samples.append(elapsed)
                # Readback and validation are deliberately outside the timer.
                if firstOutput is None:
                    firstOutput = toHost(output).copy()
        finalOutput = toHost(output)
        if not np.isfinite(finalOutput).all() or np.any(finalOutput < -1e-5) or np.any(finalOutput > 1 + 1e-5):
            raise RuntimeError(f"Invalid probabilities for {name}")
        if not changed:
            np.testing.assert_allclose(finalOutput, firstOutput, atol=1e-6, rtol=1e-6)
        median = float(np.median(samples))
        result = dict(case=name, changedParameters=[parameterNames[i] for i in changed],
                      medianMs=median * 1000, minMs=min(samples) * 1000,
                      p95Ms=float(np.percentile(samples, 95)) * 1000,
                      eventsPerSecond=count / median, samplesSeconds=samples,
                      probabilityMean=float(finalOutput.mean()))
        results.append(result)
        print(f"{name:48s} {median * 1000:12.3f} ms  {count / median:12.0f} events/s", flush=True)
    return results


def main():
    parser = makeParser()
    parser.description = __doc__
    parser.set_defaults(device="cuda")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--density", type=float, default=2.6, help="Constant density in g/cm^3")
    parser.add_argument("--electron-fraction", dest="electronFraction", type=float, default=.5)
    parser.add_argument("--model-lifetime", dest="modelLifetime", choices=["rebuild", "reuse"], default="rebuild",
                        help="rebuild matches the GUNDAM worker; reuse retains live parameter views")
    parser.add_argument("--report", help="Write complete benchmark results as JSON")
    args = parser.parse_args()
    validateArgs(parser, args)
    if args.repeats < 2 or args.warmup < 1:
        parser.error("use --repeats >= 2 and --warmup >= 1")
    if not np.isfinite(args.density) or args.density < 0:
        parser.error("density must be finite and nonnegative")
    if not np.isfinite(args.electronFraction) or not 0 <= args.electronFraction <= 1:
        parser.error("electron fraction must lie in [0, 1]")
    if args.device != "numpy" and args.device != "cpu" and not args.device.startswith("cuda"):
        parser.error("benchmark supports numpy, cpu and cuda[:index]")
    configureNuWaves(args.nuWavesPath)
    import nu_waves

    sources = resolveInputs(args)
    arrays, stats = loadEvents(sources, maxEvents=args.maxEvents, stepSize=args.stepSize, dtype=args.dtype)
    events = makeEventBatch(arrays, args.baselineKm, args.device, args.dtype)
    if args.output:
        np.savez(args.output, L_km=np.full(stats["loaded"], args.baselineKm, dtype=args.dtype), **arrays)
    metadata = dict(nuWavesPath=nu_waves.__file__, python=platform.python_version(), numpy=np.__version__,
                    device=args.device, dtype=args.dtype, modelLifetime=args.modelLifetime,
                    density=args.density, electronFraction=args.electronFraction, baselineKm=args.baselineKm,
                    warmup=args.warmup, repeats=args.repeats, counts=stats,
                    nominalParameters=dict(zip(parameterNames, nominalValues.tolist())),
                    parameterSteps=dict(zip(parameterNames, parameterSteps.tolist())),
                    inputs=sources, gradients=False, probabilityCache=False)
    if args.device != "numpy":
        import torch

        metadata.update(torch=torch.__version__, cudaVersion=torch.version.cuda, threads=torch.get_num_threads())
        if events.E_GeV.device.type == "cuda":
            metadata["gpu"] = torch.cuda.get_device_name(events.E_GeV.device)
    print(json.dumps(metadata, indent=2), flush=True)
    results = runBenchmark(events, args)
    if args.report:
        Path(args.report).write_text(json.dumps(dict(metadata=metadata, results=results), indent=2) + "\n")


if __name__ == "__main__":
    main()
