#!/usr/bin/env python3
"""Load Config.yaml through compiled GUNDAM and exercise the real PythonWorker."""

import argparse
import json
from pathlib import Path
import sys


def referenceWeights(values, events):
    """Independent PDG amplitude, evaluated in NumPy for verification only."""
    import numpy as np
    s12, s13, s23 = np.sin(values[:3])
    c12, c13, c23 = np.cos(values[:3])
    phase = np.exp(1j * values[3])
    matrix = np.array([
        [c12*c13, s12*c13, s13*phase.conjugate()],
        [-s12*c23-c12*s23*s13*phase, c12*c23-s12*s23*s13*phase, s23*c13],
        [s12*s23-c12*c23*s13*phase, -c12*s23-s12*c23*s13*phase, c23*c13]])
    coefficients = matrix[events['flavorEmit'].astype(int)].conjugate() * matrix[events['flavorDetect'].astype(int)]
    coefficients = np.where(events['isAntiNu'][:, None] > .5, coefficients.conjugate(), coefficients)
    from nu_waves.utils.units import KM_TO_EVINV, GEV_TO_EV
    masses = np.array([0., values[4], values[4] + values[5]])
    phases = np.exp(-.5j * (events['L'] / events['E'])[:, None] * (KM_TO_EVINV / GEV_TO_EV) * masses)
    return np.abs(np.sum(phases * coefficients, axis=1))**2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["MPS", "CPU", "original"], default="MPS")
    parser.add_argument("--memory-location", choices=["onDevice", "onHost"], default="onDevice")
    parser.add_argument("--worker-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--gundam-module-dir", type=Path,
                        help="Directory containing the rebuilt GUNDAM Python module for this Python version")
    args = parser.parse_args()
    if args.gundam_module_dir:
        sys.path.insert(0, str(args.gundam_module_dir.resolve()))
    import GUNDAM
    import numpy as np
    import uproot

    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parents[1] / 'src'))
    directory = here / f"run-{args.backend}-{args.memory_location}"
    directory.mkdir(exist_ok=True)
    rootPath = directory / "Events.root"
    index = np.arange(108)
    events = dict(L=np.where(index % 2, 295., 810.), E=np.linspace(.2, 5., len(index)),
                  flavorEmit=(index % 3).astype(float), flavorDetect=((index // 3) % 3).astype(float),
                  isAntiNu=((index // 9) % 2).astype(float))
    with uproot.recreate(rootPath) as file:
        tree = file.mktree("tree_mc", {name: 'float64' for name in events})
        tree.extend(events)

    # YAML is parsed by GUNDAM. There is no Python YAML parser dependency.
    builder = GUNDAM.ConfigUtils.ConfigBuilder()
    builder.setConfigFromYamlString((here / "Config.yaml").read_text())
    config = json.loads(builder.getConfig().toString())
    likelihood = config["fitterEngineConfig"]["likelihoodInterfaceConfig"]
    likelihood["dataSetList"][0]["model"]["filePathList"] = [str(rootPath)]
    manager = likelihood["backendManagerConfig"]
    manager["isEnabled"] = args.backend != "original"
    manager["type"] = "CPU" if args.backend == "original" else args.backend
    worker = likelihood["propagatorConfig"]["parametersManagerConfig"]["parameterSetList"][0]["dialSetDefinitions"][0]["options"]["workerConfig"]
    worker["pythonExecutable"] = str(args.worker_python.absolute())
    worker["evalScript"] = str(here / "Reweight.py")
    worker["backendBufferMemoryLocation"] = args.memory_location
    auditPath = directory / 'WorkerAudit.jsonl'
    auditPath.write_text('')
    worker['scriptArgs'] = [str(auditPath)]
    configPath = directory / "ResolvedConfig.json"
    configPath.write_text(json.dumps(config, indent=2))
    builder.setConfig(str(configPath))

    print("GUNDAM module:", GUNDAM.__file__, flush=True)
    print("Worker Python:", worker["pythonExecutable"], flush=True)
    print("Backend:", args.backend, "Requested memory:", args.memory_location, flush=True)
    GUNDAM.setRuntimeWorkingDirectory(str(directory))
    GUNDAM.setLightOutputMode(True)
    GUNDAM.setNumberOfThreads(1)
    reader = GUNDAM.ConfigUtils.ConfigReader(builder.getConfig())
    reader.defineField(GUNDAM.ConfigUtils.ConfigReader.FieldDefinition("fitterEngineConfig"))
    engine = GUNDAM.FitterEngine()
    engine.setConfig(reader.fetchValueConfigReader("fitterEngineConfig"))
    engine.configure()
    engine.initialize()
    likelihoodInterface = engine.getLikelihoodInterface()
    parameters = likelihoodInterface.getModelPropagator().getParametersManager().getParameterSetsList()[0].getParameterList()
    points = [[.59, .15, .81, -1.6, 7.5e-5, 2.5e-3],
              [.61, .14, .72, 1.2, 7.6e-5, -2.4e-3]]
    for values in [points[0], points[1], points[1], points[0]]:
        for parameter, value in zip(parameters, values):
            parameter.setParameterValue(value)
        engine.evaluateLikelihood()
        GUNDAM.flushOutput()
        sample = likelihoodInterface.getModelPropagator().getSampleSet().getSampleList()[0]
        actual = [entry.sumWeights for entry in sample.getHistogram().getBinContentList()]
        weights = referenceWeights(values, events)
        masks = [events['E'] < 1., events['E'] >= 1.]
        expected = [float(weights[mask].sum()) for mask in masks]
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
        actualSquares = [entry.sqrtSumSqWeights**2 for entry in sample.getHistogram().getBinContentList()]
        np.testing.assert_allclose(actualSquares, [float((weights[mask]**2).sum()) for mask in masks], rtol=1e-5, atol=1e-5)
        device = json.loads(auditPath.read_text().splitlines()[-1])['device']
        expectedDevice = 'mps' if args.backend == 'MPS' and args.memory_location == 'onDevice' else 'cpu'
        if device != expectedDevice:
            raise RuntimeError(f'Expected {expectedDevice} evaluation, got {device}')
        print(f"PASS PMNS={values}: {device} histogram={actual}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
