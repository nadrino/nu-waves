# OA2026 event loading

## Matter probability benchmark

```bash
python benchmarks/BenchmarkGundam.py \
  --dataset-list /path/to/gundam/config/dataset/datasetList.yaml \
  --nu-waves-path ./ --device cuda \
  --warmup 2 --repeats 10 --report /tmp/MatterBenchmark.json
```

This runs 23 cases: all six PMNS parameters change, none change, each of the
six changes alone, and each of the fifteen pairs changes. Each case alternates
two nearby proposals around the same nominal point; it does not accumulate
steps or change the matter density. Defaults are 295 km, 2.6 g/cm³, Ye=0.5,
float64, and probabilities without gradients. Warmup is repeated per case.
Start with `--max-events 10000` to estimate runtime before loading the full batch.

Times include parameter-buffer assignment, model reconstruction and probability
evaluation, with CUDA synchronized before and after every call. ROOT I/O,
initial event/proposal transfers, output-buffer allocation, and validation
readback are excluded. The output remains on the requested device. Each row
shows median milliseconds per complete batch and events/s; the JSON report
also records raw samples, minimum, p95, configuration and GPU/Torch information.
Use `--device cpu` for Torch CPU or `--device numpy` for NumPy comparisons.
The percentile is descriptive; increase repetitions for stable statistics.

`--model-lifetime rebuild` (default) reconstructs the oscillator every call as
the supplied GUNDAM worker does. Repeat with `--model-lifetime reuse` to retain
one oscillator per case with live scalar views into its parameter buffer.
Both modes call nu-waves on every iteration, including unchanged parameters;
there is no benchmark-side probability cache. Thus unchanged timings measure
the current recomputation cost, not the cost of a fitter skipping the worker.

The matter buffer implementation computes one propagator per event and layer,
with vectorized neutrino/antineutrino signs, then selects the requested flavor
amplitude. Energies need not repeat. Earlier versions evaluated both signs and
all nine flavor pairs separately, doing eighteen times as many event
diagonalizations. This optimization applies even when all PMNS parameters
change. It does not cache probabilities across evaluations or use energy bins;
unchanged parameters still trigger recomputation.

The Torch backend splits CUDA eigendecompositions into at most 16,384 matrices
per solver call. This avoids large-batch `cusolverDnXsyevBatched_bufferSize`
failures seen with some PyTorch/cuSOLVER versions. All matrices are still
processed on the GPU; splitting is included in the measured execution time.

`LoadGundamEvents.py` prepares a `NeutrinoEventBatch` independently of GUNDAM.
Install `numpy`, `uproot` (including Awkward), and `PyYAML` in your environment.
Torch is needed only for `--device cpu` or `--device cuda`.

From the repository root, on CC-IN2P3:

```bash
export OA_INPUT_FOLDER=/path/to/inputs
python benchmarks/LoadGundamEvents.py \
  --dataset-list /path/to/gundam/config/dataset/datasetList.yaml \
  --nu-waves-path /path/to/nu-waves \
  --device cuda --dtype float64 \
  --max-events 1000000 --output /tmp/NeutrinoEvents.npz
```

Alternatively pass ROOT files or quoted globs directly:

```bash
python benchmarks/LoadGundamEvents.py '/path/to/*_nueselec.root' \
  --tree mtuple --device numpy
```

The nu-waves path accepts a checkout root, `src`, or `src/nu_waves`.
The actual imported path is printed. Relative ROOT paths are interpreted from
`--input-base` (the working directory by default); environment variables in
YAML paths are expanded. Only a dataset list is supported, not recursive
GUNDAM fit configuration resolution. No YAML expressions are executed.

The loader reads `pnu[0]` in GeV, `ipnu[0]`, `OscChannel` and `mode`, in chunks.
Channels 1/2 map muon to muon, 3/4 electron to electron and 5/6 muon to electron;
the antineutrino flag comes from `ipnu[0] < 0`, as in the worker. Baseline defaults
to 295 km. Neutral currents (`31 <= abs(mode) <= 52`) are excluded, as in the
PMNS dial's apply condition. Invalid energies, channels and neutrino PDG codes
are counted and discarded. The row limit applies after filtering, across files;
the read/discard counters cover entire chunks, including the final chunk.

This loads individual events, without sample cuts, nominal weights, friend
trees or the worker's `useBinnedWeights` reduction. Event counts therefore need
not match selected GUNDAM samples or the number of probability bins. This is
input preparation only; it does not yet measure probability evaluation speed.
Chunked reading limits temporary I/O memory, but the final batch resides fully
in RAM and, with CUDA, on the GPU. Use `--max-events` to control memory usage.
Float64 is the default; flavor indices remain int64 and CP flags boolean.

For a later benchmark, import `configureNuWaves`, `loadEvents` and
`makeEventBatch` from this script. `loadEvents` returns host arrays and counters;
`makeEventBatch` creates the resident batch once, outside the future timing
loop. The NPZ export contains the five named batch fields on the host.
