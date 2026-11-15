# OA2026 event loading

## CUDA profiling

```bash
python benchmarks/ProfileGundam.py \
  --dataset-list /path/to/gundam/config/dataset/datasetList.yaml \
  --nu-waves-path ./ --device cuda --cache-probability \
  --cases all unchanged --warmup 3 --repeats 10 --iterations 3 \
  --profile-dir MatterProfile
```

This loads/transfers the batch once for all cases. It measures an unprofiled
wall-clock baseline after warmup, then captures CPU and CUDA activity using
Torch Profiler. `all` alternates all six PMNS parameters; `unchanged` exercises
cache hits if enabled. Optional `mass` changes only delta_m2_32; `mixing` changes
only theta_23. Probabilities are computed without gradients, matching the
existing benchmark. The output directory must not already exist.

Return the complete directory: `summary.json` records the environment,
unprofiled timings and operation statistics, each `<case>.txt` contains CPU/GPU
tables, and `<case>.trace.json` contains a Chrome/Perfetto timeline including
CUDA kernels, memory transfers, synchronization and `nuWaves::` stage labels.
If CUDA capture is unavailable (e.g. CUPTI restrictions), the script reports
zero CUDA events and warns; CPU-only output cannot establish GPU bottlenecks.
`--device cpu` is available for local smoke testing only.

Stage scopes are temporary instrumentation installed only by this script.
`matterAmplitude` identifies the direct requested-channel calculation used for
constant matter without gradients. It avoids constructing the full propagator
with a batched matrix multiplication. Gradients and layered profiles retain
the full propagator path.
`matterEvolution` includes Hamiltonian assembly, eigendecomposition and
propagator construction; its nested `eigendecomposition` scope isolates the
solver. Native operators (`exp`, `bmm`, indexing, copies, etc.) give the finer
breakdown. Inclusive parent/child times must not be summed. Device operation
time sums are not necessarily elapsed time; inspect the timeline for overlap
and CPU launch gaps. Per-operation times aggregate all captured evaluations.

The separate unprofiled baseline is the performance reference: profiler hooks
change timings. Shapes and memory tracking are disabled by default to reduce
distortion; enable `--record-shapes` / `--profile-memory` for a deeper second
capture. Peak CUDA allocated/reserved memory is reported when available, but
profiler overhead can affect it. This is operator/timeline profiling, not a
hardware-counter occupancy or bandwidth analysis.

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
Both modes call nu-waves on every iteration, including unchanged parameters.
Add `--cache-probability` to enable the library's batch result cache. The cache
belongs to the batch, so it also works with `--model-lifetime rebuild`. Without
this flag every iteration recomputes the probabilities. With the flag, only
the unchanged case hits: changing cases alternate two proposals and the cache
retains only the most recent result. Cache lookup and result copying are timed.

The matter buffer implementation computes one propagator per event and layer,
with vectorized neutrino/antineutrino signs, then selects the requested flavor
amplitude. Energies need not repeat. Earlier versions evaluated both signs and
all nine flavor pairs separately, doing eighteen times as many event
diagonalizations. This optimization applies even when all PMNS parameters
change. It does not cache probabilities across evaluations or use energy bins;
unchanged parameters still trigger recomputation unless caching is enabled.

### Batch probability cache API

```python
events = NeutrinoEventBatch(lengths, energies, emitted, detected, anti,
                            cacheProbability=True)
oscillator.probability(events, outputBuffer=weights)
# Repeated calls with the same physical model reuse the last result.
# After ANY modification of event arrays, including shared/external writes:
events.invalidateCache()
```

Caching is opt-in, for the buffer evaluation path (explicit `outputBuffer`, or
automatic output allocation for device batches). It assumes event arrays stay
fixed until `invalidateCache()` is called. No checksum, event comparison or
implicit detection of in-place event edits is performed. Separate batches have
separate caches. Recreating a batch creates an empty cache.

The key compares exact values of the small prepared physical model and matter
profile, output device/precision/shape, CP default and gradient basis. Parameter
changes, including writes through external shared buffers, are detected by value,
not by tensor identity or version. Gradients are cached together with their
probabilities. Custom profile tangents are included in the model comparison.
CUDA model comparisons involve host readback/synchronization; event arrays and
cached results stay on their device. This mode therefore relaxes the ordinary
buffer path's guarantee of no host synchronization.

Only one result is retained, requiring an additional N-element probability
array and, if requested, an N-by-P gradient array. Cache hits copy into the
requested output buffers; caller writes to previous results do not corrupt the
cache. `invalidateCache()` also releases these retained arrays. The cache does
not currently retain event preprocessing or skip small model construction.

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
