# OA2026 event loading

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
