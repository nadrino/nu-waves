# nu-waves with real GUNDAM buffers

This draft exercises `Oscillator.probability(events, outputBuffer=weights)`
through GUNDAM's real ExternalWeight/PythonWorker transport. Compile GUNDAM
with its Python interface and the desired backend first.

`Run.py` creates 108 ROOT events covering all flavor channels, neutrinos and
antineutrinos, and two baselines. It loads `Config.yaml`, runs four parameter
points (including a repeat and inverted ordering), and checks histogram sums
and squared weights against an independent NumPy PDG amplitude. It also checks
that the worker actually evaluated on the requested device. Artifacts go to
`run-*`. There is no native compilation or Python YAML parser dependency here.

The runner needs GUNDAM, NumPy and uproot. The worker needs NumPy and, for MPS,
Torch. `Reweight.py` imports nu-waves directly from this checkout.

```bash
python drafts/PythonWorkerBuffers/Run.py --backend MPS \
  --worker-python drafts/venv/bin/python
python drafts/PythonWorkerBuffers/Run.py --backend CPU \
  --worker-python drafts/venv/bin/python
python drafts/PythonWorkerBuffers/Run.py --backend MPS --memory-location onHost \
  --worker-python drafts/venv/bin/python
python drafts/PythonWorkerBuffers/Run.py --backend original \
  --worker-python drafts/venv/bin/python
```

Use `--gundam-module-dir` to select a rebuilt GUNDAM module outside PYTHONPATH.
Its Python version must match the runner; the worker may use another venv.

Host buffers stay NumPy float64; MPS buffers stay Torch float32. The worker
passes the original arrays to nu-waves without `.to()`, `.cpu()` or NumPy
conversions. GUNDAM may initialize on CPU before activating MPS and restarting
the worker. The bootstrap handles synchronization before acknowledging results.

The example supplies six PMNS parameters in the YAML order. Slice each scalar
as `parameterBuffer[i:i+1].reshape(())` to retain a view on both NumPy and Torch.
An integer index into a NumPy array would instead copy a scalar. No explicit
nu-waves backend selection is required: the output buffer selects it locally.

Device inputs also keep results on their device when no output buffer is
supplied, including analytic gradients. Providing GUNDAM's output buffers still
avoids allocating a separate result. Mixing host arrays/CPU tensors and GPU
arrays is an error. Python numeric constants (for example a fixed baseline or
mixing angle) are permitted; zero-dimensional CPU tensors remain host data.

## Gradients

nu-waves accepts `parameterBasis=BASIS` and
`gradientOutputBuffer=command['gradients']`, with shape `(N, len(BASIS))`.
It writes physical derivatives dP/dp in the requested basis order, on the same
device and in the same real dtype as the weights. If the gradient buffer is
omitted, nu-waves allocates a Jacobian on that same device.

GUNDAM currently transports gradients on host only. The example advertises
`supportsGradient` on CPU, handles `withGradient`, and acknowledges
`gradientGeneration`. Once GUNDAM exports device gradients, the same call can
use them; update that capability declaration when the transport is available.
The runner verifies weights; device gradients are covered separately by
`tests/TestProbabilityBuffers.py`.

## Adapting the production PMNS worker

The existing production script's NumPy preparation and `compileEvents()` path
must be bypassed when using caller-owned buffers:

- Build `NeutrinoEventBatch` from the supplied event arrays. Floating flavor
  codes must be exact zero-based integers; antineutrinos use `isAntiNu > 0.5`.
  nu-waves converts those categories on their current device. For a fixed
  baseline, `energy * 0 + 295.` creates an array on the energy's device.
- Bind Mixing/Spectrum values to scalar views of `parameterBuffer`, mapping
  names through `parameterInfo` if the configured parameter order can vary.
  Python constants can supply fixed parameters. Avoid `float(tensor)` and
  comparisons of parameter values on the host.
- Call `oscillator.probability(events, outputBuffer=command['weights'], ...)`
  directly. Remove the result's `np.asarray` and the subsequent output copy.
  The existing compiled-event grouping and NumPy binning are not supported by
  this buffer path. Start without `--bin` / `--ptheta-binning`.
- Set gradient support according to GUNDAM's actual transport. Keep the
  existing `withGradient` and generation acknowledgement protocol.

The default production setup uses vacuum propagation, which is supported on
MPS. Matter buffer evaluation also supports MPS: eigendecomposition temporarily
transfers matrices to CPU and its results back to MPS. Final probability and
gradient buffers stay on MPS. CUDA execution is implemented through Torch but requires validation on
CUDA hardware and a future GUNDAM CUDA buffer transport.
