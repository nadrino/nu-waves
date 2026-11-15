# Running the examples

From a checkout, use the environment containing NumPy and Matplotlib:

```bash
python examples/matter_constant.py
python examples/vacuum_pmns.py --quick
python examples/matter_layer_mps.py --quick
```

The scripts also work when launched by absolute path from another directory.
`Support/ExampleSupport.py` locates the checkout's `src`, data and figure directories
relative to its own file. Figures go to `examples/figures/`. `CI=1` disables file
exports; `MPLBACKEND=Agg` disables interactive plot windows for automated runs.

`--quick` reduces grid sizes and Monte Carlo sample counts for smoke testing;
the original full sampling settings remain the default. The reduced grids are
not a convergence study. `NU_WAVES_EXAMPLE_QUICK=1` is equivalent to `--quick`.
Full runs such as `vacuum_2flavors.py` can require substantial memory.

The examples construct `Mixing`, `Spectrum(dm2=...)`, an explicit vacuum or
matter `Hamiltonian`, then `Oscillator(hamiltonian)`. Configure matter using
`hamiltonian.set_constant_density(...)` or `hamiltonian.set_matter_profile(...)`.
Use `MatterProfile` from `nu_waves.hamiltonian.matter`, not the legacy profile
module. Smearing uses `probability_sampled(...)` with explicit sampling arguments.

NumPy is the default. Torch is optional: `vacuum_backend.py` and
`matter_layer_mps.py` compare resident `NeutrinoEventBatch`/output buffers on
Torch CPU and available CUDA/MPS devices. They skip unavailable devices without
hiding numerical failures. MPS matter diagonalization uses the library's CPU
fallback. `probabilityOnDevice` in the helper shows construction of all nine
channels as a flat batch, reshaping the result to `[energy, emitted, detected]`.

## Solar examples

Solar/PREM scripts additionally require `tqdm`. The SSM solar examples require
the BS05-AGS,OP table, stored at `examples/data/ssm/bs05_agsop.dat`. To use another
location:

```bash
NU_WAVES_SOLAR_TABLE=/path/to/bs05_agsop.dat python examples/adiabatic_sun_ssm.py
```

If the default table is missing or invalid, full runs attempt a download from
the original IAS URL, validating it before replacing the cache. If unavailable,
they report how to supply the table. Quick runs without a usable default table
instead warn and use the analytic toy `SolarProfile`; those are smoke tests,
not BS05 predictions. An explicitly supplied invalid table always fails.

The removed solar methods on `Oscillator` are replaced by explicit NumPy
helpers in `Support/ExampleSupport.py`. These assume adiabatic following, tracking
instantaneous eigenvectors by overlap, and average relative phases before
projecting onto vacuum mass states. They omit coherent interference and
non-adiabatic jumps. `non_adiabatic_sun.py` retains an adiabatic curve with a
separate Landau–Zener diagnostic; the diagnostic does not modify that curve.
`non_adiabatic_sn.py` retains its separate phenomenological shock/jump example.

The Hyper-K notebook in `notebooks/` already uses the current Hamiltonian API;
it can be run from the repository root or the notebook directory.
