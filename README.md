# Neutrino Interferometry - nu-waves

## What is it?

**Neutrino Interferometry**, or `nu_waves`, is a simple Python library
that calculate flavor oscillation of neutrinos.
You can input your own parameters and get the oscillation probabilities.


## How to install?

```bash
pip install nu-waves
```


## Features

- Embedded GPU acceleration (MPS, CUDA)
- Oscillation framework with `N` neutrinos
- Vacuum oscillations
- Custom smearing function (L and E)
- Constant matter MSW
- Multi-layer matter MSW
- Earth model (PREM) with `cosz`
- Adiabatic transitions

## Oscillation convention

The active three-flavor PMNS matrix uses the PDG convention,
`U_e3 = sin(theta13) * exp(-1j * deltaCP)`. The evolution operator stores
`S[energy, detected, emitted]`, while public probabilities retain the order
`P[energy, emitted, detected] = abs(S[energy, detected, emitted])**2`
(singleton axes may be squeezed). Antineutrinos conjugate the mixing matrix
and reverse the matter potential. Matter layers are ordered from source to detector.

The CP convention correction changes predictions from earlier implementations
that transposed transition amplitudes. Remove any external phase-sign or channel-swap
workarounds when adopting this correction, and regenerate affected predictions.
The notebook `notebooks/04_vacuum_executor_benchmarks.ipynb` checks an independent
PDG amplitude before benchmarking; `tests/TestPdgConvention.py` also covers matter,
antineutrinos, layer ordering and decoherence.

## Some nice pictures

![vacuum_pmns.jpg](figures/vacuum_pmns.jpg)
![matter_constant_test.jpg](figures/matter_constant_test.jpg)
![matter_prem_test.jpg](figures/matter_prem_test.jpg)
![adiabatic_sun_ssm_test.jpg](figures/adiabatic_sun_ssm_test.jpg)
![vacuum_2d_pmns.jpg](figures/vacuum_2d_pmns.jpg)
![vacuum_2flavors.jpg](figures/vacuum_2flavors.jpg)

## Examples

### 2 flavors oscillation in vacuum

```python
import numpy as np
import matplotlib.pyplot as plt
from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.propagation.oscillator import Oscillator
from nu_waves.hamiltonian import vacuum
from nu_waves.utils import flavors

# sterile test
osc_amplitude = 0.1  # sin^2(2\theta)
angles = {(1, 2): np.arcsin(np.sqrt(osc_amplitude)) / 2}
dm2={(2, 1): 1}

mixing = Mixing(n_neutrinos=2, mixing_angles=angles)
spectrum = Spectrum(n_neutrinos=2, m_lightest=0., dm2={(2, 1): 1})
H = vacuum.Hamiltonian(
    mixing=mixing,
    spectrum=spectrum,
    antineutrino=False
)

# oscillator object that calculates the oscillation probability
osc = Oscillator(hamiltonian=H)

# get the oscillation probabilities
E_fixed = 3E-3
L_min, L_max = 1e-3, 20e-3
L_list = np.linspace(L_min, L_max, 200)
print(L_list)
P = osc.probability(
    L_km=L_list, E_GeV=E_fixed,
    flavor_emit=flavors.electron,
    flavor_det=flavors.electron  # muon could be sterile
)

# draw it
plt.figure(figsize=(6.5, 4.0))

plt.plot(L_list * 1000, P, label=r"$P_{e e}$ disappearance", lw=2)
plt.plot(L_list * 1000, [1] * len(L_list), "--", label="Total probability", lw=1.5)

plt.xlabel(r"$L_\nu$ [m]")
plt.ylabel(r"Probability")
plt.title(f"eV$^2$ sterile with $E_\\nu$ = {E_fixed * 1000} MeV")
# plt.xlim(L_min, L_max)
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()
plt.show()
```


### Analytic probability gradients

Vacuum, constant-density and layered matter Hamiltonians support physical parameter
Jacobians through the same interface as probability evaluation:

```python
from nu_waves.propagation.gradient import MixingAngle, DiracPhase, MassSquaredDifference

probabilities, gradients = osc.probability(
    L_km=295.0, E_GeV=energies,
    flavor_emit=flavors.muon, flavor_det=flavors.electron,
    parameterBasis=[MixingAngle(2, 3), DiracPhase(1, 3), MassSquaredDifference(3, 2)],
)
```

For constant matter, the basis may also contain density and electron fraction:

```python
from nu_waves.propagation.gradient import MatterDensity, ElectronFraction

probabilities, gradients = osc.probability(
    events,
    parameterBasis=[MixingAngle(2, 3), MatterDensity(), ElectronFraction()],
)
```

The density derivative uses g/cm³ as its input unit; electron fraction is
dimensionless. Each partial derivative holds the other parameters fixed.
These parameters require a matter Hamiltonian and can be mixed,
reordered or repeated in the basis. Their contributions are added directly to
the Hamiltonian tangent, reusing the same eigendecomposition. No division by
density or electron fraction is needed, so zero values are supported.
Since the potential depends on their product, the derivatives satisfy
`rho * dP/drho = Ye * dP/dYe`.

For layered profiles, select each density or electron fraction using an explicit
zero-based `layerIndex`:

```python
from nu_waves.hamiltonian.matter import MatterProfile

osc.hamiltonian.set_matter_profile(MatterProfile.from_fractions(
    rho_gcm3=[2.6, 4.1], Ye=[0.5, 0.47], fractions=[0.4, 0.6],
))
probabilities, gradients = osc.probability(
    events,
    parameterBasis=[MixingAngle(2, 3), MatterDensity(layerIndex=0),
                    ElectronFraction(layerIndex=1)],
)
```

Both fractional profiles and absolute segments (`from_segments`, lengths in km)
are supported. Layers are applied in their declared traversal order. Layer widths
remain fixed; oscillation parameters affect every layer, while an indexed matter
parameter affects only its selected layer. A missing or invalid layer index is
rejected for layered profiles. Constant-density media continue to use
`MatterDensity()` and `ElectronFraction()` without an index.

Probability-only evaluation and gradients share the same layer-product kernel.
Derivatives use `d(U S) = dU S + U dS`, reusing one eigendecomposition per layer
and event for all requested parameters. No tangent arrays are allocated when
`parameterBasis` is omitted. NumPy and Torch CPU/CUDA/MPS use the same kernels;
MPS retains the backend's existing CPU fallback for eigendecomposition.
