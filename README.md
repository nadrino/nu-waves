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
import nu_waves.utils.flavors as flavors

# sterile test
osc_amplitude = 0.1  # sin^2(2\theta)
angles = {(1, 2): np.arcsin(np.sqrt(osc_amplitude)) / 2}
pmns = Mixing(n_neutrinos=2, mixing_angles=angles)
U_pmns = pmns.build_mixing_matrix()
print(np.round(U_pmns, 3))

# 1 eV^2
spec = Spectrum(n_neutrinos=2, m_lightest=0.)
spec._generate_dm2_matrix({(2, 1): 1})
spec.summary()
m2_diag = np.diag(spec.get_m2())

# oscillator object that calculates the oscillation probability
osc = Oscillator(mixing_matrix=U_pmns, m2_list=spec.get_m2())

# get the oscillation probabilities
E_fixed = 3E-3
L_min, L_max = 1e-3, 20e-3
L_list = np.linspace(L_min, L_max, 200)
print(L_list)
P = osc.probability(
    L_km=L_list, E_GeV=E_fixed,
    flavor_emit=flavors.electron,
    flavor_det=flavors.electron,  # muon could be sterile
    antineutrino=True
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


