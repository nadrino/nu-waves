from Support.ExampleSupport import figureDirectory, dataDirectory, sampleCount
import numpy as np
import os
import matplotlib.pyplot as plt
from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.hamiltonian import vacuum
from nu_waves.propagation.oscillator import Oscillator
from nu_waves.globals.backend import Backend
import nu_waves.utils.flavors as flavors
import nu_waves.utils.style
import time


# Backend.set_api(torch, device='mps')
# Backend.set_api(torch, device='cpu')

# import jax
# Backend.set_api(jax, device='mps')
# Backend.set_api(jax, device='cpu')

nBins_L = sampleCount(200, 32)
nSamples_E = sampleCount(int(1E6), 32)
E_res = 0.1

# 3 flavors PMNS, PDG values (2025)
n_neutrinos = 2
osc_amplitude = 0.2 # sin^2(2\theta)
angles = {(1, 2): np.arcsin(np.sqrt(osc_amplitude))/2}
dm2 = {(2, 1): 1}

# oscillator
h = vacuum.Hamiltonian(
    mixing=Mixing(n_neutrinos=n_neutrinos, mixing_angles=angles),
    spectrum=Spectrum(n_neutrinos=n_neutrinos, dm2=dm2),
    antineutrino=True
)
osc = Oscillator(hamiltonian=h)

E_fixed = 3E-3
L_min, L_max = 1e-3, 20e-3
L_list = np.linspace(L_min, L_max, nBins_L)
P = osc.probability(
    L_km=L_list, E_GeV=E_fixed,
    flavor_emit=flavors.electron,
    flavor_det=flavors.electron, # muon could be sterile
)


# Example: fractional Gaussian energy resolution and fixed baseline blur
def gaussian_E_sampler(E, n_samples, sigma_rel=E_res):
    # E: (nE,)
    xp = Backend.xp()
    noise = sigma_rel * xp.abs(E)[:, None] * xp.asarray(
        xp.random.standard_normal((E.shape[0], n_samples)), dtype=Backend.real_dtype()
    )
    out = E[:, None] + noise
    return xp.maximum(out, xp.asarray(1e-12, dtype=Backend.real_dtype()))

t0 = time.perf_counter()
P_damp = osc.probability_sampled(
    L_km=L_list, E_GeV=E_fixed,
    flavor_emit=flavors.electron,
    flavor_det=flavors.electron, # muon could be sterile
    n_samples=nSamples_E,
    E_sample_fct=gaussian_E_sampler,
)
t1 = time.perf_counter()
print(f"Execution time: {t1 - t0:.3f} s")

# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
plt.figure(figsize=(6.5, 4.0), dpi=150)

plt.plot(L_list*1000, P, label=r"$P_{e e}$ disappearance", lw=2)
plt.plot(L_list*1000, P_damp, label=f"$P_{{e e}}$ disappearance ($\\sigma$(E)/E = {E_res*100}%)", lw=2)
plt.plot(L_list*1000, [1]*len(L_list), "--", label="Total probability", lw=1.5)

plt.xlabel(r"$L_\nu$ [m]")
plt.ylabel(r"Probability")
plt.title(f"eV$^2$ sterile with $E_\\nu$ = {E_fixed*1000} MeV")
# plt.xlim(L_min, L_max)
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()

plt.savefig(str(figureDirectory / "vacuum_2flavors.jpg"), dpi=150)  if not os.environ.get("CI") else None
plt.show()
