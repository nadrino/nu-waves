from Support.ExampleSupport import figureDirectory, dataDirectory, sampleCount
import numpy as np
import os
import matplotlib.pyplot as plt
from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.hamiltonian import vacuum
from nu_waves.globals.backend import Backend
from nu_waves.propagation.oscillator import Oscillator
import nu_waves.utils.flavors as flavors
import nu_waves.utils.style
import time


# import torch
# Backend.set_api(torch, device='mps')
# Backend.set_api(torch, device='cpu')

# import jax
# Backend.set_api(jax, device='mps')
# Backend.set_api(jax, device='cpu')

nPoints = sampleCount(int(1E5), 32)

# 3 flavors PMNS, PDG values (2025)
angles = {(1, 2): np.deg2rad(33.4), (1, 3): np.deg2rad(8.6), (2, 3): np.deg2rad(49)}
phases = {(1, 3): np.deg2rad(195)}
# Masses, normal ordering
dm2 = {(2, 1): 7.42e-5, (3, 2): 0.0024428}

h = vacuum.Hamiltonian(
    mixing=Mixing(n_neutrinos=3, mixing_angles=angles, dirac_phases=phases),
    spectrum=Spectrum(n_neutrinos=3, m_lightest=0, dm2=dm2),
    antineutrino=False
)
osc = Oscillator(hamiltonian=h)

# Compute probabilities:
# α = 1 (νμ source), β = [1,2,3] → (νμ, νe, ντ)
E_min, E_max = 0.2, 3.0
Enu_list = np.linspace(E_min, E_max, nPoints)

t0 = time.perf_counter()
P = osc.probability(
    L_km=295, E_GeV=Enu_list,
    flavor_emit=flavors.muon,
    flavor_det=[flavors.electron, flavors.muon, flavors.tau],
)
t1 = time.perf_counter()
print(f"Execution time: {t1 - t0:.3f} s")

if nPoints > int(1E5):
    print(f"Number of points: {nPoints} -> SKIPPING PLOT GENERATION")
    exit(0)

# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
plt.figure(figsize=(6.5, 4.0))

plt.plot(Enu_list, P[:, flavors.electron], label=r"$P_{\mu e}$ appearance", lw=2)
plt.plot(Enu_list, P[:, flavors.muon], label=r"$P_{\mu\mu}$ disappearance", lw=2)
plt.plot(Enu_list, P[:, flavors.tau], label=r"$P_{\mu\tau}$ appearance", lw=2)
plt.plot(Enu_list, P.sum(axis=1), "--", label="Total probability", lw=1.5)

plt.xlabel(r"$E_\nu$ [GeV]")
plt.ylabel(r"Probability")
plt.title(r"T2K-like vacuum oscillation ($L=295\,\mathrm{km}$)")
plt.xlim(E_min, E_max)
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()

plt.savefig(str(figureDirectory / "vacuum_pmns.jpg"), dpi=150)  if not os.environ.get("CI") else None
plt.show()

