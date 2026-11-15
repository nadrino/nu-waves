from Support.ExampleSupport import figureDirectory, dataDirectory, sampleCount
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.hamiltonian import vacuum
from nu_waves.propagation.oscillator import Oscillator
import nu_waves.utils.flavors as flavors
import nu_waves.utils.style

from nu_waves.globals.backend import Backend
# import torch
# Backend.set_api(torch, device='mps')

# import jax
# Backend.set_api(jax, device='mps')
# Backend.set_api(jax, device='cpu')

# 3 flavors PMNS, PDG values (2025)
angles = {(1, 2): np.deg2rad(33.4), (1, 3): np.deg2rad(8.6), (2, 3): np.deg2rad(49)}
phases = {(1, 3): np.deg2rad(195)}
dm2 = {(2, 1): 7.42e-5, (3, 2): 0.0024428}

# oscillator
h = vacuum.Hamiltonian(
    mixing=Mixing(n_neutrinos=3, mixing_angles=angles, dirac_phases=phases),
    spectrum=Spectrum(n_neutrinos=3, dm2=dm2),
    antineutrino=False
)
osc = Oscillator(hamiltonian=h)

# --- Grid definition (x: energy, y: baseline) ---
nE = sampleCount(1000)
nL = sampleCount(1000)
E_vals = np.linspace(0.2, 5.0, nE)     # GeV  (x-axis)
L_vals = np.linspace(1000, 2000.0, nL) # km   (y-axis)
E_grid, L_grid = np.meshgrid(E_vals, L_vals, indexing="xy")

L_flat = L_grid.ravel()  # shape (nE * nL,)
E_flat = E_grid.ravel()

# --- Choose vacuum or matter ---
# This example uses the vacuum Hamiltonian constructed above.

t0 = time.perf_counter()
# --- Compute probabilities in grid mode (shape -> (nL, nE) after selection) ---
# alpha=1 (muon), beta=0 (electron) → appearance
P_mue = osc.probability(L_km=L_flat, E_GeV=E_flat, flavor_emit=1, flavor_det=0)   # (nL, nE)
# alpha=1 (muon), beta=1 (muon) → disappearance
P_mumu = osc.probability(L_km=L_flat, E_GeV=E_flat, flavor_emit=1, flavor_det=1)  # (nL, nE)
t1 = time.perf_counter()
print(f"Computation time: {t1 - t0:.3f} s")

if nE * nL > 1E6:
    print("Too many bins to draw, skipping.")
    exit(0)

P_mue = P_mue.reshape(nL, nE)
P_mumu = P_mumu.reshape(nL, nE)


# --- Plot helper ---
def plot_oscillogram(ax, E, L, P, title):
    # imshow expects array indexed as [y, x] = [L, E]
    im = ax.imshow(
        P + 1e-10,  # avoid log(0)
        origin="lower",
        aspect="auto",
        extent=[E.min(), E.max(), L.min(), L.max()],
        norm=LogNorm(vmin=1e-3, vmax=1.0),
        interpolation="nearest",
        cmap='inferno'
    )
    ax.axhline(y=1300, color='white', linestyle='--', alpha=0.5)
    ax.set_xlabel(r"$E_\nu$ [GeV]")
    ax.set_ylabel(r"$L$ [km]")
    
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("Probability")

# --- Draw ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
plot_oscillogram(axes[0], E_vals, L_vals, P_mue,  r"$P(\nu_\mu\to\nu_e)$")
plot_oscillogram(axes[1], E_vals, L_vals, P_mumu, r"$P(\nu_\mu\to\nu_\mu)$")

# plt.savefig(str(figureDirectory / "vacuum_2d_pmns.pdf")) # too heavy
plt.savefig(str(figureDirectory / "vacuum_2d_pmns.jpg"), dpi=150) if not os.environ.get("CI") else None
plt.show()

