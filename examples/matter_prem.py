import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import time

from nu_waves.globals.backend import Backend
from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.hamiltonian import matter
from nu_waves.propagation.oscillator import Oscillator
from nu_waves.matter.prem import PREMModel
from nu_waves.matter.profile import MatterProfile
import nu_waves.utils.style


import torch
Backend.set_api(torch, device='mps') # no eigen decompose in MPS, will fallback to CPU
# Backend.set_api(torch, device='cpu') # fastest

# import jax
# Backend.set_api(jax, device='mps')
# Backend.set_api(jax, device='cpu')


E_GeV = np.logspace(-1, 2, 400)     # x
cosz_binning  = np.linspace(-1.0, 1.0, 400)     # y (upgoing)
prem  = PREMModel()

# choose one:
# SCHEME = "prem_layers"      # PREM shells
SCHEME = "hist_density"   # fine histogram of density along path
n_bins_layers = 1000
n_bins_density = 100 #

# 3 flavors PMNS, PDG values (2025)
angles = {(1, 2): np.deg2rad(33.4), (1, 3): np.deg2rad(8.6), (2, 3): np.deg2rad(49)}
phases = {(1, 3): np.deg2rad(195)}
# Masses, normal ordering
dm2 = {(2, 1): 7.42e-5, (3, 2): 0.0024428}

h_matter = matter.Hamiltonian(
    mixing=Mixing(n_neutrinos=3, mixing_angles=angles, dirac_phases=phases),
    spectrum=Spectrum(n_neutrinos=3, m_lightest=0, dm2=dm2),
    antineutrino=False
)
osc = Oscillator(hamiltonian=h_matter)

# test for thickness
cosz_profile = prem.profile_from_coszen(+0.3, h_atm_km=15.0)
print("L_tot(downgoing) =", sum(L.weight for L in cosz_profile.layers))  # ≈ 15 km

for cosz in (-1e-3, +1e-3):
    cosz_profile = prem.profile_from_coszen(cosz, h_atm_km=15.0)
    print(cosz, "L_atm =", sum(layer.weight for layer in cosz_profile.layers if layer.rho_in_g_per_cm3 == 0.0))

# --- arrays to hold all 4 panels ---
P_mue      = np.zeros((len(cosz_binning), len(E_GeV)))
P_mumu     = np.zeros_like(P_mue)
P_mue_bar  = np.zeros_like(P_mue)
P_mumu_bar = np.zeros_like(P_mue)

t0 = time.perf_counter()
for iy, cosz in tqdm(enumerate(cosz_binning), total=len(cosz_binning)):
    # 1) build PREM profile for this cos(zenith)
    cosz_profile = prem.profile_from_coszen(
        cosz, scheme=SCHEME,
        n_bins=n_bins_layers, nbins_density=n_bins_density, merge_tol=0.0,  # keep your knobs
        h_atm_km=15.0                                  # thin atmosphere
    )
    h_matter.set_matter_profile(cosz_profile)

    # 2) total baseline for this row (sum of absolute segments)
    L_tot = float(sum(layer.weight for layer in cosz_profile.layers))

    # 3) compute ν and ν̄ for the two channels
    osc.hamiltonian.set_antineutrino(False)
    P_mu_i    = osc.probability(L_km=L_tot, E_GeV=E_GeV, flavor_emit=1, flavor_det=[0, 1])

    osc.hamiltonian.set_antineutrino(True)
    P_mubar_i = osc.probability(L_km=L_tot, E_GeV=E_GeV, flavor_emit=1, flavor_det=[0, 1])

    P_mue[iy], P_mumu[iy] = P_mu_i[..., 0], P_mu_i[..., 1]
    P_mue_bar[iy], P_mumu_bar[iy] = P_mubar_i[..., 0], P_mubar_i[..., 1]

t1 = time.perf_counter()
print(f"Computation time: {t1 - t0:.3f} s")

E_edges  = np.geomspace(E_GeV.min(), E_GeV.max(), E_GeV.size + 1)
CZ_edges = np.linspace(cosz_binning.min(), cosz_binning.max(), cosz_binning.size + 1)

def draw_panel(ax, Z, label_tex, text_color, fontsize=20):
    pc = ax.pcolormesh(
        E_edges, CZ_edges, Z,
        vmin=0.0, vmax=1.0, shading="auto",
        cmap="inferno_r"
    )
    ax.set_xscale("log")

    # add grid lines
    # ax.grid(True, which="both", color="w", alpha=0.25, lw=0.5)
    ax.grid(True, which="both", color="w", alpha=0.3, lw=0.4, ls="--")
    # optional: add minor ticks for better readability
    # ax.minorticks_on()

    ax.text(0.96, 0.96, label_tex,
            transform=ax.transAxes, ha="right", va="top",
            color=text_color, fontsize=fontsize, weight="bold")
    return pc

# --- create figure with constrained layout (better with colorbars) ---
fig, axs = plt.subplots(
    2, 2, figsize=(9.8, 8.0),
    dpi=150,
    constrained_layout=True
)

# draw all panels
m0 = draw_panel(axs[0,0], P_mue,      r"$P_{\nu_\mu \rightarrow \nu_e}$",      "black")
_  = draw_panel(axs[0,1], P_mumu,     r"$P_{\nu_\mu \rightarrow \nu_\mu}$",    "white")
_  = draw_panel(axs[1,0], P_mue_bar,  r"$P_{\bar{\nu}_\mu \rightarrow \bar{\nu}_e}$",  "black")
_  = draw_panel(axs[1,1], P_mumu_bar, r"$P_{\bar{\nu}_\mu \rightarrow \bar{\nu}_\mu}$","white")

# axis labels
for ax in axs[0,:]:
    ax.set_xlabel("")
for ax in axs[:,1]:
    ax.set_ylabel("")
for ax in axs[1,:]:
    ax.set_xlabel(r"$E_\nu$ [GeV]")
for ax in axs[:,0]:
    ax.set_ylabel(r"$\cos\theta_z$")

# single colorbar (automatically aligned to full figure height)
cbar = fig.colorbar(m0, ax=axs, location="right", fraction=0.05, pad=0.03)
cbar.set_label("Oscillation probability", labelpad=10, fontsize=13)

plt.savefig("./figures/matter_prem_test.pdf") # too heavy
plt.savefig("./figures/matter_prem_test.jpg", dpi=150) if not os.environ.get("CI") else None
plt.show()
