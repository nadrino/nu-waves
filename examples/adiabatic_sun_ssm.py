from Support.ExampleSupport import figureDirectory, dataDirectory, sampleCount
import os
from pathlib import Path
import urllib.request
import numpy as np
import matplotlib.pyplot as plt

from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.propagation.oscillator import Oscillator
from nu_waves.hamiltonian import vacuum
from Support.ExampleSupport import adiabaticMassFractions, initialMassComposition, loadSolarProfile, solarHamiltonians
from nu_waves.matter.solar import load_bs05_agsop
import nu_waves.utils.style


sol = loadSolarProfile()

# 3) Oscillator (NumPy backend)
angles = {(1, 2): np.deg2rad(33.4), (1, 3): np.deg2rad(8.6), (2, 3): np.deg2rad(49)}
phases = {(1, 3): np.deg2rad(195)}

# Masses, normal ordering
spec = Spectrum(n_neutrinos=3, dm2={(2, 1): 7.42e-5, (3, 2): 0.0024428})

osc = Oscillator(vacuum.Hamiltonian(Mixing(n_neutrinos=3, mixing_angles=angles, dirac_phases=phases), spec, False))

# 4) Physics knobs
E_GeV   = 0.008                      # 10 MeV
r_emit  = 0.05 * sol.R_sun_km        # emission radius
# distances from emission to the surface; include 0 for marker
s_max   = sol.R_sun_km - r_emit
s_km    = np.concatenate(([0.0], np.geomspace(1.0, s_max, sampleCount(799))))  # (n,)

# 5) Adiabatic evolution from emission
F = adiabaticMassFractions(osc,
    E_GeV=E_GeV, profile=sol, r_emit_km=r_emit, s_km=s_km, alpha=0, antineutrino=False
)  # shape (n, N)

# Consistency check: initial point equals vacuum_from_matter at emission
F0_vfm = initialMassComposition(osc,
    alpha=0, basis="vacuum_from_matter", E_GeV=E_GeV, profile=sol, r_emit_km=r_emit
)
# np.testing.assert_allclose(F[0], F0_vfm, atol=1e-10)

# 6) Plot vs radius (log x)
r_path = r_emit + s_km
x = r_path / sol.R_sun_km  # r/R_sun in [~0.05, 1]
labels = [r"$\nu_1$", r"$\nu_2$", r"$\nu_3$"]
colors = ["C0", "C1", "C2"]

plt.figure(figsize=(7.5, 4.2))
for i, (lab, col) in enumerate(zip(labels, colors)):
    plt.plot(x, F[:, i], lw=2, label=lab, color=col)
    # initial and final markers
    plt.scatter([x[0], x[-1]], [F[0, i], F[-1, i]], s=25, color=col, zorder=5)

plt.xscale("log")
plt.xlabel(r"$r/R_\odot$")
plt.ylabel("Mass–state fraction (adiabatic)")
plt.ylim(0, 1.0)
plt.grid(True, which="both", alpha=0.3)
plt.legend(frameon=False)
plt.tight_layout()

plt.savefig(str(figureDirectory / "adiabatic_sun_ssm_test.jpg"), dpi=150) if not os.environ.get("CI") else None
plt.show()
