from Support.ExampleSupport import figureDirectory, dataDirectory, sampleCount
import numpy as np
import matplotlib.pyplot as plt

from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.propagation.oscillator import Oscillator
from nu_waves.hamiltonian import vacuum
from Support.ExampleSupport import adiabaticMassFractions, initialMassComposition, loadSolarProfile, solarHamiltonians
from nu_waves.matter.solar import SolarProfile


# 3 flavors PMNS, PDG values (2025)
angles = {(1, 2): np.deg2rad(33.4), (1, 3): np.deg2rad(8.6), (2, 3): np.deg2rad(49)}
phases = {(1, 3): np.deg2rad(195)}

# Masses, normal ordering
spec = Spectrum(n_neutrinos=3, dm2={(2, 1): 7.42e-5, (3, 2): 0.0024428})

osc = Oscillator(vacuum.Hamiltonian(Mixing(n_neutrinos=3, mixing_angles=angles, dirac_phases=phases), spec, False))


sol = SolarProfile()

E_GeV = 0.010  # 10 MeV

# neutrino start
r_emit = 0.01 * sol.R_sun_km
s = np.concatenate(([0.0], np.geomspace(1.0, sol.R_sun_km - r_emit, sampleCount(799))))  # km (include 0)

F0_vac = initialMassComposition(osc, alpha=0, basis="vacuum")
print(f"F0 vacuum: {F0_vac}")
F0_m   = initialMassComposition(osc, alpha=0, basis="matter", E_GeV=0.010, profile=sol, r_emit_km=r_emit)
print(f"F0 m: {F0_m}")
F0_vfm = initialMassComposition(osc, alpha=0, basis="vacuum_from_matter", E_GeV=0.010, profile=sol, r_emit_km=r_emit)
print(f"F0 vfm: {F0_vfm}")

F = adiabaticMassFractions(osc,
    E_GeV=E_GeV,
    profile=sol,
    r_emit_km=r_emit,
    s_km=s,
    alpha=0
)

# plot (log radius or symlog distance—your pick)
x = s / 1e5
import matplotlib.pyplot as plt
cols = ["C0","C1","C2"]; labs = [r"$\nu_1$", r"$\nu_2$", r"$\nu_3$"]
plt.figure(figsize=(7.5,4.2))
for i,(c,lab) in enumerate(zip(cols,labs)):
    plt.plot(x, F[:, i], lw=2, color=c, label=lab)
    # plt.scatter([x[0]], [F[0, i]], color=c, s=25)  # initial (vacuum-basis at r_emit)
    # plt.scatter([x[0]], [F0_vac[i]], color=c, s=25)  # initial (vacuum-basis at r_emit)
    # plt.scatter([x[0]], [F0_m[i]], color=c, s=25)  # initial (vacuum-basis at r_emit)
    plt.scatter([x[0]], [F0_vfm[i]], color=c, s=25)  # initial (vacuum-basis at r_emit)
plt.xlabel(r"Distance from emission [$10^5$ km]"); plt.ylabel("Mass–state fraction")
plt.ylim(0,1); plt.legend(frameon=False); plt.grid(True, which="both", alpha=0.3)
plt.show()

print("Initial (matter-basis) weights:", F0_vfm)
print("Initial (vacuum-basis)  weights:", F0_vac)

