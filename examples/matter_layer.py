from Support.ExampleSupport import figureDirectory, dataDirectory, sampleCount
import numpy as np
import matplotlib.pyplot as plt
from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.propagation.oscillator import Oscillator
from nu_waves.hamiltonian import matter, vacuum
from nu_waves.hamiltonian.matter import MatterProfile
import nu_waves.utils.flavors as flavors

# 3 flavors PMNS, PDG values (2025)
angles = {(1, 2): np.deg2rad(33.4), (1, 3): np.deg2rad(8.6), (2, 3): np.deg2rad(49)}
phases = {(1, 3): np.deg2rad(195)}
pmns = Mixing(n_neutrinos=3, mixing_angles=angles, dirac_phases=phases)
U_pmns = pmns.build_mixing_matrix()

# Masses, normal ordering
spec = Spectrum(n_neutrinos=3, dm2={(2, 1): 7.42e-5, (3, 2): 0.0024428})

# oscillator
osc = Oscillator(vacuum.Hamiltonian(pmns, spec, False))
hMatter = matter.Hamiltonian(pmns, spec, False)

osc.hamiltonian = vacuum.Hamiltonian(pmns, spec, False)
P_vac = osc.probability(L_km=295, E_GeV=np.linspace(0.3,3,sampleCount(200)), flavor_emit=1, flavor_det=0)
hMatter.set_matter_profile(MatterProfile.from_fractions([0.0],[0.5],[1.0]))
osc.hamiltonian = hMatter
P_chk = osc.probability(L_km=295, E_GeV=np.linspace(0.3,3,sampleCount(200)), flavor_emit=1, flavor_det=0)
print("Checking null density")
np.testing.assert_allclose(P_vac, P_chk, atol=1e-12)

prof_AB = MatterProfile.from_fractions([2.8, 11.0], [0.5, 0.467], [0.7, 0.3])
hMatter.set_matter_profile(prof_AB)
P_full = osc.probability(L_km=1000, E_GeV=np.linspace(0.3,5,sampleCount(120)), flavor_emit=None, flavor_det=None)
print("Checking Unitarity")
np.testing.assert_allclose(P_full.sum(axis=-2), 1.0, atol=2e-6)

prof_BA = MatterProfile.from_fractions([11.0, 2.8], [0.467, 0.5], [0.3, 0.7])
hMatter.set_matter_profile(prof_AB); P_AB = osc.probability(L_km=1000, E_GeV=np.linspace(0.3,5,sampleCount(200)), flavor_emit=1, flavor_det=0)
hMatter.set_matter_profile(prof_BA); P_BA = osc.probability(L_km=1000, E_GeV=np.linspace(0.3,5,sampleCount(200)), flavor_emit=1, flavor_det=0)
print("Checking ordering of layers -> should have an effect")
print(np.max(np.abs(P_AB - P_BA)))
assert np.max(np.abs(P_AB - P_BA)) > 1e-3


print("One layer should correspond to constant matter density")

# DUNE-like config
L_km = 1300.0
E = np.linspace(0.3, 5.0, sampleCount(250))
rho, Ye = 2.8, 0.5

# --- constant-density path ---
hMatter.set_matter_profile(None)
hMatter.set_constant_density(rho_in_g_per_cm3=rho, Ye=Ye)
osc.hamiltonian = hMatter
P_cd = osc.probability(L_km=L_km, E_GeV=E, flavor_emit=None, flavor_det=None)
osc.hamiltonian = vacuum.Hamiltonian(pmns, spec, False)

# --- single-layer profile (identical physical parameters) ---
profile = MatterProfile.from_fractions([rho], [Ye], [1.0])
hMatter.set_matter_profile(profile)
osc.hamiltonian = hMatter
P_1l = osc.probability(L_km=L_km, E_GeV=E, flavor_emit=None, flavor_det=None)

# --- compare ---
np.testing.assert_allclose(P_cd, P_1l, atol=1e-10)
print("One-layer profile reproduces constant-density result")

