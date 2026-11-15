from Support.ExampleSupport import figureDirectory, dataDirectory, sampleCount
import numpy as np
import matplotlib.pyplot as plt
from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.hamiltonian import vacuum
from nu_waves.propagation.oscillator import Oscillator
from nu_waves.globals.backend import Backend
import nu_waves.utils.flavors as flavors
import nu_waves.utils.style

# import torch
# Backend.set_api(torch, device='mps')

# import jax
# Backend.set_api(jax, device='mps')
# Backend.set_api(jax, device='cpu')

angles = {(1, 2): np.deg2rad(33.4), (1, 3): np.deg2rad(8.6), (2, 3): np.deg2rad(49)}
phases = {(1, 3): np.deg2rad(195)}
dm2 = {(2, 1): 7.42e-5, (3, 2): 0.0024428}

h = vacuum.Hamiltonian(
    mixing=Mixing(n_neutrinos=3, mixing_angles=angles, dirac_phases=phases),
    spectrum=Spectrum(n_neutrinos=3, m_lightest=0, dm2=dm2),
    antineutrino=False
)
osc = Oscillator(hamiltonian=h)

E_min, E_max = 0.2, 3.0
Enu_list = np.linspace(E_min, E_max, sampleCount(200))
P_no = osc.probability(L_km=295, E_GeV=Enu_list, flavor_emit=flavors.muon, flavor_det=flavors.electron)

h.spectrum.set_dm2({(3, 2): -0.0024428})
P_io = osc.probability(L_km=295, E_GeV=Enu_list, flavor_emit=flavors.muon, flavor_det=flavors.electron)

# Add plotting code
plt.figure(figsize=(6.5, 4.0))
plt.plot(Enu_list, P_no, label='Normal Ordering', lw=2)
plt.plot(Enu_list, P_io, label='Inverted Ordering', lw=2)
plt.xlabel(r'$E_\nu$ [GeV]')
plt.ylabel(r'$P_{\mu e}$')
plt.title(r'$\nu_\mu \rightarrow \nu_e$ Oscillation Probability')
plt.xlim(E_min, E_max)
plt.ylim(0, 0.1)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

