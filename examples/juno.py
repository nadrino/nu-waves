from Support.ExampleSupport import figureDirectory, dataDirectory, sampleCount
import numpy as np
import matplotlib.pyplot as plt
from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.hamiltonian import vacuum
from nu_waves.globals.backend import Backend
from nu_waves.propagation.oscillator import Oscillator
from nu_waves.sources.reactor_flux import ReactorSpectrum, FissionFractions
from nu_waves.interactions.ibd_cross_section import IBDCrossSection
import nu_waves.utils.flavors as flavors
import nu_waves.utils.style


resolution_at_1MeV = 0.03
n_Enu = sampleCount(int(1E3), 32)
n_samples = sampleCount(int(5E4), 32)


E = np.linspace(1.8, 9.5, n_Enu)
E_GeV = E*1e-3

# 1. Build non-oscillated emitted flux × sigma_IBD
flux_model = ReactorSpectrum(FissionFractions(0.55, 0.30, 0.07, 0.08))
ibd = IBDCrossSection()
flux = flux_model.build_flux(E)
det_flux = ibd.weighted_flux(E, flux)  # this is your PDF before oscillations

angles = {(1, 2): np.deg2rad(33.4), (1, 3): np.deg2rad(8.6), (2, 3): np.deg2rad(49)}
phases = {(1, 3): np.deg2rad(195)}
dm2 = {(2, 1): 7.42e-5, (3, 2): 0.0024428}
h = vacuum.Hamiltonian(
    mixing=Mixing(n_neutrinos=3, mixing_angles=angles, dirac_phases=phases),
    spectrum=Spectrum(n_neutrinos=3, m_lightest=0, dm2=dm2),
    antineutrino=True
)
osc = Oscillator(hamiltonian=h)

spectrum_NoOsc = det_flux

xp = Backend.xp()
def smear_energy(E_nu_true_GeV, n_samples, resolution_at_1MeV=resolution_at_1MeV, rng=None):
    """
    Sample reconstructed positron energy E_e_rec [MeV] from true neutrino energy E_nu_true [GeV].

    Parameters
    ----------
    E_nu_true_GeV : array_like
        True neutrino energies in GeV.
    n_samples : int
        Number of reconstructed samples to draw per true energy.
    resolution_at_1MeV : float
        Fractional energy resolution at 1 MeV (default = 0.03 means 3%/sqrt(E/MeV)).
    rng : np.random.Generator or None
        Optional random generator for reproducibility.

    Returns
    -------
    E_e_rec : ndarray
        Reconstructed positron energies [MeV] of shape (len(E_nu_true_GeV), n_samples).
    """
    # Convert E_nu_true from GeV to MeV
    E_nu_true = xp.asarray(E_nu_true_GeV) * 1e3  # MeV

    # Approximate visible energy: E_vis ≈ E_nu - 0.78 MeV
    # E_vis_true = np.clip(E_nu_true - 0.78, 0.0, None)

    # Energy resolution scaling with sqrt(E)
    sigma_E = resolution_at_1MeV * xp.sqrt(xp.clip(E_nu_true, 1e-6, None)) * 1.0  # MeV

    # Draw Gaussian-smeared samples
    E_nu_rec = E_nu_true[:, None] + sigma_E[:, None] * xp.asarray(
        xp.random.standard_normal((E_nu_true.shape[0], n_samples)), dtype=Backend.real_dtype()
    )

    # Clip negative reconstructed energies
    E_nu_rec = xp.clip(E_nu_rec, 0.0, None)

    return E_nu_rec * 1e-3


Pee_NO = osc.probability_sampled(
    L_km=52.5, E_GeV=E_GeV,
    flavor_emit=flavors.electron, flavor_det=flavors.electron,
    n_samples=n_samples, E_sample_fct=smear_energy,
)
spectrum_NO = det_flux * Pee_NO

# inverted ordering
osc.hamiltonian.spectrum.set_dm2({(3, 2): -dm2[(3, 2)]})
Pee_IO = osc.probability_sampled(
    L_km=52.5, E_GeV=E_GeV,
    flavor_emit=flavors.electron, flavor_det=flavors.electron,
    n_samples=n_samples, E_sample_fct=smear_energy,
)
spectrum_IO = det_flux * Pee_IO

# plot the results
plt.figure(figsize=(7,4), dpi=150)
# plt.plot(E, spectrum_NoOsc, label="No oscillations")
plt.plot(E, spectrum_NO, label="Detected spectrum (NO)")
plt.plot(E, spectrum_IO, "--", label="Inverted ordering (IO)")


plt.xlabel(f"Reconstructed neutrino energy [MeV]")
plt.ylabel("Arbitrary units")
plt.title("Reactor antineutrino spectrum at JUNO baseline")
plt.grid(True, alpha=0.3)
plt.xlim(left=1.6, right=9)
plt.legend(
    loc="best",
    title=f"Resolution at 1MeV: {resolution_at_1MeV*100:.1f}%",
    title_fontsize=13,
)
plt.tight_layout()
plt.show()




