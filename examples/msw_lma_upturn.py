from Support.ExampleSupport import figureDirectory, dataDirectory, sampleCount
# examples/solar_daytime_upturn.py
import os, numpy as np, matplotlib.pyplot as plt

from nu_waves.sources.sun import SolarModel, SolarSource, R_SUN
from nu_waves.matter.solar import (
    radial_nodes_and_weights_from_pdf,
    pee_day_from_exit_fractions,
    pee_lowE_vacuum_avg,
    pee_highE_msw,
)
from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.propagation.oscillator import Oscillator
from nu_waves.hamiltonian import vacuum
from Support.ExampleSupport import loadSolarProfile, exitMassFractionsAveraged
from nu_waves.matter.solar import load_bs05_agsop # solar model


# --- 0) User knobs
SOURCE = SolarSource.B8
N_R_NODES = sampleCount(24, 16)         # importance-sampled production radii
N_STEPS_PER_RAY = sampleCount(10, 16)  # steps along r->surface for the adiabatic solver
E_MIN_MeV, E_MAX_MeV, N_E = 0.1, 15.0, 50   # log-spaced energies


# 1 sigma band
theta_12 = [33.68-0.70, 33.68+0.73]

angles = {(1, 2): np.deg2rad(33.68), (1, 3): np.deg2rad(8.6), (2, 3): np.deg2rad(49)}
phases = {(1, 3): np.deg2rad(195)}

spec = Spectrum(n_neutrinos=3, dm2={(2, 1): 7.42e-5, (3, 2): 0.0024428})

mixing = Mixing(n_neutrinos=3, mixing_angles=angles, dirac_phases=phases)
U = mixing.build_mixing_matrix()
osc = Oscillator(vacuum.Hamiltonian(mixing, spec, False))

sol = loadSolarProfile()

E_MeV = np.logspace(np.log10(E_MIN_MeV), np.log10(E_MAX_MeV), N_E)
E_GeV = 1e-3 * E_MeV

def main():
    # Production model and nodes
    smod = SolarModel.standard_model()
    r_nodes, w_nodes = radial_nodes_and_weights_from_pdf(smod, SOURCE, n_nodes=N_R_NODES)

    plt.figure(figsize=(6.8, 4.2))

    # Exit mass fractions p_i(E) averaged over production radius
    p_mass = exitMassFractionsAveraged(
        osc, sol, E_GeV, r_nodes, w_nodes, n_steps_per_ray=N_STEPS_PER_RAY
    )  # shape (N_E, 3)

    # --- Plots
    # 1) Mass-state exit fractions vs E
    labels = [r"$p_1$", r"$p_2$", r"$p_3$"]
    for i in range(p_mass.shape[1]):
        plt.plot(E_MeV, p_mass[:, i], label=labels[i])
    plt.xscale("log")
    plt.ylim(0, 1)
    plt.xlabel(r"$E_\nu$ [MeV]")
    plt.ylabel("Mass-state fraction at Sun exit")
    plt.title(f"{SOURCE.value}: mass-state exit fractions vs energy (day)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


    # Daytime Pee(E)
    pee_day = pee_day_from_exit_fractions(p_mass, U)

    # 2) Pee(E) with low/high-E sanity lines
    lo = pee_lowE_vacuum_avg(U)
    hi = pee_highE_msw(U, mass_index=1)

    plt.figure(figsize=(6.8, 4.2))
    plt.plot(E_MeV, pee_day, lw=2.0, label=r"$P_{ee}^{\rm day}(E)$")
    plt.axhline(lo, ls="--", alpha=0.7, label=f"low-E limit ≈ {lo:.3f}")
    plt.axhline(hi, ls="--", alpha=0.7, label=f"high-E limit ≈ {hi:.3f}")
    plt.xscale("log")
    plt.ylim(0, 1)
    plt.xlabel(r"$E_\nu$ [MeV]")
    plt.ylabel(r"$P_{ee}$ (day)")
    plt.title("Solar upturn (daytime, radius-averaged)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    # plt.savefig(os.path.join(OUTDIR, "pee_day_upturn.pdf"))
    plt.show()

    # quick console check
    print(f"Low-E limit (Σ|U_ei|^4) ≈ {lo:.3f}; High-E MSW limit (|U_e2|^2) ≈ {hi:.3f}")
    print(f"Pee_day(E=0.2 MeV) ≈ {np.interp(0.2, E_MeV, pee_day):.3f}")
    print(f"Pee_day(E=10  MeV) ≈ {np.interp(10.0, E_MeV, pee_day):.3f}")


if __name__ == "__main__":
    main()
