from Support.ExampleSupport import figureDirectory, dataDirectory, sampleCount
import numpy as np
import matplotlib.pyplot as plt

from nu_waves.models.mixing import Mixing
from nu_waves.models.spectrum import Spectrum
from nu_waves.matter.supernova import CoreCollapseSN

# ---- PMNS / spectrum (NH) ----
theta12 = np.deg2rad(33.44)
theta13 = np.deg2rad(8.57)
dm21 = 7.42e-5     # eV^2
dm31 = 2.517e-3    # eV^2
E_MeV = 10.0

angles = {(1, 2): theta12, (1, 3): theta13, (2, 3): 0.0}
pmns = Mixing(n_neutrinos=3, mixing_angles=angles)
U = pmns.build_mixing_matrix()

spec = Spectrum(n_neutrinos=3, m_lightest=0.0)
spec = Spectrum(n_neutrinos=3, dm2={(2, 1): dm21, (3, 1): dm31})
m2 = spec.get_m2()

# ---- 1) Locate the H-resonance on a smooth profile (no shock) ----
base = CoreCollapseSN(shock_drop=1.0)  # S(r)=1 everywhere
rH0 = base.resonance_radius_H(dm31, theta13, E_MeV)
rL0 = base.resonance_radius_L(dm21, theta12, theta13, E_MeV)

# --- DIAGNOSTICS ---
E_eV = 1e6 * E_MeV
target_H = (dm31 * np.cos(2*theta13)) / (2.0 * E_eV)
target_L = (dm21 * np.cos(2*theta12)) / (2.0 * E_eV) / (np.cos(theta13)**2)

rH_base = base.resonance_radius_H(dm31, theta13, E_MeV)
rL_base = base.resonance_radius_L(dm21, theta12, theta13, E_MeV)

print("\n--- Baseline (no shock) ---")
print(f"target_H = {target_H:.3e} eV, target_L = {target_L:.3e} eV")
print(f"rH_base = {rH_base:.1f} km, rL_base = {rL_base:.1f} km")
print(f"Ve(rH_base) = {base.Ve(rH_base):.3e} eV   (should ≈ target_H)")
print(f"dlnNe/dr(rH_base) = {base.dlnNe_dr(rH_base):.3e} 1/km")

# Try your current non-adiabatic profile (sn_na) AFTER you construct it
def dump_profile(sn, label):
    rH = sn.resonance_radius_H(dm31, theta13, E_MeV)
    print(f"\n--- {label} ---")
    print(f"shock @ {sn.rs:.1f} km, width={sn.dw:.1f} km, drop f={sn.f}")
    print(f"rH = {rH} km")
    if rH:
        print(f"   Ve(rH) = {sn.Ve(rH):.3e} eV (target_H={target_H:.3e} eV)")
        print(f"   dlnNe/dr(rH) = {sn.dlnNe_dr(rH):.3e} 1/km")
        pref = (dm31/(2.0*E_eV)) * (np.sin(2*theta13)**2/np.cos(2*theta13))
        gamma = pref * 5.0677307e9 / abs(sn.dlnNe_dr(rH))
        Pc = np.exp(-0.5*np.pi*gamma)
        print(f"   pref={pref:.3e} eV, gamma={gamma:.3e} -> Pc_H≈{Pc:.3f}")


# ---- 2) Build two profiles with the shock centered at r_H ----
# Broad shock → almost adiabatic; Narrow shock → non-adiabatic
# sn_ad = CoreCollapseSN(shock_radius_km=rH0, shock_width_km=1500.0, shock_drop=0.6, Ye=0.5)
# sn_na = CoreCollapseSN(shock_radius_km=rH0, shock_width_km=8.0,    shock_drop=0.2, Ye=0.5)

# factor at the shock *center* for the tanh model
f_ad, f_na = 0.6, 0.2      # broad/adiabatic, sharp/non-adiabatic drops
center_fac_ad = 0.5*(1.0 + f_ad)
center_fac_na = 0.5*(1.0 + f_na)

# choose rs so that the *center* of the shock is at the H resonance
rs_ad = base._find_root_on_grid(target_H / center_fac_ad)
rs_na = base._find_root_on_grid(target_H / center_fac_na)

def tune_shock_for_target_Pc(base_profile, target_Pc, dm31, theta13, E_MeV, f=0.2, n=3.0):
    """Return (rs, width_km) so that Pc_H ≈ target_Pc at the shock center."""
    # 1) place the *center* of the shock on resonance
    E = 1e6 * E_MeV
    target_H = (dm31 * np.cos(2*theta13)) / (2.0 * E)
    center_fac = 0.5 * (1.0 + f)
    rs = base_profile._find_root_on_grid(target_H / center_fac)

    # 2) compute the slope required for the given Pc
    pref = (dm31/(2.0*E)) * (np.sin(2*theta13)**2/np.cos(2*theta13))
    gamma = max(1e-6, -2.0*np.log(max(1e-12, target_Pc))/np.pi)
    slope_target = (pref * 5.0677307e9) / gamma  # [1/km]

    # 3) infer the needed shock width (clip to avoid negatives)
    base_slope = -n / rs
    shock_term = max(1e-6, slope_target - base_slope)
    width_km = (1.0 - f) / ((1.0 + f) * shock_term)
    return float(rs), float(width_km)

sn_ad = CoreCollapseSN(shock_radius_km=rs_ad, shock_width_km=1500.0, shock_drop=f_ad, Ye=0.5)
sn_na = CoreCollapseSN(shock_radius_km=rs_na, shock_width_km=3.0,    shock_drop=f_na, Ye=0.5)

# Smooth base to locate resonance
# base = CoreCollapseSN(shock_drop=1.0)
# # Aim for, say, PH≈0.4
# rs, width = tune_shock_for_target_Pc(base, target_Pc=0.4,
#                                      dm31=dm31, theta13=theta13, E_MeV=E_MeV,
#                                      f=0.2, n=3.0)
# sn_na = CoreCollapseSN(shock_radius_km=rs, shock_width_km=width, shock_drop=0.2, Ye=0.5)

dump_profile(sn_ad, "sn_ad")
dump_profile(sn_na, "sn_na")


# ---- helper: compute Pc at H/L and mass-state fractions along radius ----
def mass_state_path(sn: CoreCollapseSN, E_MeV: float, use_lz: bool,
                    rmin=30.0, rmax=1.5e5, n=1400):
    r = np.geomspace(rmin, rmax, n)

    # Landau–Zener jump probabilities from the profile
    PcH, rH = sn.parke_Pc('H', E_MeV, dm21, dm31, theta12, theta13)
    PcL, rL = sn.parke_Pc('L', E_MeV, dm21, dm31, theta12, theta13)

    PcH_used = PcH if use_lz else 0.0
    PcL_used = PcL if use_lz else 0.0

    # smooth steps to visualize where the swaps occur
    def step(arr, r0, frac_width=0.02):
        if r0 is None: return np.zeros_like(arr)
        w = frac_width * r0
        return 0.5 * (1 + np.tanh((arr - r0) / w))

    SH = step(r, rH); SL = step(r, rL)

    # NH, ν_e at production: at surface → ν3 if adiabatic at H
    p3 = 1.0 - PcH_used * SH
    p2 = PcH_used * SH * (1.0 - PcL_used) + PcL_used * SL * (1.0 - SH)
    p1 = 1.0 - p2 - p3

    return r, (p1, p2, p3), dict(PcH=PcH, rH=rH, PcL=PcL, rL=rL)

# ---- 3) Evaluate & plot ----
rA, (p1A, p2A, p3A), infoA = mass_state_path(sn_ad, E_MeV, use_lz=False)
rN, (p1N, p2N, p3N), infoN = mass_state_path(sn_na, E_MeV, use_lz=True)

print(f"[Adiabatic-like]  Pc_H={infoA['PcH']:.3e} at r_H={infoA['rH']:.0f} km, "
      f"Pc_L={infoA['PcL']:.3e} at r_L={infoA['rL']:.0f} km")
print(f"[Non-adiabatic ]  Pc_H={infoN['PcH']:.3f} at r_H={infoN['rH']:.0f} km, "
      f"Pc_L={infoN['PcL']:.3e} at r_L={infoN['rL']:.0f} km")

fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

def panel(ax, r, p1, p2, p3, title, rH, rL):
    ax.plot(r, p1, label=r'$\nu_1$')
    ax.plot(r, p2, label=r'$\nu_2$')
    ax.plot(r, p3, label=r'$\nu_3$')
    for rr, lbl in [(rH, 'H'), (rL, 'L')]:
        if rr is not None:
            ax.axvline(rr, ls='--', lw=1, color='k', alpha=0.6)
            ax.text(rr*1.02, 0.05, lbl, rotation=90, va='bottom', ha='left', fontsize=9)
    ax.set_xscale('log'); ax.set_xlim(r[0], r[-1])
    ax.set_ylim(-0.02, 1.02)  # avoid hiding p3≈1 on the top frame
    ax.set_xlabel('r [km]'); ax.set_title(title); ax.grid(True, which='both', alpha=0.25)

panel(ax[0], rA, p1A, p2A, p3A,
      f'Adiabatic (E={E_MeV:.0f} MeV)\n$P_H=0$', infoA['rH'], infoA['rL'])
panel(ax[1], rN, p1N, p2N, p3N,
      f'Non-adiabatic with shock (E={E_MeV:.0f} MeV)\n$P_H={infoN["PcH"]:.2f}$',
      infoN['rH'], infoN['rL'])

ax[0].set_ylabel('Mass-state fraction')
ax[1].legend(loc='lower right', frameon=False)
plt.tight_layout(); plt.show()
