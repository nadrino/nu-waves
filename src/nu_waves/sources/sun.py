from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Callable, Optional, Dict
import numpy as np
integrateTrapezoid = getattr(np, 'trapezoid', None) or np.trapz

from nu_waves.utils.units import R_SUN


class SolarSource(Enum):
    PP   = "pp"
    BE7  = "Be7"
    B8   = "B8"
    PEP  = "pep"
    HEP  = "hep"
    N13  = "N13"
    O15  = "O15"
    F17  = "F17"

@dataclass
class SourceSpec:
    name: SolarSource
    # Energy spectrum pdf(E) in MeV^-1; returns array matching E
    spectrum_pdf: Callable[[np.ndarray], np.ndarray]
    # Support [Emin, Emax] in MeV (used for grids/sampling)
    e_range: Tuple[float, float]
    # Radial pdf over r in [0, R_SUN] in km; returns array matching r
    radial_pdf: Callable[[np.ndarray], np.ndarray]

@dataclass
class SolarModel:
    """Container for production PDFs and handy samplers."""
    sources: Dict[SolarSource, SourceSpec]

    @staticmethod
    def standard_model() -> "SolarModel":
        """
        Lightweight parametric approximations; quick to run and easy to
        swap for tabulated SSM later.
        """
        # --- Energy spectra (normalized internally)
        def _normalize(x, pdf):
            y = pdf(x)
            area = integrateTrapezoid(y, x)
            return lambda z: pdf(z) / area if area > 0 else lambda z: np.zeros_like(z)

        # pp: ~phase-space-like up to 0.42 MeV
        E_pp_min, E_pp_max = 0.0, 0.42
        _pp_raw = lambda E: np.clip((E_pp_max - E), 0, None) * np.clip(E, 0, None)
        pp_pdf  = _normalize(np.linspace(E_pp_min, E_pp_max, 1001), _pp_raw)

        # pep line ~1.44 MeV (narrow)
        E_pep_min, E_pep_max = 1.3, 1.6
        pep_mu, pep_sig = 1.44, 0.01
        _pep_raw = lambda E: np.exp(-0.5*((E - pep_mu)/pep_sig)**2)
        pep_pdf  = _normalize(np.linspace(E_pep_min, E_pep_max, 800), _pep_raw)

        # Be7 lines at 0.862 (dominant) & 0.384 MeV; approximate as two narrow Gaussians
        E_be7_min, E_be7_max = 0.2, 1.1
        be7_lines = [(0.862, 0.008, 0.9), (0.384, 0.008, 0.1)]
        _be7_raw = lambda E: sum(w*np.exp(-0.5*((E-m)/0.008)**2) for (m,_,w) in be7_lines)
        be7_pdf  = _normalize(np.linspace(E_be7_min, E_be7_max, 1000), _be7_raw)

        # B8: broad up to ~16 MeV; simple beta-like shape E^2 (Emax-E)^2
        E_b8_min, E_b8_max = 1.0, 16.0
        _b8_raw = lambda E: np.clip(E, 0, None)**2 * np.clip(E_b8_max - E, 0, None)**2
        b8_pdf  = _normalize(np.linspace(E_b8_min, E_b8_max, 2000), _b8_raw)

        # hep: up to ~18.8 MeV, very small flux; similar shape
        E_hep_min, E_hep_max = 2.0, 18.8
        _hep_raw = lambda E: np.clip(E, 0, None)**2 * np.clip(E_hep_max - E, 0, None)**2
        hep_pdf  = _normalize(np.linspace(E_hep_min, E_hep_max, 2000), _hep_raw)

        # CNO (N13, O15, F17): simple beta-like shapes with different endpoints
        def cno_pdf(e_max):
            _raw = lambda E: np.clip(E, 0, None)**2 * np.clip(e_max - E, 0, None)
            return _normalize(np.linspace(0.0, e_max, 1200), _raw)

        # --- Radial PDFs (parametric; replace with tabulated SSM later)
        # Use dimensionless x = r / R_sun; Beta-like shapes peaked in the core
        def beta_core(x, a, b):
            x = np.clip(x, 0, 1)
            y = np.power(x, a-1) * np.power(1-x, b-1)
            return y

        # def make_radial(a, b):
        #     # Return pdf over r in km by normalizing over [0, R_SUN]
        #     def _raw(r):
        #         x = np.clip(r / R_SUN, 0, 1)
        #         return beta_core(x, a, b)
        #     # normalize numerically
        #     rr = np.linspace(0, R_SUN, 2000)
        #     norm = integrateTrapezoid(_raw(rr), rr)
        #     return lambda r: _raw(r) / norm if norm > 0 else (lambda r: 0*r)
        #
        # # Rough shapes (tunable):
        # # pp is broader, Be7/B8 more central.
        # radial_pp  = make_radial(a=2.5, b=6.0)   # peaks ~0.15–0.2 R☉
        # radial_pep = make_radial(a=3.2, b=7.0)
        # radial_be7 = make_radial(a=3.8, b=8.0)   # more central
        # radial_b8  = make_radial(a=4.2, b=9.0)   # very central
        # radial_hep = make_radial(a=3.5, b=7.5)
        # radial_cno = make_radial(a=3.0, b=7.0)

        def make_beta_radial_with_peak(x_peak: float, concentration: float):
            """
            Beta(a,b) over x=r/Rsun with a,b>1 and peak at x_peak.
            'concentration' controls sharpness (larger -> narrower).
            """
            x_peak = float(np.clip(x_peak, 1e-4, 1 - 1e-4))
            # total pseudo-counts t = a + b, with t >= 4 to keep both >1
            t = max(concentration, 4.0)
            a = 1.0 + x_peak * (t - 2.0)
            b = t - a

            def _raw_r(r_km):
                x = np.clip(r_km / R_SUN, 0, 1)
                y = np.power(x, a - 1.0) * np.power(1.0 - x, b - 1.0)
                return y

            rr = np.linspace(0, R_SUN, 3000)
            norm = integrateTrapezoid(_raw_r(rr), rr)
            return (lambda r_km: _raw_r(r_km) / norm if norm > 0 else (lambda r_km: 0 * r_km))

        # Peaks informed by SSM intuition: B8 most central, Be7 central, pp broadest.
        # concentration values set relative widths (tighter for B8).
        radial_pp = make_beta_radial_with_peak(x_peak=0.20, concentration=10.0)  # broad
        radial_pep = make_beta_radial_with_peak(x_peak=0.12, concentration=14.0)
        radial_be7 = make_beta_radial_with_peak(x_peak=0.08, concentration=18.0)  # central
        radial_b8 = make_beta_radial_with_peak(x_peak=0.05, concentration=30.0)  # very central
        radial_hep = make_beta_radial_with_peak(x_peak=0.07, concentration=16.0)
        radial_cno = make_beta_radial_with_peak(x_peak=0.10, concentration=14.0)

        sources = {
            SolarSource.PP:  SourceSpec(SolarSource.PP,  pp_pdf,  (E_pp_min,  E_pp_max),  radial_pp),
            SolarSource.PEP: SourceSpec(SolarSource.PEP, pep_pdf, (E_pep_min, E_pep_max), radial_pep),
            SolarSource.BE7: SourceSpec(SolarSource.BE7, be7_pdf, (E_be7_min, E_be7_max), radial_be7),
            SolarSource.B8:  SourceSpec(SolarSource.B8,  b8_pdf,  (E_b8_min,  E_b8_max),  radial_b8),
            SolarSource.HEP: SourceSpec(SolarSource.HEP, hep_pdf, (E_hep_min, E_hep_max), radial_hep),
            SolarSource.N13: SourceSpec(SolarSource.N13, cno_pdf(1.20), (0.0, 1.20), radial_cno),
            SolarSource.O15: SourceSpec(SolarSource.O15, cno_pdf(1.73), (0.0, 1.73), radial_cno),
            SolarSource.F17: SourceSpec(SolarSource.F17, cno_pdf(1.74), (0.0, 1.74), radial_cno),
        }
        return SolarModel(sources)

    # --------- Public API ---------
    def radial_pdf(self, source: SolarSource, r_km: np.ndarray) -> np.ndarray:
        return self.sources[source].radial_pdf(r_km)

    def spectrum_pdf(self, source: SolarSource, E_MeV: np.ndarray) -> np.ndarray:
        return self.sources[source].spectrum_pdf(E_MeV)

    def sample(self,
               source: SolarSource,
               n: int,
               rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Draw (r_km, E_MeV, w) samples from factorized pdf: f_r(r) * f_E(E).
        Returns: r (km), E (MeV), weights (all ones here).
        """
        rng = np.random.default_rng() if rng is None else rng

        # Sample r via inverse-CDF from tabulated CDF
        rgrid = np.linspace(0, R_SUN, 5000)
        fr = self.radial_pdf(source, rgrid)
        cdf_r = np.cumsum(fr)
        cdf_r /= cdf_r[-1]
        u = rng.uniform(size=n)
        r = np.interp(u, cdf_r, rgrid)

        # Sample E similarly
        Emin, Emax = self.sources[source].e_range
        egrid = np.linspace(Emin, Emax, 6000)
        fE = self.spectrum_pdf(source, egrid)
        cdf_E = np.cumsum(fE)
        cdf_E /= cdf_E[-1]
        v = rng.uniform(size=n)
        E = np.interp(v, cdf_E, egrid)

        w = np.ones_like(E)
        return r, E, w

    def grid(self,
             source: SolarSource,
             nr: int = 400,
             nE: int = 400) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return meshgrids (r_km, E_MeV) and joint pdf f(r,E) = fr(r) * fE(E).
        """
        Emin, Emax = self.sources[source].e_range
        r = np.linspace(0, R_SUN, nr)
        E = np.linspace(Emin, Emax, nE)
        fr = self.radial_pdf(source, r)         # (nr,)
        fE = self.spectrum_pdf(source, E)       # (nE,)
        f = np.outer(fr, fE)                    # (nr, nE)
        # normalize to 1 over r,E
        norm = integrateTrapezoid(integrateTrapezoid(f, E, axis=1), r)
        f /= norm if norm > 0 else 1.0
        R, EE = np.meshgrid(r, E, indexing="ij")
        return R, EE, f

