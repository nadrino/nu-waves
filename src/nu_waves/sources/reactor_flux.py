import numpy as np
from dataclasses import dataclass

@dataclass
class FissionFractions:
    """Fractional contributions of fission isotopes in reactor fuel."""
    f235: float = 0.55
    f239: float = 0.30
    f238: float = 0.07
    f241: float = 0.08


class ReactorSpectrum:
    """
    Build the emitted anti-νe spectrum from reactor fission isotopes.
    Default uses smooth approximations to Huber–Mueller parameterizations.
    """

    def __init__(self, fission_fractions: FissionFractions | None = None):
        self.ff = fission_fractions or FissionFractions()

    # Optional polynomial coefficients for actual Huber–Mueller if you want to replace later
    # Each list are coefficients a0..a5 for exp(sum a_k E^k)
    HUBER_COEFFS = {
        "U235":  [4.367, -4.577, 2.100, -0.5294, 0.06186, -0.002777],
        "Pu239": [4.757, -5.392, 2.563, -0.6596, 0.0782,  -0.003536],
        "U238":  [2.990, -2.882, 1.278, -0.3343, 0.03905, -0.001754],
        "Pu241": [4.833, -5.392, 2.569, -0.6596, 0.0782,  -0.003536],
    }

    @staticmethod
    def _huber_flux(E, coeffs):
        """Return ν̄ / MeV / fission for one isotope."""
        poly = sum(a * E**k for k, a in enumerate(coeffs))
        return np.exp(poly)

    def build_flux(self, E_MeV, use_huber=True):
        """
        Parameters
        ----------
        E_MeV : array_like
            Neutrino energy in MeV.
        use_huber : bool
            If True use Huber–Mueller polynomials, else use a smooth fallback.

        Returns
        -------
        flux : ndarray
            Emitted flux shape dN/dEν (arbitrary normalization).
        """
        ff = self.ff
        if use_huber:
            S235 = self._huber_flux(E_MeV, self.HUBER_COEFFS["U235"])
            S239 = self._huber_flux(E_MeV, self.HUBER_COEFFS["Pu239"])
            S238 = self._huber_flux(E_MeV, self.HUBER_COEFFS["U238"])
            S241 = self._huber_flux(E_MeV, self.HUBER_COEFFS["Pu241"])
        else:
            # smoother but less physical fallback
            def smooth(E, c0, c1, c2, cutoff):
                return c0 * np.exp(-c1*E) * (1 + c2*E) * (E < cutoff)
            S235 = smooth(E_MeV, 5.8, 0.60, 0.12, 8.0)
            S239 = smooth(E_MeV, 4.4, 0.65, 0.10, 7.6)
            S238 = smooth(E_MeV, 1.6, 0.50, 0.20, 8.5)
            S241 = smooth(E_MeV, 2.7, 0.62, 0.11, 7.8)

        flux = (ff.f235 * S235 +
                ff.f239 * S239 +
                ff.f238 * S238 +
                ff.f241 * S241)
        return flux / np.trapezoid(flux, E_MeV)  # normalize to unit area
