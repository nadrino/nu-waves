"""Midpoint discretization of continuous profiles on fixed layer boundaries."""
import numpy as np
from types import MappingProxyType

from nu_waves.hamiltonian.matter import MatterProfile, MatterLayer


class ParameterizedMatterProfile(MatterProfile):
    """Sample rho(x, parameters) and Ye(x, parameters) on a fixed grid.

    Coordinates are x/L for fraction slicing, and km for absolute slicing.
    Value callbacks return scalars or arrays of shape (nLayers,). Gradient
    callbacks return dictionaries mapping parameter names to the same shapes.
    Missing keys mean zero dependence, but a requested parameter must appear
    in at least one gradient dictionary. Derivative callbacks are only called
    for gradient evaluation. Rebuild the profile when parameter values change;
    this object represents a snapshot, with fixed boundaries and values.
    """
    def __init__(self, edges, parameters, density, electronFraction,
                 densityGradient=None, electronFractionGradient=None, slicing="fraction"):
        edges = np.asarray(edges, dtype=float)
        if (edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all()
                or edges[0] != 0 or np.any(np.diff(edges) <= 0)):
            raise ValueError("Edges must be finite, strictly increasing and start at zero.")
        if slicing not in ("fraction", "absolute"):
            raise ValueError("Slicing must be 'fraction' or 'absolute'.")
        if slicing == "fraction" and not np.isclose(edges[-1], 1., rtol=0, atol=1e-14):
            raise ValueError("Fractional profile edges must end at one.")
        self.parameters = MappingProxyType(dict(parameters))
        self.midpoints = (edges[:-1] + edges[1:]) / 2
        self._densityGradient = densityGradient
        self._electronFractionGradient = electronFractionGradient
        rho = self._sample(density(self.midpoints.copy(), dict(self.parameters)))
        ye = self._sample(electronFraction(self.midpoints.copy(), dict(self.parameters)))
        if np.any(rho < 0) or np.any((ye < 0) | (ye > 1)):
            raise ValueError("Density must be nonnegative and electron fraction must be in [0, 1].")
        super().__init__([MatterLayer(float(r), float(y), float(w))
                          for r, y, w in zip(rho, ye, np.diff(edges))], slicing)

    def _sample(self, values):
        try:
            result = np.broadcast_to(np.asarray(values, dtype=float), self.midpoints.shape).copy()
        except ValueError as error:
            raise ValueError("Profile values and tangents must be scalar or match nLayers.") from error
        if not np.isfinite(result).all():
            raise ValueError("Profile values and tangents must be finite.")
        return result

    def parameterGradients(self, names):
        """Return requested (dRho, dYe) arrays without event-by-layer Jacobians."""
        if not names:
            return {}
        rho = {} if self._densityGradient is None else self._densityGradient(self.midpoints.copy(), dict(self.parameters))
        ye = {} if self._electronFractionGradient is None else self._electronFractionGradient(self.midpoints.copy(), dict(self.parameters))
        result = {}
        for name in names:
            if name not in self.parameters or (name not in rho and name not in ye):
                raise ValueError(f"No analytic profile derivative supplied for {name!r}.")
            result[name] = self._sample(rho.get(name, 0.)), self._sample(ye.get(name, 0.))
        return result
