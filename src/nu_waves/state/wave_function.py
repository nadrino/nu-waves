from nu_waves.globals.backend import Backend

from dataclasses import dataclass
from enum import Enum


class Basis(Enum):
    MASS = "mass"               # natural base for neutrino propagation
    FLAVOR = "flavor"           # natural base for interaction
    MATTER = "matter_custom"    # instantaneous eigen state in a given matter environment


@dataclass
class WaveFunction:
    current_basis: Basis
    values: any                 # complex xp.ndarray (NumPy or Torch). Shape: (nE, nF)
    eigen_vectors: any = None   # optional: eigenvector matrix defining this basis (nF, nF)

    def __post_init__(self):
        # placeholder
        pass

    @property
    def shape(self):
        return self.values.shape

    @property
    def n_flavors(self):
        return self.shape[-1]

    def copy(self):
        xp = Backend.xp()
        return WaveFunction(
            current_basis=self.current_basis,
            values=xp.copy(self.values),
            eigen_vectors=None if self.eigen_vectors is None else xp.copy(self.eigen_vectors),
        )

    def density_matrix(self):
        """Return the density matrix for every propagated state.

        ``values`` stores state vectors with the flavor index on its last
        axis.  The returned array consequently has shape
        ``(..., n_flavors, n_flavors)``.  This is deliberately a small,
        basis-agnostic primitive: higher-level correlation observables live
        in :mod:`nu_waves.correlations`.
        """
        xp = Backend.xp()
        return self.values[..., :, None] * xp.conjugate(self.values[..., None, :])

    def to_basis(self, target_basis: Basis, eigen_vectors):
        """
        Rotate the wavefunction to a new basis defined by `eigen_vectors`.

        Math convention is to left multiply, but because of the data
        structure holding the `nF` possible states on the right,
        right multiplication is more natural. Left multiplication would
        force ourselves to swap axis, which triggers a broadcasted,
        non-contiguous multiplication that remains fragile in complex64 (MPS).

        Parameters
        ----------
        target_basis : Basis
            Target basis label (for bookkeeping)
        eigen_vectors : xp.ndarray
            Shape (nF, nF). Right multiplication assumed.

        For instance, right multiplication assumes:
            FLAVOR -> MASS: U
            MASS -> FLAVOR: U_dagger
        """
        # H' = U† H U  → ψ' = U† ψ
        # (nE, nFe, nF) @ (nF, nF) -> (nE, nFe, nF)
        self.values = self.values @ eigen_vectors
        self.current_basis = target_basis
