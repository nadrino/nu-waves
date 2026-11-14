from nu_waves.globals.backend import Backend
import numpy as np


class Spectrum:
    """
    Neutrino mass spectrum container.

    Initialize with:
        Spectrum(n=3, m_lightest=0.01)

    Then inject Δm²_ij values via:
        set_dm2({(2,1): 7.4e-5, (3,1): 2.517e-3})

    Attributes
    ----------
    _m_lightest : float
        Lightest mass in eV (optional, default 0)
    _dm2_matrix : np.ndarray
        Antisymmetric matrix of Δm²_ij = m_i² − m_j²
    """

    def __init__(self, n_neutrinos: int, dm2: dict = None, m_lightest: float = 0.0):
        if n_neutrinos < 2:
            raise ValueError("Number of mass states must be ≥ 2.")
        self._m_lightest = float(m_lightest)
        self._n_neutrinos = n_neutrinos
        self._dm2_dict = dm2
        # init
        self._generate_dm2_matrix()

    @property
    def n_neutrinos(self):
        return self._n_neutrinos

    @property
    def get_m_lightest(self):
        return self._m_lightest

    def set_dm2(self, dm2: dict):
        for (i, j), val in dm2.items():
            self._dm2_dict[(i, j)] = val
        self._generate_dm2_matrix()

    # ---------- Input of Δm² ----------
    def relativeMasses(self):
        """Resolve graph-connected mass differences without reading array values."""
        xp = Backend.xp()
        n = self.n_neutrinos
        adjacency = [[] for _ in range(n)]
        seen = set()
        for (i, j), value in (self._dm2_dict or {}).items():
            if value is None:
                continue
            if not (1 <= i <= n and 1 <= j <= n) or i == j:
                raise ValueError("Invalid mass-difference indices.")
            if (j, i) in seen:
                raise ValueError("Redundant mass differences.")
            seen.add((i, j))
            context = Backend._bufferContext.get()
            if context is not None:
                if isinstance(value, np.ndarray) and context._api_name != 'numpy':
                    raise ValueError('Mass-difference arrays must share the same backend/device; found a host NumPy array.')
                if type(value).__module__.startswith('torch'):
                    if context._api_name != 'torch' or value.device != xp.device:
                        raise ValueError('Mass-difference tensors must share the same device.')
            value = xp.asarray(value, dtype=Backend.real_dtype())
            adjacency[j - 1].append((i - 1, value))
            adjacency[i - 1].append((j - 1, -value))
        if len(seen) > n - 1:
            raise ValueError("Too many independent mass differences.")
        values = {0: xp.asarray(0., dtype=Backend.real_dtype())}
        queue = [0]
        for j in queue:
            for i, value in adjacency[j]:
                if i not in values:
                    values[i] = values[j] + value
                    queue.append(i)
        if len(values) != n:
            raise ValueError("Mass differences must connect all mass states.")
        return xp.stack([values[i] for i in range(n)])

    def _generate_dm2_matrix(self):
        """
        Set Δm²_ij values, checking for consistency and redundancy.
        Example:
            spec.set_dm2({(2,1): 7.4e-5, (3,1): 2.517e-3})
        """
        # Mutable parameter views use graph topology, never scalar value reads.
        arrays = [value for value in (self._dm2_dict or {}).values()
                  if isinstance(value, np.ndarray) or type(value).__module__.startswith("torch")]
        if arrays:
            with Backend.useBuffer(arrays[0]):
                masses = self.relativeMasses()
                self._dm2_matrix = masses[:, None] - masses[None, :]
            return
        # init
        self._dm2_matrix = Backend.xp().zeros((self._n_neutrinos, self._n_neutrinos), dtype=Backend.real_dtype())
        self._dm2_matrix = Backend.to_device(self._dm2_matrix)

        # leave is not set
        if self._dm2_dict is None:
            return

        # --- Loop over entries to validate indices and redundancy ---
        seen_pairs = set()
        for (i, j), val in self._dm2_dict.items():
            # 0. Manually invalidated
            if val is None:
                continue
            # 1. Invalid or self-referencing indices
            if not (1 <= i <= self.n_neutrinos and 1 <= j <= self.n_neutrinos):
                raise IndexError(f"indices ({i},{j}) out of range for n={self.n_neutrinos}")
            if i == j:
                raise ValueError(f"Invalid Δm²({i},{j}): i and j must differ.")

            # 2. Detect redundant or antisymmetric duplicates
            if (i, j) in seen_pairs or (j, i) in seen_pairs:
                raise ValueError(f"Redundant or conflicting entry for ({i},{j}) / ({j},{i}).")

            seen_pairs.add((i, j))

        # 3. Too many independent entries (redundant information)
        if len(seen_pairs) > self.n_neutrinos - 1:
            raise ValueError(
                f"Too many Δm² entries ({len(seen_pairs)}) for n={self.n_neutrinos}. "
                f"Maximum independent differences is {self.n_neutrinos - 1}."
            )

        # --- passed validation, ready for injection below ---


        """
        Update the internal Δm² matrix with new values, ensuring global transitivity
        and antisymmetry.

        Behavior:
          - Values in dm2_dict override previous ones.
          - Missing entries are inferred from transitivity.
          - Slight inconsistencies are corrected (auto-heal).
        """
        xp = np
        # xp = Backend.xp()

        n = self.n_neutrinos
        tol = 1e-10
        self._dm2_matrix = Backend.from_device(self._dm2_matrix)
        M = xp.copy(self._dm2_matrix)

        # --- 1. Direct injection of user values ---
        for (i, j), val in self._dm2_dict.items():
            M[i - 1, j - 1] = val
            M[j - 1, i - 1] = -val

        # --- 2. Build connectivity graph ---
        adjacency = {k: set() for k in range(n)}
        for i in range(n):
            for j in range(n):
                if not xp.isnan(M[i, j]) and abs(M[i, j]) > 0:
                    adjacency[i].add(j)

        # --- 3. Choose a reference (first connected node) ---
        ref = 0
        visited = {ref}
        queue = [ref]
        delta = xp.zeros(n)
        delta[:] = xp.nan
        delta[ref] = 0.0

        # --- 4. Propagate Δ_i = Δm²_{i,ref} using BFS ---
        while queue:
            j = queue.pop(0)
            for k in adjacency[j]:
                if xp.isnan(M[k, j]):
                    continue
                new_val = delta[j] + M[k, j]
                if xp.isnan(delta[k]):
                    delta[k] = new_val
                    visited.add(k)
                    queue.append(k)
                else:
                    # check consistency
                    if abs(delta[k] - new_val) > tol:
                        delta[k] = 0.5 * (delta[k] + new_val)  # smooth correction

        if xp.any(xp.isnan(delta)):
            raise ValueError("Incomplete Δm² network: some states are disconnected.")

        # --- 5. Fill / correct all entries by transitivity ---
        for i in range(n):
            for j in range(n):
                if i == j:
                    M[i, j] = 0.0
                    continue
                expected = delta[i] - delta[j]
                if xp.isnan(M[i, j]) or abs(M[i, j] - expected) > tol:
                    M[i, j] = expected
                    M[j, i] = -expected

        # --- 6. Final antisymmetrization & store ---
        self._dm2_matrix = 0.5 * (M - M.T)
        self._dm2_matrix = Backend.xp().asarray(self._dm2_matrix)

    def get_dm2_matrix(self):
        return self._dm2_matrix

    def get_dm2(self, i: int, j: int) -> float:
        """Return Δm²_ij = m_i² − m_j² (1-based indices)."""
        return float(self._dm2_matrix[i - 1, j - 1])

    def get_m2(self):
        """
        Compute the vector of mass-squared values [m1², m2², ...]
        from the antisymmetric Δm² matrix and the lightest mass.

        The reference state (lightest) is identified automatically
        as the one with the smallest average m² offset.
        """
        xp = Backend.xp()
        M = self._dm2_matrix
        n = M.shape[0]

        # --- identify lightest state (min mean Δm²) ---
        # For a consistent antisymmetric M, the lightest has negative mean shifts
        mean_shift = xp.mean(M, axis=1)
        ref = xp.argmin(mean_shift)  # index of lightest (0-based)

        # --- reconstruct absolute m² values ---
        m2_lightest = self._m_lightest ** 2
        m2 = m2_lightest + M[:, ref]

        # --- sanity checks ---
        if Backend.xp().any(m2 < -1e-18):
            raise ValueError(f"Inconsistent Δm² matrix: negative m² values found {m2}")

        return m2

    def get_m(self):
        xp = Backend.xp()
        return xp.sqrt(self.get_m2())

    # ---------- Utilities ----------
    def summary(self):
        xp = Backend.xp()
        print(f"Spectrum with {self.n_neutrinos} mass states:")
        print(f"  Lightest mass = {self._m_lightest:.6f} eV")
        print(f"  Masses = {xp.round(self.get_m(), 6)} eV")
        print("Δm² matrix [eV²]:")
        print(xp.round(self._dm2_matrix, 6))


if __name__ == "__main__":

    print("Providing 2,1 and 3,1:")
    spec = Spectrum(n_neutrinos=3, m_lightest=0.01)
    spec.set_dm2({
        (2, 1): 7.42e-5,
        (3, 1): 2.517e-3
    })
    spec.summary()
    print("Δm²_21 =", spec.get_dm2(2, 1))
    print("Δm²_31 =", spec.get_dm2(3, 1))
    print("Δm²_32 =", spec.get_dm2(3, 2))

    # coherence tests
    print("Providing 2,1 and 3,2:")
    spec = Spectrum(n_neutrinos=3, m_lightest=0.01)
    spec.set_dm2({
        (2, 1): 7.42e-5,
        (3, 2): 0.0024428
    })
    spec.summary()
    print("Δm²_21 =", spec.get_dm2(2, 1))
    print("Δm²_31 =", spec.get_dm2(3, 1))
    print("Δm²_32 =", spec.get_dm2(3, 2))

    print("Providing 3,1 and 3,2:")
    spec = Spectrum(n_neutrinos=3, m_lightest=0.01)
    spec.set_dm2({
        (3, 1): 2.517e-3,
        (3, 2): 0.0024428
    })
    spec.summary()
    print("Δm²_21 =", spec.get_dm2(2, 1))
    print("Δm²_31 =", spec.get_dm2(3, 1))
    print("Δm²_32 =", spec.get_dm2(3, 2))

    print("Inverted ordering:")
    spec.set_dm2({
        (2, 1): 7.42e-5,
        (3, 1): -2.517e-3
    })
    spec.summary()
    print("Δm²_21 =", spec.get_dm2(2, 1))
    print("Δm²_31 =", spec.get_dm2(3, 1))
    print("Δm²_32 =", spec.get_dm2(3, 2))

    print("With sterile:")
    spec = Spectrum(n_neutrinos=4, m_lightest=0.01)
    spec.set_dm2({
        (3, 1): 2.517e-3,
        (3, 2): 0.0024428,
        (4, 1): 1,
    })
    spec.summary()

