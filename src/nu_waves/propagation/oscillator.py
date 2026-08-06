from nu_waves.hamiltonian.base import HamiltonianBase
from nu_waves.state.wave_function import WaveFunction, Basis
from nu_waves.globals.backend import Backend
from nu_waves.utils.units import GEV_TO_EV, KM_TO_EVINV

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True, slots=True)
class NeutrinoEvent:
    L_km: float
    E_GeV: float
    flavor_emit: int
    flavor_det: int
    isAntiNu: bool | None = None


@dataclass(frozen=True, slots=True)
class NeutrinoEventBatch:
    L_km: any
    E_GeV: any
    flavor_emit: any
    flavor_det: any
    isAntiNu: any = None


@dataclass(frozen=True, slots=True)
class EventGroup:
    indices: np.ndarray
    L_km: np.ndarray
    E_GeV: np.ndarray
    flavor_emit: int
    flavor_det: int
    isAntiNu: bool


@dataclass(frozen=True, slots=True)
class CompiledEventBatch:
    n_events: int
    groups: tuple[EventGroup, ...]


def _sample_array(X, n_samples, sampling_fct):
    if sampling_fct is not None:
        nX = X.shape[0]
        X_sampled = sampling_fct(X, n_samples)  # (nX, n_samples)
        if X_sampled.shape != (nX, n_samples):
            raise ValueError(f"E_sampled must return shape {(nX, n_samples)}, got {X_sampled.shape}")
    else:
        X_sampled = Backend.xp().repeat(X[:, None], n_samples, axis=1)  # (nX, n_samples)

    # flatten to feed probability(...)
    return X_sampled.reshape(-1)  # (nX*n_samples,)


class Oscillator:

    def __init__(self, hamiltonian: HamiltonianBase, useExecutor: bool = True):
        self.hamiltonian = hamiltonian
        self.useExecutor = useExecutor
        self._executor = None
        self._executorSignature = None

    def probability(self, L_km, E_GeV=None, flavor_emit=None, flavor_det=None):
        if isinstance(L_km, CompiledEventBatch):
            if E_GeV is not None or flavor_emit is not None or flavor_det is not None:
                raise TypeError(
                    "Compiled event probability calls must only pass the CompiledEventBatch."
                )
            return self._probabilityCompiled(L_km)

        if self._is_event_arg(L_km):
            if E_GeV is not None or flavor_emit is not None or flavor_det is not None:
                raise TypeError(
                    "Event probability calls must only pass the event list or NeutrinoEventBatch."
                )
            return self._probability_events(L_km)

        if E_GeV is None:
            raise TypeError("Oscillator.probability() missing required argument: 'E_GeV'")

        return self._probability_legacy(
            L_km=L_km,
            E_GeV=E_GeV,
            flavor_emit=flavor_emit,
            flavor_det=flavor_det,
        )

    def _probability_legacy(self, L_km, E_GeV, flavor_emit=None, flavor_det=None):
        # unify array format
        L, E = self._generate_L_and_E_arrays(L_km, E_GeV)
        flavor_emit = self._format_flavor_arg(flavor_emit)
        flavor_det = self._format_flavor_arg(flavor_det)

        # convert units
        # don't use `*=` since some duplicated numbers could refer to the same memory address
        L = L * KM_TO_EVINV
        E = E * GEV_TO_EV

        # compute probabilities
        out = self._probability(L=L, E=E, flavor_emit=flavor_emit, flavor_det=flavor_det)

        # back to CPU
        return Backend.from_device(self._squeeze_array(out))

    def _probability_events(self, events):
        if not self.useExecutor:
            return self._probabilityEventsLegacy(events)
        return self._probabilityCompiled(self.compileEvents(events))

    def _probabilityEventsLegacy(self, events):
        batch = self._coerce_event_batch(events)
        xp = Backend.xp()

        L = xp.asarray(batch.L_km, dtype=Backend.real_dtype()).reshape(-1)
        E = xp.asarray(batch.E_GeV, dtype=Backend.real_dtype()).reshape(-1)
        flavor_emit = xp.asarray(batch.flavor_emit).reshape(-1)
        flavor_det = xp.asarray(batch.flavor_det).reshape(-1)
        isAntiNu = xp.asarray(self._format_event_antinu_arg(batch.isAntiNu, n_events=L.shape[0]))

        self._validate_event_arrays(
            L=Backend.from_device(L),
            E=Backend.from_device(E),
            flavor_emit=Backend.from_device(flavor_emit),
            flavor_det=Backend.from_device(flavor_det),
            isAntiNu=Backend.from_device(isAntiNu),
        )

        L = L * KM_TO_EVINV
        E = E * GEV_TO_EV

        all_flavors = list(range(int(self.hamiltonian.n_neutrinos)))
        out = xp.zeros((L.shape[0],), dtype=Backend.real_dtype())

        original_antineutrino = self.hamiltonian._antineutrino
        try:
            for antineutrino in (False, True):
                mask = isAntiNu == antineutrino
                if not bool(Backend.from_device(xp.any(mask))):
                    continue

                self.hamiltonian.set_antineutrino(antineutrino)
                probs = self._probability(
                    L=L[mask],
                    E=E[mask],
                    flavor_emit=all_flavors,
                    flavor_det=all_flavors,
                )
                event_idx = xp.asarray(range(probs.shape[0]))
                out[mask] = probs[event_idx, flavor_emit[mask], flavor_det[mask]]
        finally:
            self.hamiltonian.set_antineutrino(original_antineutrino)

        return Backend.from_device(out)

    def _probabilityCompiled(self, compiled_batch: CompiledEventBatch):
        if not self.useExecutor:
            return self._probabilityCompiledLegacy(compiled_batch)

        executor = self._getExecutor()
        return executor.probabilityCompiled(compiled_batch)

    def _getExecutor(self):
        backend = Backend.xp()
        backend_name = getattr(backend, "__name__", backend.__class__.__name__)
        backend_device = getattr(backend, "device", None)
        signature = (
            id(self.hamiltonian),
            backend_name,
            str(backend_device),
            bool(getattr(self.hamiltonian, "_matter_profile", None) is not None),
            bool(getattr(self.hamiltonian, "enableConstantMatterBatchOptimization", False)),
        )

        if self._executor is None or self._executorSignature != signature:
            self._executor = self.hamiltonian.makeExecutor(oscillator=self)
            self._executorSignature = signature

        return self._executor

    def _probabilityCompiledLegacy(self, compiled_batch: CompiledEventBatch):
        out = np.empty(compiled_batch.n_events, dtype=float)
        original_antineutrino = self.hamiltonian._antineutrino
        try:
            for group in compiled_batch.groups:
                self.hamiltonian.set_antineutrino(group.isAntiNu)
                probs = self._probability_legacy(
                    L_km=group.L_km,
                    E_GeV=group.E_GeV,
                    flavor_emit=group.flavor_emit,
                    flavor_det=group.flavor_det,
                )
                out[group.indices] = np.asarray(probs, dtype=float).reshape(-1)
        finally:
            self.hamiltonian.set_antineutrino(original_antineutrino)
        return out

    def compileEvents(self, events) -> CompiledEventBatch:
        if isinstance(events, CompiledEventBatch):
            return events

        batch = self._coerce_event_batch(events)

        L = np.asarray(batch.L_km, dtype=float).reshape(-1)
        E = np.asarray(batch.E_GeV, dtype=float).reshape(-1)
        flavor_emit = np.asarray(batch.flavor_emit).reshape(-1)
        flavor_det = np.asarray(batch.flavor_det).reshape(-1)
        isAntiNu = self._format_event_antinu_arg(batch.isAntiNu, n_events=L.shape[0])

        self._validate_event_arrays(L=L, E=E, flavor_emit=flavor_emit, flavor_det=flavor_det, isAntiNu=isAntiNu)

        group_map = {}
        for index in range(L.shape[0]):
            key = (bool(isAntiNu[index]), int(flavor_emit[index]), int(flavor_det[index]))
            group = group_map.setdefault(key, {"indices": [], "L_km": [], "E_GeV": []})
            group["indices"].append(index)
            group["L_km"].append(L[index])
            group["E_GeV"].append(E[index])

        groups = []
        for (group_is_antinu, group_flavor_emit, group_flavor_det), values in group_map.items():
            groups.append(EventGroup(
                indices=np.asarray(values["indices"], dtype=int),
                L_km=np.asarray(values["L_km"], dtype=float),
                E_GeV=np.asarray(values["E_GeV"], dtype=float),
                flavor_emit=group_flavor_emit,
                flavor_det=group_flavor_det,
                isAntiNu=group_is_antinu,
            ))

        groups.sort(key=lambda group: (group.isAntiNu, group.flavor_emit, group.flavor_det))
        return CompiledEventBatch(n_events=L.shape[0], groups=tuple(groups))

    def compile_events(self, events) -> CompiledEventBatch:
        return self.compileEvents(events)

    def probability_sampled(self, L_km, E_GeV, n_samples, flavor_emit=None, flavor_det=None, E_sample_fct=None, L_sample_fct=None):
        if E_sample_fct is None and L_sample_fct is None:
            raise ValueError("Must specify either E_sample_fct or L_sample_fct")

        # unify array format
        L, E = self._generate_L_and_E_arrays(L_km, E_GeV)
        flavor_emit = self._format_flavor_arg(flavor_emit)
        flavor_det = self._format_flavor_arg(flavor_det)

        # save the original number of entries
        nE = E.shape[0]

        # perform the sampling or repeat the value n_samples times
        E_sampled = _sample_array(X=E, n_samples=n_samples, sampling_fct=E_sample_fct)
        L_sampled = _sample_array(X=L, n_samples=n_samples, sampling_fct=L_sample_fct)

        # unit conversion
        L_sampled = L_sampled * KM_TO_EVINV     # (nE*n_samples,)
        E_sampled = E_sampled * GEV_TO_EV       # (nE*n_samples,)

        # compute probability
        P_sampled = self._probability(L=L_sampled, E=E_sampled, flavor_emit=flavor_emit, flavor_det=flavor_det)

        # perform the averaging over n_samples
        nFe, nFd = P_sampled.shape[-2], P_sampled.shape[-1]
        P_sampled = P_sampled.reshape(nE, n_samples, nFe, nFd)  # (nE, n_samples, nFe, nFd)
        P_sampled = Backend.xp().mean(P_sampled, axis=1)  # (nE, nFe, nFd)
        return Backend.from_device(self._squeeze_array(P_sampled))

    def generate_initial_state(self, flavor_emit, E_GeV):
        E = Backend.xp().asarray(E_GeV) * GEV_TO_EV
        return Backend.from_device(
            self._generate_initial_state(flavor_emit=flavor_emit, E=E)
        )

    def propagate_state(self, L_km, E_GeV, flavor_emit=None) -> WaveFunction:
        """Propagate flavor state vectors and return them in the flavor basis.

        This additive API intentionally follows the legacy pairwise ``L`` / ``E``
        semantics.  Unlike :meth:`probability`, it preserves complex amplitudes
        so phase-sensitive observables (density matrices and flavor-mode
        correlations) can be computed.  It does not use the probability
        executors; their optimized paths and public behaviour are unchanged.

        The returned values have shape ``(n_points, n_initial_flavors,
        n_flavors)`` and remain on the configured backend.
        """
        L, E = self._generate_L_and_E_arrays(L_km, E_GeV)
        flavor_emit = self._format_flavor_arg(flavor_emit)
        psi = self._generate_initial_state(
            flavor_emit=flavor_emit,
            E=E * GEV_TO_EV,
        )
        self.hamiltonian.propagate_state(
            psi=psi,
            L=L * KM_TO_EVINV,
            E=E * GEV_TO_EV,
        )
        return psi

    def density_matrix(self, L_km, E_GeV, flavor_emit=None, coherence_model=None):
        """Return flavor density matrices for propagated initial flavor states.

        Parameters are identical to :meth:`propagate_state`.  When supplied, a
        ``coherence_model`` must provide ``apply(rho, L_eV_inv, E_eV,
        hamiltonian)``; see :class:`GaussianWavePacket` for the standard
        vacuum wave-packet decoherence model.

        The result is copied back to NumPy, consistently with
        :meth:`probability`, and has shape ``(n_points, n_initial_flavors,
        n_flavors, n_flavors)``.
        """
        L, E = self._generate_L_and_E_arrays(L_km, E_GeV)
        psi = self.propagate_state(L_km=L, E_GeV=E, flavor_emit=flavor_emit)
        rho = psi.density_matrix()
        if coherence_model is not None:
            rho = coherence_model.apply(
                rho=rho,
                L_eV_inv=L * KM_TO_EVINV,
                E_eV=E * GEV_TO_EV,
                hamiltonian=self.hamiltonian,
            )
        return Backend.from_device(rho)

    def _probability(self, L, E, flavor_emit, flavor_det):
        psi = self._generate_initial_state(flavor_emit=flavor_emit, E=E)
        self.hamiltonian.propagate_state(psi=psi, L=L, E=E) # return the state in flavor basis

        # select the components we want
        amp = psi.values[..., flavor_det]

        prob = Backend.xp().abs(amp) ** 2
        return prob

    def _generate_initial_state(self, flavor_emit, E) -> WaveFunction:
        import numpy as np
        xp = np
        complex_type = np.complex128

        # xp = Backend.xp()
        # complex_type = Backend.complex_dtype()

        nE = E.shape[0]
        nF = self.hamiltonian.n_neutrinos
        nFe = len(flavor_emit)

        # Create zero-filled wavefunction array
        psi = xp.zeros((nE, nFe, nF), dtype=complex_type)

        # Set the emitted flavor amplitude to 1
        psi[:, xp.arange(nFe), flavor_emit] = 1.0

        psi = Backend.xp().asarray(psi, dtype=Backend.complex_dtype())

        # Create the holder
        return WaveFunction(
            current_basis=Basis.FLAVOR,
            values=psi
        )

    def _generate_L_and_E_arrays(self, L_km, E_GeV):
        # ---------- normalize inputs ----------
        xp = Backend.xp()

        L_in = xp.asarray(L_km, dtype=Backend.real_dtype())
        E_in = xp.asarray(E_GeV, dtype=Backend.real_dtype())

        if xp.ndim(L_in) == 0:
            L_in = L_in.reshape(1)
        if xp.ndim(E_in) == 0:
            E_in = E_in.reshape(1)

        # ---------- enforce pairwise semantics ----------
        if xp.size(L_in) == 1 and xp.size(E_in) > 1:
            Lc = xp.broadcast_to(L_in, E_in.shape)
            Ec = E_in
        elif xp.size(E_in) == 1 and xp.size(L_in) > 1:
            Lc = L_in
            Ec = xp.broadcast_to(E_in, L_in.shape)
        else:
            if xp.size(L_in) != xp.size(E_in):
                raise ValueError(
                    f"Length mismatch: L_km has {xp.size(L_in)}, E_GeV has {xp.size(E_in)}. "
                    "They must match for pairwise propagation."
                )
            Lc, Ec = L_in, E_in

        center_shape = Lc.shape

        # ---------- prepare flattened arrays ----------
        E_flat = Ec.reshape(-1)
        L_flat = Lc.reshape(-1)

        return L_flat, E_flat

    def _is_event_arg(self, arg):
        if isinstance(arg, NeutrinoEventBatch):
            return True
        if isinstance(arg, (list, tuple)) and len(arg) > 0:
            return all(isinstance(event, NeutrinoEvent) for event in arg)
        return False

    def _coerce_event_batch(self, events) -> NeutrinoEventBatch:
        if isinstance(events, NeutrinoEventBatch):
            return events
        if isinstance(events, (list, tuple)) and all(isinstance(event, NeutrinoEvent) for event in events):
            return NeutrinoEventBatch(
                L_km=[event.L_km for event in events],
                E_GeV=[event.E_GeV for event in events],
                flavor_emit=[event.flavor_emit for event in events],
                flavor_det=[event.flavor_det for event in events],
                isAntiNu=[event.isAntiNu for event in events],
            )
        raise TypeError("Expected a list of NeutrinoEvent or a NeutrinoEventBatch.")

    def _format_event_antinu_arg(self, isAntiNu, n_events):
        default = bool(self.hamiltonian._antineutrino)
        if isAntiNu is None:
            return np.full(int(n_events), default, dtype=bool)

        values = np.asarray(isAntiNu, dtype=object)
        if values.ndim == 0:
            values = np.full(int(n_events), bool(values.item()), dtype=bool)
        else:
            values = values.reshape(-1)
            if values.shape[0] == 1 and n_events != 1:
                values = np.full(int(n_events), bool(values[0]), dtype=bool)
            elif values.shape[0] != n_events:
                values = values.astype(bool)
            else:
                values = np.asarray([default if value is None else bool(value) for value in values], dtype=bool)

        return values

    def _validate_event_arrays(self, L, E, flavor_emit, flavor_det, isAntiNu):
        n_events = L.shape[0]
        if E.shape[0] != n_events or flavor_emit.shape[0] != n_events or flavor_det.shape[0] != n_events or isAntiNu.shape[0] != n_events:
            raise ValueError(
                "Event arrays must have matching lengths: "
                f"L_km has {L.shape[0]}, E_GeV has {E.shape[0]}, "
                f"flavor_emit has {flavor_emit.shape[0]}, flavor_det has {flavor_det.shape[0]}, "
                f"isAntiNu has {isAntiNu.shape[0]}."
            )

        if n_events == 0:
            return

        n_flavors = int(self.hamiltonian.n_neutrinos)
        if np.any((flavor_emit < 0) | (flavor_emit >= n_flavors)):
            raise ValueError(f"flavor_emit values must be in [0, {n_flavors - 1}].")
        if np.any((flavor_det < 0) | (flavor_det >= n_flavors)):
            raise ValueError(f"flavor_det values must be in [0, {n_flavors - 1}].")

    def _format_flavor_arg(self, arg):
        """
        Normalize to list[int].
        Allowed: None → full range [0..n_flavors-1], int, list[int].
        Disallowed: anything else.
        """
        xp = Backend.xp()
        if arg is None:
            return list(range(int(self.hamiltonian.n_neutrinos)))

        if isinstance(arg, int):
            out = [int(arg)]
        elif isinstance(arg, list) and all(isinstance(x, int) for x in arg):
            out = [int(x) for x in arg]
        else:
            raise TypeError(f"Flavor arg must be None, int, or list of int.")

        return out

    def _squeeze_array(self, x, preserve_axes=None):
        """
        Remove all dims of size 1.
        If preserve_axes is provided (int or iterable, supports negative indices),
        those axes are kept even if they have size 1.
        """
        # fast path when we don't need to preserve anything
        if preserve_axes is None or (isinstance(preserve_axes, (list, tuple)) and len(preserve_axes) == 0):
            # NumPy and Torch both implement .squeeze()
            return x.squeeze()

        # normalize preserve set with positive indices
        ndim = x.ndim if hasattr(x, "ndim") else x._n_neutrinos()
        if isinstance(preserve_axes, int):
            preserve_axes = (preserve_axes,)
        pres = set()
        for a in preserve_axes:
            a = int(a)
            if a < 0:
                a += ndim
            if a < 0 or a >= ndim:
                raise IndexError(f"preserve axis {a} out of range for ndim={ndim}")
            pres.add(a)

        # build new shape (keep any non-1 dims or preserved axes)
        shape = [int(s) for s in x.shape]
        new_shape = [s for i, s in enumerate(shape) if (s != 1) or (i in pres)]

        # if everything would be squeezed, return a scalar (0-d)
        if len(new_shape) == 0:
            return x.reshape(())

        # reshape is metadata-only in NumPy, and in Torch returns a view if possible
        return x.reshape(new_shape)
