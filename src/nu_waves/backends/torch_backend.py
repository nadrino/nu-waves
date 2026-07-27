

class TorchBackend:

    def __init__(self, device=None):
        import torch
        self.xp = torch
        self.linalg = self._make_linalg_namespace()
        self._device = device or self.xp.device("cpu")

        print("[Torch] Using device:", self._device)
        # faster?
        # self.xp.autograd.grad_mode.inference_mode(mode=False)
        pass

    # ------------------------------------------------------------------
    # Custom linear algebra namespace
    # ------------------------------------------------------------------
    def _make_linalg_namespace(self):
        class _Linalg:
            pass

        L = _Linalg()

        # fallback reference to native torch.linalg
        L._torch_linalg = self.xp.linalg

        # transparent eigh override
        def eigh(H):
            """Safe Hermitian eigendecomposition, MPS-aware and analytic for 2x2/3x3."""
            nF = H.shape[-1]
            device = getattr(H, "device", None)
            if device is not None and device.type == "mps":
                # Analytic 2x2 / 3x3
                # if nF == 2:
                #     return TorchBackend._eigh_2x2(H)
                # elif nF == 3 and False: # DISABLED
                #     return TorchBackend._eigh_3x3(H)
                # CPU fallback for larger matrices
                # print(f"[WARN] eigh not defined on device: {device}. Using CPU fallback.")
                H_cpu = H.to("cpu")
                # use double precision instead
                # H_cpu = self.xp.asarray(H_cpu, device="cpu", dtype=self.xp.complex128)
                vals, vecs = self.xp.linalg.eigh(H_cpu)
                # explicitly downcast *on CPU* before returning to MPS
                vals = vals.to(dtype=self.xp.complex64).to(H.device)
                vecs = vecs.to(dtype=self.xp.complex64).to(H.device)
                return vals.contiguous(), vecs.contiguous()
            # other devices (cuda/cpu) → native
            return self.xp.linalg.eigh(H)

        L.eigh = eigh

        return L

    @property
    def device(self):
        return self._device

    """Subset of Array-API interface mapped to torch."""
    def __getattr__(self, name):
        # delegate to torch if it exists
        if hasattr(self.xp, name):
            return getattr(self.xp, name)
        raise AttributeError(f"TorchCompat: torch has no attribute '{name}'")

    # explicit overrides for missing Array API functions
    def asarray(self, x, dtype=None):
        return self.xp.as_tensor(x, device=self.device, dtype=dtype)

    def matrix_transpose(self, M):
        """
        Backend-agnostic matrix transpose (swap last two axes).
        Preserves batch dimensions.
        """
        out = M.mT
        if M.device is not None and M.device.type == "mps":
            # force it not to be a view
            out = out.contiguous()
        return out

    def copy(self, x):
        return x.clone()

    def eye(self, n, m=None, dtype=None):
        return self.xp.eye(n, m or n,
                         device=self._device,
                         dtype=dtype or self._default_dtype)

    def empty(self, shape, dtype=None, device=None):
        if device is None:
            device = self.device
        return self.xp.empty(shape, device=device, dtype=dtype)

    def zeros(self, shape, dtype=None, device=None):
        if device is None:
            device = self.device
        return self.xp.zeros(shape, device=device, dtype=dtype)

    def conjugate(self, x):
        return self.xp.conj(x)

    def ndim(self, x):
        """Return number of dimensions of tensor."""
        return x.ndim

    def shape(self, x):
        """Return shape tuple."""
        return tuple(x.shape)

    def size(self, x, dim=None):
        """Return total number of elements or along a given dimension."""
        if dim is not None:
            return x.shape[dim]
        # flatten self.xp.Size to int
        return int(self.xp.tensor(x.numel()).item())

    def repeat(self, a, repeats, axis=None):
        """NumPy-style repeat for torch backend."""
        if not self.xp.is_tensor(a):
            a = self.xp.as_tensor(a)

        if axis is None:
            # Flatten then repeat globally
            a_flat = a.flatten()
            return a_flat.repeat_interleave(repeats)

        # Axis-specific repeat
        if isinstance(repeats, int):
            return a.repeat_interleave(repeats, dim=axis)
        else:
            # repeats is sequence of per-element counts
            if len(repeats) != a.size(axis):
                raise ValueError("len(repeats) must match dimension length")
            # Build index for repeat_interleave
            idx = self.xp.arange(a.size(axis), device=a.device).repeat_interleave(
                self.xp.as_tensor(repeats, device=a.device))
            return self.xp.index_select(a, axis, idx)

    class _Random:
        def standard_normal(self, shape):
            from nu_waves.globals.backend import Backend
            device = Backend.device() if hasattr(Backend, "device") else "cpu"
            return Backend.xp().randn(shape, device=device)

    random = _Random()
