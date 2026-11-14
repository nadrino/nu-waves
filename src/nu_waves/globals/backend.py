import numpy as np
from contextlib import contextmanager
from contextvars import ContextVar


# static class
class Backend:
    _bufferContext = ContextVar("nuWavesBufferBackend", default=None)

    @classmethod
    @contextmanager
    def useBuffer(cls, buffer):
        """Select an isolated backend from caller-owned storage for this call."""
        class BufferBackend(cls):
            pass
        if isinstance(buffer, np.ndarray):
            BufferBackend.set_api(np)
            if buffer.dtype not in (np.float32, np.float64):
                raise TypeError("Output buffers must use float32 or float64.")
            single = buffer.dtype == np.float32
        else:
            import torch
            if not isinstance(buffer, torch.Tensor):
                raise TypeError("Output buffers must be NumPy arrays or Torch tensors.")
            if buffer.dtype not in (torch.float32, torch.float64):
                raise TypeError("Output buffers must use float32 or float64.")
            BufferBackend.set_api(torch, device=buffer.device)
            single = buffer.dtype == torch.float32
        BufferBackend._real_dtype = "float32" if single else "float64"
        BufferBackend._complex_dtype = "complex64" if single else "complex128"
        token = cls._bufferContext.set(BufferBackend)
        try:
            yield
        finally:
            cls._bufferContext.reset(token)
    _current_api = np  # default
    _api_name = np.__name__
    _real_dtype = "float64"
    _complex_dtype = "complex128"
    _device = None  # e.g. "cuda", "cpu", "mps"

    @classmethod
    def set_api(cls, module, device=None):
        """Set xp backend: numpy, torch, cupy, jax.numpy, etc."""
        cls._real_dtype, cls._complex_dtype = "float64", "complex128"
        cls._device = None
        cls._api_name = module.__name__
        if cls._api_name == "torch":
            from nu_waves.backends.torch_backend import TorchBackend
            cls._current_api = TorchBackend(device=device)
            cls._device = cls._current_api.device

            if device == "mps":
                # Apple MPS backend currently limited to 32-bit
                cls._real_dtype = "float32"
                cls._complex_dtype = "complex64"
        elif cls._api_name == "jax":
            from nu_waves.backends.jax_backend import JaxBackend
            cls._current_api = JaxBackend(device=device)
            cls._device = cls._current_api.device
            # To ensure identical numerical behavior across all devices,
            # JAX defaults to float32 everywhere, even on CPU.
            cls._real_dtype = "float32"
            cls._complex_dtype = "complex64"
        else:
            cls._current_api = module

    @classmethod
    def xp(cls):
        """Return the current array API namespace."""
        cls = cls._bufferContext.get() or cls
        return cls._current_api

    @classmethod
    def real_dtype(cls):
        cls = cls._bufferContext.get() or cls
        xp = cls._current_api
        return getattr(xp, cls._real_dtype, cls._real_dtype)

    @classmethod
    def complex_dtype(cls):
        cls = cls._bufferContext.get() or cls
        xp = cls._current_api
        return getattr(xp, cls._complex_dtype, cls._complex_dtype)

    @classmethod
    def to_device(cls, arr):
        """Move an array to the configured device if supported."""
        cls = cls._bufferContext.get() or cls
        xp = cls._current_api
        if hasattr(xp, "asarray") and cls._device:
            if "torch" in xp.__name__:
                return arr.to(cls._device)
            elif "cupy" in xp.__name__:
                return xp.asarray(arr)  # CuPy arrays already live on GPU
            # JAX: handled automatically
        return arr

    @classmethod
    def from_device(cls, arr):
        """Pull array back to CPU (NumPy)."""
        cls = cls._bufferContext.get() or cls
        xp = cls._current_api
        if "torch" in xp.__name__:
            return arr.detach().resolve_conj().resolve_neg().cpu().numpy()
        elif "cupy" in xp.__name__:
            import cupy as cp
            return cp.asnumpy(arr)
        elif "jax" in xp.__name__:
            import jax.numpy as jnp
            import numpy as np
            return np.asarray(jnp.array(arr))
        return arr


# default is Numpy
Backend.set_api(np)
