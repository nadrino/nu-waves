"""Bound eigensolver batches while preserving Hermitian eigenpairs and shape."""
import importlib.util
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch is not installed")
class TestTorchEighBatch(unittest.TestCase):
    def checkBatches(self, device, shape, batchSize):
        import torch
        from nu_waves.backends.torch_backend import TorchBackend

        backend = TorchBackend(device=torch.device(device))
        torch.manual_seed(12)
        source = torch.randn(shape, dtype=torch.complex128, device=device)
        matrices = source + source.mH
        # Exercise non-contiguous batch storage too.
        matrices = matrices.transpose(0, 1) if matrices.ndim == 4 else matrices
        native = torch.linalg.eigh
        with patch.object(torch.linalg, "eigh", wraps=native) as solver:
            values, vectors = backend.eighBatched(matrices, batchSize)
        for call in solver.call_args_list:
            self.assertLessEqual(call.args[0].numel() // 9, batchSize)
        self.assertEqual(values.shape, matrices.shape[:-1])
        self.assertEqual(vectors.shape, matrices.shape)
        self.assertEqual(vectors.device, matrices.device)
        self.assertEqual(vectors.dtype, matrices.dtype)
        torch.testing.assert_close(matrices @ vectors, vectors * values.unsqueeze(-2))
        torch.testing.assert_close(vectors @ vectors.mH,
                                   torch.eye(3, dtype=matrices.dtype, device=device).expand_as(matrices))
        if device == "cpu":
            torch.testing.assert_close(values, native(matrices)[0])

    def testCpuShapes(self):
        for shape in ((3, 3), (0, 3, 3), (7, 3, 3), (2, 5, 3, 3)):
            self.checkBatches("cpu", shape, 4)

    def testCudaLargeBatch(self):
        import torch
        from nu_waves.backends.torch_backend import TorchBackend

        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        self.checkBatches("cuda", (65536, 3, 3), TorchBackend.cudaEighBatchSize)
        # Check public dispatch as well as the splitting helper.
        backend = TorchBackend(device=torch.device("cuda"))
        matrices = torch.eye(3, dtype=torch.complex128, device="cuda").expand(65536, 3, 3)
        with patch.object(backend, "eighBatched", wraps=backend.eighBatched) as split:
            values, _ = backend.linalg.eigh(matrices)
        split.assert_called_once()
        torch.testing.assert_close(values, torch.ones_like(values))
