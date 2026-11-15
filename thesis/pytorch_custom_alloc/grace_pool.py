"""Allocate a selected CUDA tensor in the Grace DRAM MemPool."""

import ctypes
from pathlib import Path

import torch


class GracePool:
    """Keep the pool active only around each requested tensor allocation."""

    def __init__(self, library):
        library = str(Path(library).resolve(strict=True))
        self.library = ctypes.CDLL(library)
        self.library.grace_verify.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        self.library.grace_verify.restype = ctypes.c_ssize_t
        self.library.grace_live_bytes.argtypes = ()
        self.library.grace_live_bytes.restype = ctypes.c_size_t
        self.library.is_device_pointer.argtypes = (ctypes.c_void_p,)
        self.library.is_device_pointer.restype = ctypes.c_int

        allocator = torch.cuda.memory.CUDAPluggableAllocator(
            library, "grace_alloc", "grace_free"
        )
        self.pool = torch.cuda.MemPool(allocator.allocator(), use_on_oom=False)

    def empty_on_grace(self, shape, *, dtype):
        if self.pool is None:
            raise RuntimeError("Grace pool is closed")
        with torch.cuda.use_mem_pool(self.pool, device=0):
            return torch.empty(shape, dtype=dtype, device="cuda:0")

    def verified_pages(self, tensor):
        result = self.library.grace_verify(tensor.data_ptr(), tensor.nbytes)
        if result < 0:
            raise RuntimeError("Grace page verification failed")
        return result

    def live_bytes(self):
        return self.library.grace_live_bytes()

    def is_hbm(self, tensor):
        result = self.library.is_device_pointer(tensor.data_ptr())
        if result < 0:
            raise RuntimeError("CUDA pointer classification failed")
        return bool(result)

    def close(self):
        # Call after releasing all tensors from the pool. The CUDA allocator
        # can keep inactive segments until the pool is destroyed.
        self.pool = None
