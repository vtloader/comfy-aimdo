import torch
import ctypes

import logging

from . import control

class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int32)]

class _DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]

class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]

class _DLManagedTensor(ctypes.Structure):
    pass

_DLManagedTensorDeleter = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensor))
_DLManagedTensor._fields_ = [
    ("dl_tensor", _DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", _DLManagedTensorDeleter),
]

# kDLExtDev, the device type torch_npu uses for Ascend tensors.
_DLPACK_DEVICE_NPU = 12

# kDLInt=0, kDLUInt=1, kDLFloat=2, kDLBfloat=4, kDLBool=6.
_DLPACK_DTYPES = {
    torch.uint8: (1, 8),
    torch.int8: (0, 8),
    torch.int16: (0, 16),
    torch.int32: (0, 32),
    torch.int64: (0, 64),
    torch.float16: (2, 16),
    torch.bfloat16: (4, 16),
    torch.float32: (2, 32),
    torch.float64: (2, 64),
    torch.bool: (6, 8),
}

# The deleter only drops the Python reference to the holder. The VA belongs to
# aimdo and must never be freed here.
_dlpack_holders = {}

def _dlpack_deleter(managed):
    _dlpack_holders.pop(managed.contents.manager_ctx, None)

_dlpack_deleter_ref = _DLManagedTensorDeleter(_dlpack_deleter)

# The destructor must take a raw pointer: a py_object argument would INCREF the
# capsule while it is being deallocated and retrigger deallocation forever.
_CapsuleDestructor = ctypes.CFUNCTYPE(None, ctypes.c_void_p)

_PyCapsule_New = ctypes.pythonapi.PyCapsule_New
_PyCapsule_New.restype = ctypes.py_object
_PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, _CapsuleDestructor]

_PyCapsule_GetName = ctypes.pythonapi.PyCapsule_GetName
_PyCapsule_GetName.restype = ctypes.c_char_p
_PyCapsule_GetName.argtypes = [ctypes.c_void_p]

_PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
_PyCapsule_GetPointer.restype = ctypes.c_void_p
_PyCapsule_GetPointer.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

def _capsule_release(capsule):
    # torch renames a consumed capsule to "used_dltensor"; the tensor storage
    # then owns the DLManagedTensor and calls its deleter.
    if _PyCapsule_GetName(capsule) != b"dltensor":
        return
    ptr = _PyCapsule_GetPointer(capsule, b"dltensor")
    if ptr:
        _dlpack_holders.pop(ctypes.cast(ptr, ctypes.POINTER(_DLManagedTensor)).contents.manager_ctx, None)

_capsule_release_ref = _CapsuleDestructor(_capsule_release)

class _DLPackHolder:
    def __init__(self, ptr, size, device, dtype=torch.uint8):
        code, bits = _DLPACK_DTYPES[dtype]
        itemsize = (bits + 7) // 8
        if size % itemsize:
            raise ValueError(f"size {size} is not a multiple of {dtype} element size")
        device_id = getattr(device, "index", None)
        if device_id is None:
            device_id = torch.npu.current_device()
        self._shape = (ctypes.c_int64 * 1)(size // itemsize)
        self._managed = _DLManagedTensor()
        self._managed.dl_tensor.data = ctypes.c_void_p(ptr)
        self._managed.dl_tensor.device = _DLDevice(_DLPACK_DEVICE_NPU, device_id)
        self._managed.dl_tensor.ndim = 1
        self._managed.dl_tensor.dtype = _DLDataType(code, bits, 1)
        self._managed.dl_tensor.shape = self._shape
        self._managed.dl_tensor.strides = None
        self._managed.dl_tensor.byte_offset = 0
        self._managed.manager_ctx = id(self)
        self._managed.deleter = _dlpack_deleter_ref

def get_npu_tensor_from_raw_ptr(ptr, size, device, dtype=torch.uint8):
    holder = _DLPackHolder(ptr, size, device, dtype)
    _dlpack_holders[id(holder)] = holder
    capsule = _PyCapsule_New(ctypes.cast(ctypes.pointer(holder._managed), ctypes.c_void_p), b"dltensor",
                             _capsule_release_ref)
    return torch.utils.dlpack.from_dlpack(capsule)

def get_tensor_from_raw_ptr(ptr, size, device):
    if getattr(device, "type", None) == "npu":
        return get_npu_tensor_from_raw_ptr(ptr, size, device)

    container = {
        "shape": (size,),
        "typestr": "|u1",
        "data": (ptr, False), #writable
        "version": 3,
    }

    class Holder:
        pass

    holder = Holder()
    holder.__cuda_array_interface__ = container

    return torch.as_tensor(holder, device=device)

def aimdo_to_tensor(alloc, device):
    _, ptr, size = alloc
    return get_tensor_from_raw_ptr(ptr, size, device)

def hostbuf_to_tensor(hostbuf):
    byte_view = (ctypes.c_uint8 * hostbuf.size).from_address(hostbuf.get_raw_address())
    return torch.frombuffer(byte_view, dtype=torch.uint8)

#pytorch doesnt have an API for a CUDAPluggableAllocator from an already loaded
#library. Rather than force a second load that pytorch owns, construct these
#pytorch internals outselves as sperate CDLL loads is far too risky.

class CUDAPluggableAllocator(torch.cuda.memory.CUDAPluggableAllocator):
    def __init__(self):
        alloc_fn = ctypes.cast(getattr(control.lib, "alloc_fn"), ctypes.c_void_p).value
        free_fn = ctypes.cast(getattr(control.lib, "free_fn"), ctypes.c_void_p).value
        assert alloc_fn is not None
        assert free_fn is not None
        self._allocator = torch._C._cuda_customAllocator(alloc_fn, free_fn)

def get_npu_allocator():
    """The Ascend pluggable allocator, built from the library comfy_aimdo holds.

    torch_npu's own loader dlopens the file again and takes the allocator's
    addresses from that second handle; taking them from the handle this module
    already holds keeps one copy of the allocator state.
    """
    import torch_npu

    class _NPUPluggableAllocator(torch_npu.npu.memory.NPUPluggableAllocator):
        def __init__(self):
            alloc_fn = ctypes.cast(control.lib.alloc_fn_npu, ctypes.c_void_p).value
            free_fn = ctypes.cast(control.lib.free_fn_npu, ctypes.c_void_p).value
            assert alloc_fn is not None
            assert free_fn is not None
            self._allocator = torch_npu._C._npu_customAllocator(alloc_fn, free_fn)

    control.npu_pluggable_allocator = True
    return _NPUPluggableAllocator()


def get_torch_allocator():
    #As of this writing (pytorch 2.10), pytorch MemPools + CUDAPluggableAllocator
    #considers the Mempool and pool usage context each as a hard reference to the
    #tensors completely preventing reasonable garbage collection. A read of the code
    #suggests that the assumptions of cudaGraphs completely prohibits pool cleanup
    #on VRAM pressure which ultimately makes this un-usable for our high pressure
    #allocator.
    if control.lib is None:
        return None
    if control.backend == "npu":
        return get_npu_allocator()
    logging.warning(f"WARNING: Aimdo+CUDAPluggableAllocator is experimental and unsupported.")
    return CUDAPluggableAllocator()
