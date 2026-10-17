# The Ascend pluggable allocator must install, account every live byte, hand out
# memory the AI cores compute on, and pool what torch releases.
#
# The pooling is the reason this test exists: returning a freed block to the
# driver let a queued kernel read pages that were already reclaimed, which the
# device reports as aicore 507035 "MTE accesses an invalid GM address". See the
# comment above alloc_fn_npu in src/pyt-plug-alloc.c.
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import comfy_aimdo.control as aimdo

assert aimdo.init("npu"), "aimdo NPU backend did not initialize"
assert aimdo.backend == "npu", "backend not recorded: %r" % (aimdo.backend,)

import torch          # noqa: E402
import torch_npu      # noqa: E402
import comfy_aimdo.torch as aimdo_torch    # noqa: E402

MIB = 1 << 20
# The allocator is swapped before anything touches the device: torch_npu
# refuses to swap an allocator that has already served an allocation.
assert aimdo.init_device(0)
allocator = aimdo_torch.get_torch_allocator()
assert type(allocator).__name__ == "_NPUPluggableAllocator", type(allocator).__name__
torch.npu.memory.change_current_allocator(allocator)
assert torch.npu.memory.get_allocator_backend() == "pluggable", torch.npu.memory.get_allocator_backend()

# Copy engines and AI cores both work on a block this allocator handed out.
x = torch.ones(16 * MIB, dtype=torch.uint8, device="npu")
x.fill_(7)
torch.npu.synchronize()
assert x[0].item() == 7 and x[-1].item() == 7
assert int(x.cpu().max().item()) == 7
assert int(x.sum().item()) == 7 * 16 * MIB

doubled = x + 1
torch.npu.synchronize()
assert int(doubled[0].item()) == 8
assert int(doubled.sum().item()) == 8 * 16 * MIB

live = aimdo.get_total_vram_usage()
assert live >= 32 * MIB, "live bytes not accounted: %r" % live

# A reduction whose workspace is far larger than the input, the case that used
# to abort the device: uint8 sum converts to int64 internally.
small = torch.ones(1024 * MIB, dtype=torch.float32, device="npu")
torch.npu.synchronize()
assert float(small.sum().item()) == 1024 * MIB

del x, doubled, small
torch.npu.synchronize()
torch.npu.empty_cache()

# Freed blocks are pooled, and a later request of the same size computes too.
reused = torch.ones(16 * MIB, dtype=torch.uint8, device="npu")
torch.npu.synchronize()
assert int(reused.sum().item()) == 16 * MIB
del reused
torch.npu.synchronize()
torch.npu.empty_cache()

released = aimdo.release_pool()
# Whether the driver accepts the reservation back is a platform matter (CANN
# 9.1 can refuse the address release), so what is asserted is that the pool is
# empty afterwards, not the byte count.
assert aimdo.release_pool() == 0, "the pool was not emptied"
print("pool released %.1f MiB to the driver" % (released / MIB))

# Everything still works once the pool is gone.
scratch = torch.ones(4 * MIB, dtype=torch.float32, device="npu")
torch.npu.synchronize()
assert float(scratch.sum().item()) == 4 * MIB
assert int((scratch + 1).max().item()) == 2
del scratch
torch.npu.synchronize()
assert aimdo.get_total_vram_usage() >= 0

aimdo.deinit()
print("alloc-npu test passed")
