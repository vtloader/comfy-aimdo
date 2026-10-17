# One process, two Ascend devices.
#
# Both the pool and the recording graph are per device: a block reserved on one
# device cannot be handed to the other, and two devices recording at once must
# not share one remembered stream. A process that drives a single device cannot
# tell either mistake apart, so this test exists.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import comfy_aimdo.control as aimdo

assert aimdo.init("npu"), "aimdo NPU backend did not initialize"

import torch          # noqa: E402
import torch_npu      # noqa: E402
import comfy_aimdo.torch as aimdo_torch    # noqa: E402

MIB = 1 << 20

if torch.npu.device_count() < 2:
    print("only one NPU is present, skipping the multi device test")
    aimdo.deinit()
    raise SystemExit(0)

# Both devices get their state before the allocator swap, which torch_npu only
# allows while nothing has been allocated yet.
assert aimdo.init_devices([0, 1]), "aimdo could not initialize both devices"
torch.npu.memory.change_current_allocator(aimdo_torch.get_torch_allocator())

# A block released on device 0 goes into that device's pool. The pool is emptied
# first so that the reuse check below is about this block alone: any pooled block
# is a valid answer for a later request, not necessarily this one.
aimdo.release_pool()

torch.npu.set_device(0)
block = torch.full((4 * MIB,), 3, dtype=torch.uint8, device="npu:0")
device0_address = block.data_ptr()
del block
torch.npu.synchronize()
torch.npu.empty_cache()

# Device 1 asking for the same size must not be given device 0's reservation.
torch.npu.set_device(1)
other = torch.full((4 * MIB,), 9, dtype=torch.uint8, device="npu:1")
torch.npu.synchronize()
assert other.data_ptr() != device0_address, (
    "device 1 was handed a block reserved on device 0: %#x" % other.data_ptr())

# The same request on device 0 is served from its own pool.
torch.npu.set_device(0)
again = torch.full((4 * MIB,), 5, dtype=torch.uint8, device="npu:0")
torch.npu.synchronize()
assert again.data_ptr() == device0_address, (
    "device 0 did not reuse its own pooled block: %#x vs %#x" % (again.data_ptr(), device0_address))

# Live blocks on both devices stay distinct and keep their own contents.
both = [torch.full((8 * MIB,), index + 1, dtype=torch.float32, device=device)
        for index, device in enumerate(("npu:0", "npu:1"))]
torch.npu.synchronize()
addresses = [tensor.data_ptr() for tensor in both]
assert len(set(addresses)) == 2, [hex(address) for address in addresses]
assert all(float(tensor[0].item()) == index + 1 for index, tensor in enumerate(both))

# And a copy engine moves data across the two devices.
moved = both[1].to("npu:0")
torch.npu.synchronize()
assert float(moved[0].item()) == 2.0 and float(moved[-1].item()) == 2.0
assert float(again[0].item()) == 5 and float(other[0].item()) == 9

del both, moved, again, other
torch.npu.synchronize()
torch.npu.empty_cache()
print("multi device NPU test passed (released %d bytes)" % aimdo.release_pool())
aimdo.deinit()
