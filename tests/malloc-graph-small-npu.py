# The recorder must capture one iteration's allocations and replay them onto the
# same addresses.
#
# Unlike the CUDA test of this name, this one installs the pluggable allocator
# first: the CUDA graph is fed by the cudaMallocAsync detour, and torch_npu has no
# async allocator backend, so the stream-carrying pluggable allocator is the only
# path a graph can match on. Without it record() refuses instead of returning an
# empty graph.
#
# The CUDA test also asserts that the caching allocator splits a freed block for a
# smaller later request; torch_npu does not split that way here, so what is
# asserted is the graph's own guarantee: recorded size and exact replay.
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import comfy_aimdo.control as aimdo

assert aimdo.init("npu"), "aimdo NPU backend did not initialize"
import torch          # noqa: E402
import torch_npu      # noqa: E402
import comfy_aimdo.torch as aimdo_torch    # noqa: E402

M = 1024 * 1024
device_index = 0
assert aimdo.init_device(device_index)
device = torch.device("npu", device_index)

# Refuses while no NPU pluggable allocator is installed: the graph would be
# empty. A stub stream keeps the device untouched before the allocator swap,
# which torch_npu only allows while its own allocator has not been used.
class _StubStream:
    class device:
        type = "npu"


try:
    aimdo.record(_StubStream())
except NotImplementedError:
    pass
else:
    raise AssertionError("record() should refuse on NPU without the pluggable allocator")

torch.npu.memory.change_current_allocator(aimdo_torch.get_torch_allocator())
torch.empty(1, device=device)

graph = aimdo.record(torch.npu.current_stream())
first = torch.empty(1 * M, dtype=torch.uint8, device=device)
second = torch.empty(2 * M, dtype=torch.uint8, device=device)
pointers = (first.data_ptr(), second.data_ptr())
del first, second
graph.pop()

assert graph.virtual_bytes == 8 * M, graph.virtual_bytes
assert graph.physical_bytes == 8 * M, graph.physical_bytes

graph.push()
first = torch.empty(1 * M, dtype=torch.uint8, device=device)
second = torch.empty(2 * M, dtype=torch.uint8, device=device)
replayed = (first.data_ptr(), second.data_ptr())
assert replayed == pointers, ([hex(p) for p in replayed], [hex(p) for p in pointers])
del first, second
graph.pop()

del graph
gc.collect()
aimdo.deinit()
print("Small NPU malloc graph test passed")
