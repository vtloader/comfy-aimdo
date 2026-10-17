# The recorder's control surface: pausing stops recording, resuming continues,
# named iteration reuses a scope, another stream can be recorded from, and abort
# drops the graph. The other NPU malloc graph test covers recording and replay.
#
# The controls only work while the graph is the active one, so they run inside
# the scope record() opened, before the pop that completes the graph.
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
assert aimdo.init_device(0)
device = torch.device("npu", 0)

torch.npu.memory.change_current_allocator(aimdo_torch.get_torch_allocator())
torch.empty(1, device=device)

graph = aimdo.record(torch.npu.current_stream())
# Everything stays alive until the end: a released range can be reused by the
# next allocation, which would hide what the graph did or did not record.
first = torch.empty(1 * M, dtype=torch.uint8, device=device)
recorded = graph.virtual_bytes
assert recorded > 0, recorded
assert 0 < graph.peak_used <= recorded, (graph.peak_used, recorded)
assert graph.physical_bytes > 0, graph.physical_bytes
assert isinstance(graph.rogue_count, int), type(graph.rogue_count)

# A paused allocation is not part of the graph.
graph.pause()
paused = torch.empty(4 * M, dtype=torch.uint8, device=device)
assert graph.virtual_bytes == recorded, (graph.virtual_bytes, recorded)
graph.resume()

# Recording continues after the resume.
second = torch.empty(16 * M, dtype=torch.uint8, device=device)
after_resume = graph.virtual_bytes
assert after_resume > recorded, (after_resume, recorded)

# Named iteration: repeating a name reuses that scope instead of nesting.
graph.iterate("layer")
third = torch.empty(1 * M, dtype=torch.uint8, device=device)
graph.iterate("layer")
fourth = torch.empty(1 * M, dtype=torch.uint8, device=device)
graph.iterate()
assert graph.virtual_bytes >= after_resume, (graph.virtual_bytes, after_resume)

# Recording from another stream is what use_stream is for; torch_npu exposes
# npu_stream, which the wrapper has to accept. The allocations themselves may be
# rogues rather than graph members, so only the round trip is asserted.
other = torch.npu.Stream()
with graph.use_stream(other):
    fifth = torch.empty(1 * M, dtype=torch.uint8, device=device)
    del fifth
    torch.npu.synchronize()

graph.pop()

# A second graph shows abort dropping one without completing it.
graph = aimdo.record(torch.npu.current_stream())
scratch = torch.empty(1 * M, dtype=torch.uint8, device=device)
graph.abort()
del first, paused, second, third, fourth, scratch, graph
gc.collect()
aimdo.deinit()
print("Malloc graph control NPU test passed")
