# Releasing the pool must give the reservation back to the device so the next
# allocation reuses it. On this CANN a lone cuMemUnmap left every chunk after the
# first one mapped, halMemAddressFree refused the range with "ACL runtime error",
# and the address grew with every round: vrambuf_destroy now unmaps chunk by
# chunk, which is what this test locks down.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import comfy_aimdo.control as aimdo

assert aimdo.init("npu"), "aimdo NPU backend did not initialize"
assert aimdo.backend == "npu", "backend not recorded: %r" % (aimdo.backend,)
aimdo.set_log_warning()

import torch          # noqa: E402
import torch_npu      # noqa: E402
import comfy_aimdo.torch as aimdo_torch    # noqa: E402

MIB = 1 << 20
ROUNDS = 3
assert aimdo.init_device(0)
torch.npu.memory.change_current_allocator(aimdo_torch.get_torch_allocator())

addresses = []
for round_index in range(ROUNDS):
    block = torch.full((1024 * MIB,), round_index + 1, dtype=torch.uint8, device="npu")
    addresses.append(block.data_ptr())
    torch.npu.synchronize()
    assert int(block[0].item()) == round_index + 1 and int(block[-1].item()) == round_index + 1

    del block
    torch.npu.synchronize()
    torch.npu.empty_cache()
    released = aimdo.release_pool()
    assert released >= 1024 * MIB, "round %d released only %d bytes" % (round_index, released)

torch.npu.synchronize()
assert len(set(addresses)) == 1, "the reservation was not reused: %s" % [hex(a) for a in addresses]

aimdo.deinit()
print("VA release NPU test passed (address %#x reused %d times)" % (addresses[0], ROUNDS))
