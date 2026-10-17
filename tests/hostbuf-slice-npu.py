# HostBuffer.read_file_slice with an explicit destination offset, and the
# register/unregister pair that makes a slice reachable from the device.
#
# The constructor registers whatever it extends, so this buffer commits with
# register=False: registering a range twice is refused by this CANN with
# "host memory range already registered". The other host buffer test covers
# whole-buffer reads and truncation.
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import comfy_aimdo.control as aimdo

assert aimdo.init("npu"), "aimdo NPU backend did not initialize"
assert aimdo.backend == "npu", "backend not recorded: %r" % (aimdo.backend,)
aimdo.set_log_warning()

import torch          # noqa: E402
import torch_npu      # noqa: E402
import comfy_aimdo.torch as aimdo_torch    # noqa: E402
from comfy_aimdo.host_buffer import HostBuffer    # noqa: E402

MIB = 1 << 20
assert aimdo.init_device(0)

SIZE = 8 * MIB
SLICE = 2 * MIB
OFFSET = 4 * MIB
pattern = bytes(range(256)) * (SLICE // 256)

buffer = HostBuffer(0, max_grow_size=SIZE + 2 * MIB)
assert buffer.size == 0, buffer.size
buffer.extend(SIZE, register=False)
assert buffer.get_raw_address() != 0
assert buffer.size == SIZE, buffer.size

with tempfile.NamedTemporaryFile(suffix=".bin") as handle:
    # A file whose first SLICE bytes are zero and whose next bytes are the
    # pattern, so a wrong file offset or destination offset cannot pass.
    handle.write(b"\x00" * SLICE)
    handle.write(pattern)
    handle.flush()

    # Read SLICE bytes from file offset SLICE into buffer offset OFFSET.
    buffer.read_file_slice(handle, SLICE, SLICE, offset=OFFSET)
    view = aimdo_torch.hostbuf_to_tensor(buffer)
    got = bytes(view[OFFSET:OFFSET + SLICE].cpu().numpy().tobytes())
    assert got == pattern, "read_file_slice wrote the wrong bytes at offset %d" % OFFSET
    assert bytes(view[0:16].cpu().numpy().tobytes()) == b"\x00" * 16

    # The slice is reachable from the device once registered, and the data is
    # unchanged by the round trip.
    buffer.register(OFFSET, SLICE)
    got = bytes(view[OFFSET:OFFSET + SLICE].cpu().numpy().tobytes())
    assert got == pattern, "registering the slice changed its bytes"
    buffer.unregister(OFFSET)

# Growing keeps the bytes already there.
before = buffer.size
assert buffer.extend(2 * MIB) > 0
assert buffer.size == before + 2 * MIB, (buffer.size, before)
view = aimdo_torch.hostbuf_to_tensor(buffer)
got = bytes(view[OFFSET:OFFSET + SLICE].cpu().numpy().tobytes())
assert got == pattern, "extend lost the slice"

del view, buffer
aimdo.deinit()
print("Host buffer slice NPU test passed")
