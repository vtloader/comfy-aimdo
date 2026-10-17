# Host buffers and the peer cache file reader must work on the Ascend NPU:
# pinned host memory is what the engine stages weights in, and the peer cache
# reads checkpoint files straight into device memory.
#
# extend(reallocate=True) is deliberately not exercised: its size_delta
# accounting is the same Python code on every backend and is unrelated to the
# device, so it needs its own investigation rather than a device test.
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import comfy_aimdo.control as aimdo

assert aimdo.init("npu"), "aimdo NPU backend did not initialize"
import torch          # noqa: E402
import torch_npu      # noqa: E402
import comfy_aimdo.torch as aimdo_torch      # noqa: E402
from comfy_aimdo.host_buffer import HostBuffer, read_file_to_device      # noqa: E402

MIB = 1 << 20
assert aimdo.init_device(0)
device = torch.device("npu", 0)

buffer = HostBuffer(8 * MIB)
assert buffer.get_raw_address() != 0
view = aimdo_torch.hostbuf_to_tensor(buffer)
assert view.numel() == 8 * MIB, view.numel()
view.fill_(5)
assert int(view[0].item()) == 5 and int(view[-1].item()) == 5

buffer.truncate(2 * MIB)
view = aimdo_torch.hostbuf_to_tensor(buffer)
assert view.numel() == 2 * MIB, view.numel()

# Peer cache: a checkpoint file read directly into a device tensor.
payload = bytes(range(256)) * (64 * 1024)          # 16 MiB
with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as handle:
    handle.write(payload)
    path = Path(handle.name)
destination = torch.zeros(len(payload), dtype=torch.uint8, device=device)
with path.open("rb") as source:
    read_file_to_device(source, 0, len(payload), torch.npu.current_stream().npu_stream,
                        destination.data_ptr(), 0)
torch.npu.synchronize()
copied = destination.cpu()
assert copied.numel() == len(payload)
assert int(copied[0].item()) == 0 and int(copied[255].item()) == 255
assert int(copied[-1].item()) == 255

path.unlink()
del view, destination, copied
aimdo.deinit()
print("Host buffer and peer cache NPU test passed")
