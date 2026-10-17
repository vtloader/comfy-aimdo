# A VBAR larger than the free budget must offload slices and still return
# correct data through the fallback path. The budget is squeezed with
# simple_vram_headroom so the run stays short on a large device.
import gc
import sys

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))

import comfy_aimdo.control as aimdo

assert aimdo.init("npu"), "aimdo NPU backend did not initialize"
assert aimdo.backend == "npu", "backend not recorded: %r" % (aimdo.backend,)
aimdo.set_log_warning()

import torch          # noqa: E402
import torch_npu      # noqa: E402
import comfy_aimdo.torch as aimdo_torch    # noqa: E402
from comfy_aimdo.model_vbar import ModelVBAR, vbar_fault, vbar_signature_compare, vbar_unpin  # noqa: E402

M = 1024 * 1024
GIB = 1 << 30
device_index = torch.npu.current_device()
assert aimdo.init_device(device_index)
device = torch.device("npu", device_index)

capacity = torch.npu.get_device_properties(device_index).total_memory
LAYER = 128 * M
LAYERS = 24
ITERATIONS = 3

vbar = ModelVBAR(LAYER * LAYERS + 256 * M, device_index)
allocs = [vbar.alloc(LAYER) for _ in range(LAYERS)]
source = torch.ones(LAYER // 2, dtype=torch.bfloat16)          # LAYER bytes of ones
x = torch.zeros(LAYER // 2, dtype=torch.bfloat16, device=device)

# Keep only a fraction of the weights resident, so most faults must fall back.
aimdo.set_simple_vram_headroom(capacity - 2 * GIB)
print("device %.1f GiB | %d layers x %.2f GiB | headroom leaves ~2 GiB" % (
    capacity / GIB, LAYERS, LAYER / GIB))

signatures = {}
faulted = offloaded = 0
for iteration in range(ITERATIONS):
    for index, alloc in enumerate(allocs):
        signature = vbar_fault(alloc)
        if signature is None:
            offloaded += 1
            weight = source.to(device)
        else:
            weight = aimdo_torch.aimdo_to_tensor(alloc, device).view(torch.bfloat16)
            if not vbar_signature_compare(signature, signatures.get(index)):
                weight.copy_(source)
                signatures[index] = signature
            faulted += 1
        x = x + weight
        if signature is not None:
            vbar_unpin(alloc)
        del weight
    torch.npu.synchronize()
    print("iteration %d: faulted %d, offloaded %d, x min/max %.1f/%.1f" % (
        iteration, faulted, offloaded, x.min().item(), x.max().item()))

assert offloaded > 0, "the headroom did not force any offload"
expected = float(ITERATIONS * LAYERS)
assert x.min().item() == expected and x.max().item() == expected, (
    "offloaded layers returned wrong data: %r/%r" % (x.min().item(), x.max().item()))
assert vbar.loaded_size() <= capacity, "VBAR kept more resident than the device holds"

del x, source, allocs, vbar
gc.collect()
aimdo.set_simple_vram_headroom(256 * M)
aimdo.deinit()
print("vbar-npu test passed")
