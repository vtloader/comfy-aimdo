# The VBAR control surface ComfyUI drives on top of residency: watermark limits,
# explicit frees, priority changes and the residency report. vbar-npu.py
# exercises faults and offload; this one exercises the knobs around them, so a
# limit that is ignored or a free that overruns it is a failure here.
import gc
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
from comfy_aimdo.model_vbar import (ModelVBAR, vbar_fault, vbar_signature_compare,
                                    vbar_unpin, vbars_analyze,
                                    vbars_reset_watermark_limits)  # noqa: E402

M = 1024 * 1024
PAGE = 32 * M           # VBAR_PAGE_SIZE in src/model-vbar.c
PAGES = 8
device_index = torch.npu.current_device()
assert aimdo.init_device(device_index)
device = torch.device("npu", device_index)

vbar = ModelVBAR(PAGE * PAGES, device_index)
allocs = [vbar.alloc(PAGE) for _ in range(PAGES)]
source = torch.ones(PAGE // 2, dtype=torch.bfloat16)
signatures = []

# Every page faulted in and written, so the watermark starts at the top.
for index, alloc in enumerate(allocs):
    signature = vbar_fault(alloc)
    assert signature is not None, "page %d could not be faulted in" % index
    aimdo_torch.aimdo_to_tensor(alloc, device).view(torch.bfloat16).copy_(source)
    signatures.append(signature)
    vbar_unpin(alloc)
torch.npu.synchronize()

assert vbar.get_nr_pages() == PAGES, vbar.get_nr_pages()
assert vbar.get_watermark() == PAGES, vbar.get_watermark()
assert vbar.loaded_size() == PAGE * PAGES, vbar.loaded_size()
assert all(page & 1 for page in vbar.get_residency()), vbar.get_residency()

# A signature is the page's fault serial: pages faulted in for the first time
# share one, and faulting a resident page again returns it unchanged.
assert vbar_signature_compare(signatures[0], signatures[1]), signatures[0][:2]
again = vbar_fault(allocs[0])
assert vbar_signature_compare(again, signatures[0]), "a resident page changed its serial"
vbar_unpin(allocs[0])

# A watermark limit bounds free_memory: asked for more than the gap between the
# watermark and the limit, it frees the gap and reports only that.
vbar.set_watermark_limit(2 * PAGE)
freed = vbar.free_memory(4 * PAGE)
assert freed == 4 * PAGE, (freed, 4 * PAGE)
assert vbar.get_watermark() == PAGES - 4, vbar.get_watermark()
assert vbar.loaded_size() == PAGE * (PAGES - 4), vbar.loaded_size()
freed = vbar.free_memory(10 * PAGE)
assert freed == 2 * PAGE, (freed, 2 * PAGE)
assert vbar.get_watermark() == 2, vbar.get_watermark()
assert vbar.loaded_size() == 2 * PAGE, vbar.loaded_size()

# Pages above the watermark are gone and the two below it are still resident and
# unpinned. A fault above the watermark is refused, not admitted; raising the
# watermark is what lets the released pages come back.
residency = vbar.get_residency()
assert residency == [1] * 2 + [0] * (PAGES - 2), residency
assert vbar_fault(allocs[PAGES - 1]) is None, "a fault above the watermark was admitted"
vbar.set_watermark(PAGE * PAGES)
for index, alloc in enumerate(allocs[2:], start=2):
    refaulted = vbar_fault(alloc)
    assert refaulted is not None, "released page did not fault back in"
    assert not vbar_signature_compare(refaulted, signatures[index]), (
        "page %d was evicted but kept its serial" % index)
    vbar_unpin(alloc)
torch.npu.synchronize()
assert vbar.loaded_size() == PAGE * PAGES, vbar.loaded_size()

# Prioritize pulls the watermark back to the top (ComfyUI does this when a model
# is used again); deprioritize only moves it in the order.
vbar.set_watermark_limit(2 * PAGE)
vbar.free_memory(6 * PAGE)
assert vbar.get_watermark() == 2, vbar.get_watermark()
vbar.prioritize()
assert vbar.get_watermark() == PAGES, vbar.get_watermark()
vbar.deprioritize()
assert vbar.get_watermark() == PAGES, vbar.get_watermark()

# Resetting the limits lets a later free reach the bottom of the range: the
# watermark falls to zero in that one call, but only the pages that are still
# resident count as freed (prioritize raised the watermark without re-faulting).
vbars_reset_watermark_limits()
freed = vbar.free_memory(PAGE * PAGES)
assert freed == 2 * PAGE, (freed, 2 * PAGE)
assert vbar.get_watermark() == 0, vbar.get_watermark()
assert vbar.loaded_size() == 0, vbar.loaded_size()
assert not any(page & 1 for page in vbar.get_residency()), vbar.get_residency()

# Signature comparison itself: equal buffers match, a missing signature does
# not, and mismatched lengths are refused.
assert vbar_signature_compare(signatures[0], signatures[0])
assert not vbar_signature_compare(signatures[0], None)
try:
    vbar_signature_compare(signatures[0], signatures[0][:-1])
    raise AssertionError("signatures of mismatched length must raise")
except ValueError:
    pass

analyzed = vbars_analyze(device_index)
assert isinstance(analyzed, int), type(analyzed)

del allocs, signatures, source, vbar
gc.collect()
aimdo.set_simple_vram_headroom(256 * M)
aimdo.deinit()
print("vbar control NPU test passed")
