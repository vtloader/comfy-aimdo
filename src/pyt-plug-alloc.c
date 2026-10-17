/* PyTorch pluggable allocator entry points.
 *
 * Two ABIs share this bookkeeping: CUDA and ROCm pass the device to both calls,
 * Ascend's torch_npu passes it only to the allocation and identifies the device
 * from the entry on free. The allocator's ABI is the only difference, so the
 * reservation/accounting path is shared.
 */
#include "plat.h"

#include <pthread.h>
#include "vrambuf.h"

#ifdef AIMDO_NPU
#include "acl-shim.h"
#endif

#define VMM_HASH_SHIFT  21

static inline unsigned int vmm_hash(CUdeviceptr ptr) {
    return ((uintptr_t)(void *)ptr >> VMM_HASH_SHIFT) % VMM_HASH_SIZE;
}

void allocations_analyze(bool only_dirty) {
    size_t total_size = 0;
    int count = 0;

    if (only_dirty && !allocations_dirty) {
        return;
    }
    allocations_dirty = false;

    log(DEBUG, "--- Allocation Analysis Start ---\n");

    for (int i = 0; i < VMM_HASH_SIZE; i++) {
        VramBuffer *entry = vmm_table[i];
        while (entry) {
            void* ptr = (void*)vrambuf_get(entry);
            size_t s = entry->allocated;

            log(DEBUG, "  [Bucket %4d] Ptr: %p | Size: %7zuk\n",
                i, ptr, s / K);

            total_size += s;
            count++;

            entry = entry->next;
        }
    }

    log(DEBUG, "%d Active Allocations for a total of %7zu MB\n", count, total_size / M);
}

/* VA-reserved buffer shared by the CUDA, ROCm and Ascend pluggable allocators. */
static void *plug_alloc(const char *name, size_t size, int device) {
    VramBuffer *entry;

    log(VERBOSE, "%s (start): size=%zuk, device=%d\n", name, size / K, device);

    if (!set_devctx_for_device(device)) {
        return NULL;
    }

    entry = vrambuf_create(device, size);
    if (!entry) {
        return NULL;
    }
    if (!vrambuf_grow(entry, size)) {
        vrambuf_destroy(entry);
        return NULL;
    }

    {
        unsigned int h = vmm_hash(vrambuf_get(entry));
        entry->next = vmm_table[h];
        vmm_table[h] = entry;
        allocations_dirty = true;
    }

    log(VERBOSE, "%s (return): ptr=%p\n", name, (void *)vrambuf_get(entry));
    return (void *)vrambuf_get(entry);
}

#ifdef AIMDO_NPU

/* Ascend / torch_npu ABI:
 *   void *alloc_fn_npu(ssize_t size, int device, aclrtStream stream);
 *   void  free_fn_npu(void *ptr, size_t size, aclrtStream stream);
 *
 * The buffer strategy is CUDA's and ROCm's: reserve VA and map physical pages
 * into it through plug_alloc, so the budget term can reclaim pages. Two measured
 * platform facts (CANN 9.1, torch_npu 2.13, Ascend910) shape the rest:
 *
 *  1. PyTorch reports that "the current CANN and Soc versions require
 *     processing for 32 padding size, with memory allocation", so every block
 *     carries 32 bytes of slack that the native allocator reserves too. Handing
 *     back exactly the requested size left kernels touching the tail, which the
 *     device reports as aicore 507035, "MTE accesses an invalid GM address".
 *  2. Returning a freed block to the driver is what this platform cannot
 *     survive: with the release in place a reduction that allocates a large
 *     workspace (uint8 sum converts to int64) aborted the AI cores with the same
 *     507035, and synchronizing the free's stream first did not help, while
 *     keeping the block made the same reduction succeed. The allocator therefore
 *     pools what torch releases and reuses it; see npu_pool_take below.
 *
 * free() carries no device in this ABI, so the entry found by pointer supplies
 * it. The allocation stream is unused: the shim allocates through the device
 * context. The free stream only matters to a recording graph.
 */
#define NPU_ALLOC_SLACK 32

/* The stream the active memory graph accepted, remembered from the allocation
 * side because torch_npu passes a different stream object when it frees. Kept
 * per device: two devices can record at the same time (ComfyUI compiles per
 * device), and one global would cross their streams. */
#define NPU_MAX_DEVICES 64
static CUstream npu_graph_stream[NPU_MAX_DEVICES];

/* torch keeps this allocator registered after control.deinit(); without this the
 * blocks it frees at interpreter exit would walk torn-down contexts. */
static bool npu_plug_closed;

void plug_teardown(void) {
    npu_plug_closed = true;
}

/* Freed blocks stay here instead of going back to the driver.
 *
 * torch_npu's caching allocator releases a block to the allocator as soon as its
 * own stream-ordered bookkeeping says the block is idle, and it reuses what it
 * gets back under the same rules. Letting the release reach the driver was the
 * one behaviour this platform could not survive: with it in place a reduction
 * that allocates a large workspace aborted the AI cores with aicore 507035,
 * while keeping the block (and reusing it) made the same reduction succeed, and
 * synchronizing the free's stream first did not help.
 *
 * So this allocator pools: a request reuses the smallest free block that covers
 * it, and otherwise reserves a new one. Pooled blocks are given back by
 * plug_release_pool(), which deinit() calls. The trade is deliberate: a pooled
 * block stays mapped and stays counted by get_total_vram_usage() until it is
 * given back, which keeps the budget term conservative rather than optimistic.
 */
typedef struct NpuPooled {
    struct NpuPooled *next;
    VramBuffer *entry;
} NpuPooled;

static NpuPooled *npu_pool;
static size_t npu_pool_bytes;
/* Own lock: the allocator callbacks can arrive from several threads, and this
 * must not depend on the accounting table's lock being alive yet. */
static pthread_mutex_t npu_pool_lock = PTHREAD_MUTEX_INITIALIZER;

/* Entries stay registered in vmm_table while pooled: the memory is still
 * reserved, so no other allocation can produce the same address. A request takes
 * the smallest block that covers it, which keeps one size class serving the
 * neighbours torch asks for. */
/* The pool node is the record that a block is already pooled. A VramBuffer field
 * cannot carry that: its handle_count belongs to the mapping handles, so a block
 * with a single handle would look pooled before it ever was. */
static bool npu_pool_contains(VramBuffer *entry) {
    bool found = false;

    pthread_mutex_lock(&npu_pool_lock);
    for (NpuPooled *node = npu_pool; node != NULL; node = node->next) {
        if (node->entry == entry) {
            found = true;
            break;
        }
    }
    pthread_mutex_unlock(&npu_pool_lock);
    return found;
}

static VramBuffer *npu_pool_take(size_t required, int device) {
    NpuPooled **slot = &npu_pool;
    NpuPooled **best_slot = NULL;
    NpuPooled *best = NULL;
    VramBuffer *entry = NULL;

    pthread_mutex_lock(&npu_pool_lock);
    while (*slot) {
        /* The device has to match: an address reserved on one device is not
         * usable by another, and this process can drive more than one. */
        if ((*slot)->entry->device == device && (*slot)->entry->allocated >= required
                && (best == NULL || (*slot)->entry->allocated < best->entry->allocated)) {
            best_slot = slot;
            best = *slot;
        }
        slot = &(*slot)->next;
    }
    if (best != NULL) {
        *best_slot = best->next;
        entry = best->entry;
        npu_pool_bytes -= entry->allocated;
        free(best);
    }
    pthread_mutex_unlock(&npu_pool_lock);
    return entry;
}

static void npu_pool_put(VramBuffer *entry) {
    NpuPooled *node = (NpuPooled *)malloc(sizeof(*node));

    if (!node) {
        /* Without metadata the block cannot be pooled; leaving it allocated is
         * still safer than returning it to the driver. */
        log(AIMDO_LOG_ERROR, "%s: out of host memory for a pooled block\n", __func__);
        return;
    }
    pthread_mutex_lock(&npu_pool_lock);
    node->entry = entry;
    node->next = npu_pool;
    npu_pool = node;
    npu_pool_bytes += entry->allocated;
    pthread_mutex_unlock(&npu_pool_lock);
}

/* Pooled entries stay in vmm_table so their address and size stay resolvable;
 * releasing one has to unlink it first, or a later free would walk freed memory.
 * The device context has to be current to reach that table. */
static void npu_unregister(VramBuffer *entry) {
    VramBuffer **slot = &vmm_table[vmm_hash(vrambuf_get(entry))];

    while (*slot && *slot != entry) {
        slot = &(*slot)->next;
    }
    if (*slot) {
        *slot = entry->next;
        allocations_dirty = true;
    }
}

/* Releases pooled blocks to the driver. Called by tools and tests that want the
 * device memory back; deinit() calls it too. vrambuf_destroy releases the entry. */
size_t plug_release_pool(void) {
    size_t released = 0;

    pthread_mutex_lock(&npu_pool_lock);
    while (npu_pool) {
        NpuPooled *node = npu_pool;
        VramBuffer *entry = node->entry;
        npu_pool = node->next;
        if (set_devctx_for_device(entry->device)) {
            /* Count what the driver took back. The reservation returns to the
             * device here: vrambuf_destroy unmaps every chunk before freeing the
             * address, which is what this CANN needs (a lone unmap left the rest
             * mapped and the free was refused). A driver that still refuses
             * leaves the range alive until the context goes away, and logs it. */
            npu_unregister(entry);
            released += entry->allocated;
            vrambuf_destroy(entry);   /* logs if the driver refuses the VA release */
        } else {
            free(entry);
        }
        free(node);
    }
    npu_pool_bytes = 0;
    pthread_mutex_unlock(&npu_pool_lock);
    return released;
}

SHARED_EXPORT
void *alloc_fn_npu(size_t size, int device, aclrtStream stream) {
    VramBuffer *entry;
    CUdeviceptr ptr = 0;

    log(VERBOSE, "%s (start): size=%zuk, device=%d\n", __func__, size / K, device);

    if (npu_plug_closed) {
        return NULL;
    }
    /* Binds the calling thread's context, and reports false both for a device
     * that is not ours and for a torn-down process after cleanup(). */
    if (!set_devctx_for_device(device)) {
        return NULL;
    }
    /* A recording graph owns the allocation while it is active, exactly as on
     * the CUDA async path (the only other place with a stream to match). It goes
     * before the pool so a replay reaches the addresses the graph recorded. */
    if (malloc_graph_alloc(&ptr, size, (CUstream)stream)) {
        if (device >= 0 && device < NPU_MAX_DEVICES) {
            npu_graph_stream[device] = (CUstream)stream;
        }
        return ptr ? (void *)ptr : NULL;
    }
    entry = npu_pool_take(size + NPU_ALLOC_SLACK, device);
    if (entry != NULL) {
        allocations_account_alloc(entry->base_ptr, size);
        log(VERBOSE, "%s (reuse): ptr=%p size=%zuk\n", __func__, (void *)entry->base_ptr, size / K);
        return (void *)entry->base_ptr;
    }
    ptr = (CUdeviceptr)(uintptr_t)plug_alloc("alloc_fn_npu", size + NPU_ALLOC_SLACK, device);
    if (ptr == 0) {
        return NULL;
    }
    allocations_account_alloc(ptr, size);

    log(VERBOSE, "%s (return): ptr=%p\n", __func__, (void *)ptr);
    return (void *)ptr;
}

/* Handles a free whose block belongs to the calling thread's current context:
 * pool the block if it is one of ours. Reports whether the pointer was ours. */
static bool npu_free_here(void *ptr, size_t size) {
    for (VramBuffer **curr = &vmm_table[vmm_hash((CUdeviceptr)ptr)]; *curr; curr = &(*curr)->next) {
        VramBuffer *entry = *curr;
        if (vrambuf_get(entry) != (CUdeviceptr)ptr) {
            continue;
        }
        if (npu_pool_contains(entry)) {
            /* torch_npu repeats a free for blocks it has already handed back
             * (tensor deletion plus a pool flush). The block is pooled once, so
             * the repeat is ignored rather than released twice. */
            log(DEBUG, "%s: ignoring repeated free of VRAM@%p\n", __func__, ptr);
            return true;
        }
        /* The entry stays registered and pooled, so the address cannot be handed
         * out by the driver while torch may still reach it. */
        allocations_dirty = true;
        allocations_account_free((CUdeviceptr)ptr, (CUstream)0);
        npu_pool_put(entry);
        log(VERBOSE, "Pooled: ptr=%p, size=%zuk\n", ptr, size / K);
        return true;
    }
    return false;
}

SHARED_EXPORT
void free_fn_npu(void *ptr, size_t size, aclrtStream stream) {
    CUresult graph_status = 0;
    bool graph_taken;

    log_shot(DEBUG, "Pytorch is freeing VRAM ...\n");
    log(VERBOSE, "free_fn_npu (start) ptr=%p size=%zuk\n", ptr, size / K);
    if (ptr == NULL || npu_plug_closed) {
        return;
    }
    graph_taken = malloc_graph_free((CUdeviceptr)(uintptr_t)ptr, (CUstream)stream, &graph_status);
    if (!graph_taken) {
        /* torch_npu hands free() a different stream object than the one the graph
         * accepted on the allocation; the block still belongs to the graph, and
         * its free has to be recorded for a replay to reuse the address. Every
         * recording device is tried: a graph ignores a pointer that is not in
         * its own reservation, so a wrong device cannot take it. */
        for (int i = 0; i < NPU_MAX_DEVICES && !graph_taken; i++) {
            if (npu_graph_stream[i] != NULL) {
                graph_taken = malloc_graph_free((CUdeviceptr)(uintptr_t)ptr, npu_graph_stream[i], &graph_status);
            }
        }
    }
    if (graph_taken) {
        return;
    }

    if (npu_free_here(ptr, size)) {
        return;
    }

    /* free() carries no device, and torch does not promise that the block's
     * device is the calling thread's current one: with two devices in one process
     * a copy engine or the other device can be current. Every initialized device
     * is searched, or its blocks would leak and its accounting would stay high. */
    AimdoContext *current = g_devctx;
    bool found = false;

    for (size_t index = 0, count = aimdo_devctx_count(); index < count && !found; index++) {
        AimdoContext *devctx = aimdo_devctx_at(index);

        if (devctx == current) {
            continue;
        }
        set_devctx(devctx);
        found = npu_free_here(ptr, size);
    }
    set_devctx(current);

    if (!found && malloc_graph_owns_address((CUdeviceptr)(uintptr_t)ptr)) {
        /* Blocks a graph handed out are freed by torch after the graph is gone
         * and its range is back with the driver. Every device's table was
         * searched first, so this cannot hide a live allocation of ours. */
        log(DEBUG, "%s: %p is inside a released graph range\n", __func__, ptr);
        return;
    }

    if (!found) {
        log(AIMDO_LOG_ERROR, "%s could not find VRAM@%p\n", __func__, ptr);
    }
}

#else

/* CUDA and ROCm return blocks to the driver on free, so there is nothing pooled
 * to release; the calls keep one interface for the cleanup path. */
size_t plug_release_pool(void) {
    return 0;
}

void plug_teardown(void) {
}

/* CUDA and ROCm ABI: both calls carry the device. */
SHARED_EXPORT
void *alloc_fn(size_t size, int device, cudaStream_t stream) {
    (void)stream;
    return plug_alloc("alloc_fn", size, device);
}

SHARED_EXPORT
void free_fn(void* ptr, size_t size, int device, cudaStream_t stream) {
    log_shot(DEBUG, "Pytorch is freeing VRAM ...\n");
    log(VERBOSE, "%s (start) ptr=%p size=%zuk, device=%d\n", __func__, ptr, size / K, device);
    if (ptr == NULL || !set_devctx_for_device(device)) {
        return;
    }

    for (VramBuffer **curr = &vmm_table[vmm_hash((CUdeviceptr)ptr)]; *curr; curr = &(*curr)->next) {
        VramBuffer *entry = *curr;
        if (vrambuf_get(entry) != (CUdeviceptr)ptr || entry->device != device) {
            continue;
        }

        *curr = entry->next;
        allocations_dirty = true;
        vrambuf_destroy(entry);
        log(VERBOSE, "Freed: ptr=%p, size=%zuk, stream=%p\n", ptr, size / K, stream);
        return;
    }

    log(AIMDO_LOG_ERROR, "%s could not find VRAM@%p\n", __func__, ptr);
}

#endif
