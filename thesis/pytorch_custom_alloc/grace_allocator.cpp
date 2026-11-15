// Scoped PyTorch MemPool callbacks for one visible GH200.
#include <cuda_runtime_api.h>

#include <linux/mempolicy.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>

namespace {
constexpr size_t budget = 128 * 1024 * 1024;
constexpr int grace_node = 0;

struct Segment {
    void* base;
    size_t bytes;
    size_t mapped;
    size_t charge;
};

struct PoolState {
    std::mutex mutex;
    std::map<uintptr_t, Segment> segments;
    size_t live = 0;
    bool hardware_checked = false;
};

PoolState& state() {
    // This registry must survive CUDA library teardown. The test explicitly
    // releases every segment before exiting.
    static auto* pool = new PoolState;
    return *pool;
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void check_cuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

[[noreturn]] void fail_callback(const char* operation, const std::exception& error) {
    std::fprintf(stderr, "Grace allocator %s: %s\n", operation, error.what());
    std::abort();
}

size_t page_size() {
    const long page = sysconf(_SC_PAGESIZE);
    require(page >= 4096 && (page & (page - 1)) == 0, "invalid Linux page size");
    return static_cast<size_t>(page);
}

void check_hardware(PoolState& pool, int device) {
    int current = -1;
    check_cuda(cudaGetDevice(&current), "cudaGetDevice");
    require(device == 0 && current == device, "expected CUDA device 0");
    if (pool.hardware_checked) {
        return;
    }

    int count = 0;
    int ats = 0;
    int pageable = 0;
    int concurrent = 0;
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    check_cuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    check_cuda(cudaDeviceGetAttribute(&ats, cudaDevAttrPageableMemoryAccessUsesHostPageTables,
                                     device), "query host page tables");
    check_cuda(cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, device),
               "query pageable memory access");
    check_cuda(cudaDeviceGetAttribute(&concurrent, cudaDevAttrConcurrentManagedAccess, device),
               "query concurrent memory access");
    require(count == 1 && std::strstr(properties.name, "GH200") && ats && pageable && concurrent,
            "requires one visible GH200 with ATS and pageable GPU access");
    pool.hardware_checked = true;
}

void bind_pages(void* ptr, size_t bytes) {
    const unsigned long mask = 1UL << grace_node;
    const long status = syscall(SYS_mbind, ptr, bytes, MPOL_BIND, &mask,
                                sizeof(mask) * 8UL, 0UL);
    if (status != 0) {
        throw std::runtime_error("mbind failed, errno=" + std::to_string(errno));
    }
}

bool reset_policy(void* ptr, size_t bytes) {
    return syscall(SYS_mbind, ptr, bytes, MPOL_DEFAULT, nullptr, 0UL, 0UL) == 0;
}

size_t verify_pages(const void* ptr, size_t bytes, size_t page) {
    constexpr size_t batch = 512;
    const auto address = reinterpret_cast<uintptr_t>(ptr);
    const size_t count = (address % page + bytes + page - 1) / page;
    const auto first = address - address % page;
    for (size_t offset = 0; offset < count; offset += batch) {
        void* pages[batch];
        int nodes[batch];
        const size_t size = std::min(batch, count - offset);
        for (size_t i = 0; i < size; ++i) {
            pages[i] = reinterpret_cast<void*>(first + (offset + i) * page);
        }
        const long status = syscall(SYS_move_pages, 0, size, pages, nullptr, nodes, 0);
        if (status != 0) {
            throw std::runtime_error("move_pages query failed, errno=" + std::to_string(errno));
        }
        for (size_t i = 0; i < size; ++i) {
            if (nodes[i] != grace_node) {
                throw std::runtime_error("expected NUMA node 0, move_pages returned "
                                         + std::to_string(nodes[i]));
            }
        }
    }
    return count;
}
}  // namespace

// Match the actual PyTorch 2.11 libtorch callback typedefs in Module.cpp.
extern "C" void* grace_alloc(size_t size, int device, cudaStream_t stream) noexcept {
    try {
        if (size == 0 || size > budget) {
            return nullptr;
        }
        auto& pool = state();
        const std::lock_guard<std::mutex> lock(pool.mutex);
        check_hardware(pool, device);

        cudaStreamCaptureStatus capture;
        check_cuda(cudaStreamIsCapturing(stream, &capture), "cudaStreamIsCapturing");
        require(capture == cudaStreamCaptureStatusNone, "CUDA graph capture is unsupported");

        const size_t page = page_size();
        if (page - 1 > (std::numeric_limits<size_t>::max() - size) / 2) {
            return nullptr;
        }
        const size_t mapped = (size + page - 1) & ~(page - 1);
        const size_t charge = mapped + page - 1;
        if (charge > budget - pool.live) {
            return nullptr;
        }

        void* base = std::malloc(charge);
        if (!base) {
            return nullptr;
        }
        const auto address = (reinterpret_cast<uintptr_t>(base) + page - 1) & ~(page - 1);
        auto* ptr = reinterpret_cast<void*>(address);
        try {
            bind_pages(ptr, mapped);
            std::memset(ptr, 0, mapped);
            const size_t pages = verify_pages(ptr, mapped, page);
            pool.segments.emplace(address, Segment{base, size, mapped, charge});
            pool.live += charge;
            std::fprintf(stderr, "grace_alloc ptr=%p bytes=%zu pages=%zu live=%zu\n",
                         ptr, size, pages, pool.live);
        } catch (...) {
            if (reset_policy(ptr, mapped)) {
                std::free(base);
            }
            throw;
        }
        return ptr;
    } catch (const std::exception& error) {
        fail_callback("allocate", error);
    }
}

extern "C" void grace_free(void* ptr, size_t size, int device, cudaStream_t) noexcept {
    if (!ptr) {
        return;
    }
    try {
        auto& pool = state();
        {
            const std::lock_guard<std::mutex> lock(pool.mutex);
            check_hardware(pool, device);
        }
        check_cuda(cudaDeviceSynchronize(), "synchronize before grace_free");
        const std::lock_guard<std::mutex> lock(pool.mutex);
        const auto entry = pool.segments.find(reinterpret_cast<uintptr_t>(ptr));
        require(entry != pool.segments.end(), "unowned segment in grace_free");
        const auto segment = entry->second;
        require(size == segment.bytes, "segment size mismatch");
        require(reset_policy(ptr, segment.mapped), "cannot reset segment policy before free");
        std::free(segment.base);
        pool.segments.erase(entry);
        pool.live -= segment.charge;
        std::fprintf(stderr, "grace_free ptr=%p bytes=%zu live=%zu\n",
                     ptr, segment.bytes, pool.live);
    } catch (const std::exception& error) {
        fail_callback("free", error);
    }
}

extern "C" ssize_t grace_verify(const void* ptr, size_t bytes) noexcept {
    try {
        auto& pool = state();
        const std::lock_guard<std::mutex> lock(pool.mutex);
        const auto address = reinterpret_cast<uintptr_t>(ptr);
        auto entry = pool.segments.upper_bound(address);
        if (entry == pool.segments.begin() || bytes == 0) {
            return 0;
        }
        --entry;
        const size_t offset = address - entry->first;
        if (offset >= entry->second.bytes || bytes > entry->second.bytes - offset) {
            return 0;
        }
        return static_cast<ssize_t>(verify_pages(ptr, bytes, page_size()));
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Grace allocator verification: %s\n", error.what());
        return -1;
    }
}

extern "C" size_t grace_live_bytes() noexcept {
    auto& pool = state();
    const std::lock_guard<std::mutex> lock(pool.mutex);
    return pool.live;
}

extern "C" int is_device_pointer(const void* ptr) noexcept {
    cudaPointerAttributes attr{};
    const auto result = cudaPointerGetAttributes(&attr, ptr);
    if (result != cudaSuccess) {
        std::fprintf(stderr, "cudaPointerGetAttributes: %s\n", cudaGetErrorString(result));
        return -1;
    }
    return attr.type == cudaMemoryTypeDevice && attr.device == 0;
}
