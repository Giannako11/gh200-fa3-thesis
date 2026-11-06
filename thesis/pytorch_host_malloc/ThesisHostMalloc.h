#pragma once
// Private experimental backend. Enabled only by THESIS_HOST_MALLOC=1 at load.
// Linux GH200, one visible GPU, native eager allocator only.
#include <cuda_runtime_api.h>
#include <linux/mempolicy.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace thesis_host_malloc {
inline void require(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error("THESIS_HOST_MALLOC: " + message);
}
// Called by BackendStaticInitializer: freeze the flag before any allocations.
// Do not throw from a shared-library static constructor.
inline int frozen_mode() {
  static const int mode = [] {
    const char* v = std::getenv("THESIS_HOST_MALLOC");
    return !v || !std::strcmp(v, "0") ? 0 : (!std::strcmp(v, "1") ? 1 : 2);
  }();
  return mode;
}
inline bool enabled() { return frozen_mode() != 0; }
inline size_t number(const char* name, size_t fallback) {
  const char* v = std::getenv(name);
  if (!v) return fallback;
  require(*v >= '0' && *v <= '9', std::string("invalid ") + name);
  errno = 0;
  char* end = nullptr;
  auto n = std::strtoull(v, &end, 10);
  require(!errno && end && !*end && n > 0 && n <= SIZE_MAX,
          std::string("invalid ") + name);
  return static_cast<size_t>(n);
}
struct Block { void* base; size_t bytes; size_t charge; };
struct State {
  std::mutex mutex;
  std::map<uintptr_t, Block> blocks;
  size_t live = 0;
  size_t limit = number("THESIS_HOST_MALLOC_MAX_BYTES", size_t(1) << 30);
  size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  FILE* log = nullptr;
  bool hardware_checked = false;
  State() {
    require(frozen_mode() == 1, "flag must be exactly 0 or 1");
    require(page >= 4096 && (page & (page - 1)) == 0, "invalid system page size");
    const char* path = std::getenv("THESIS_HOST_MALLOC_LOG");
    require(path && *path == '/', "absolute THESIS_HOST_MALLOC_LOG required");
    // A process owns its own fresh log; never truncate another run's evidence.
    int fd = ::open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    require(fd >= 0, "cannot exclusively create allocation log");
    log = ::fdopen(fd, "w");
    if (!log) { ::close(fd); throw std::runtime_error("THESIS_HOST_MALLOC: fdopen failed"); }
    std::setvbuf(log, nullptr, _IOLBF, 0);
    std::fprintf(log, "{\"event\":\"config\",\"schema\":1,\"pid\":%ld,\"page_size\":%zu,\"limit\":%zu,\"node\":0}\n",
                 static_cast<long>(getpid()), page, limit);
  }
};
inline State& state() {
  // CUDA/process teardown order is unspecified. Explicit releases still free
  // their blocks; OS teardown reclaims any blocks retained in PyTorch's cache.
  static State* s = new State;
  return *s;
}
inline void cuda_ok(cudaError_t e, const char* operation) {
  require(e == cudaSuccess, std::string(operation) + ": " + cudaGetErrorString(e));
}
inline void check_hardware(State& s) {
  if (s.hardware_checked) return;
  int count = 0, device = -1, ats = 0, pageable = 0, concurrent = 0;
  cuda_ok(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
  cuda_ok(cudaGetDevice(&device), "cudaGetDevice");
  require(count == 1 && device == 0, "exactly one visible GPU (logical device 0) required");
  cudaDeviceProp p{};
  cuda_ok(cudaGetDeviceProperties(&p, 0), "cudaGetDeviceProperties");
  cuda_ok(cudaDeviceGetAttribute(&ats, cudaDevAttrPageableMemoryAccessUsesHostPageTables, 0), "ATS query");
  cuda_ok(cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, 0), "pageable query");
  cuda_ok(cudaDeviceGetAttribute(&concurrent, cudaDevAttrConcurrentManagedAccess, 0), "concurrent query");
  require(p.major == 9 && p.minor == 0 && std::strstr(p.name, "GH200") &&
          ats == 1 && pageable == 1 && concurrent == 1, "requires GH200 with ATS/pageable/concurrent access");
  std::fprintf(s.log, "{\"event\":\"hardware\",\"ats\":1,\"visible_gpus\":1}\n");
  s.hardware_checked = true;
}
inline void check_stream(cudaStream_t stream) {
  cudaStreamCaptureStatus capture;
  cuda_ok(cudaStreamIsCapturing(stream, &capture), "capture query");
  require(capture == cudaStreamCaptureStatusNone, "CUDA graph capture unsupported");
}
inline void guard(cudaStream_t stream) {
  if (!enabled()) return;
  auto& s = state();
  check_stream(stream);
  std::lock_guard<std::mutex> lock(s.mutex);
  check_hardware(s);
}
inline void check_policy(void* ptr) {
  int policy = -1;
  unsigned long mask[1024 / (8 * sizeof(unsigned long))] = {};
  long rc = syscall(SYS_get_mempolicy, &policy, mask, 1024UL, ptr, MPOL_F_ADDR);
  bool only_zero = mask[0] == 1UL;
  for (size_t i = 1; i < sizeof(mask) / sizeof(mask[0]); ++i) only_zero &= mask[i] == 0;
  require(rc == 0 && policy == MPOL_BIND && only_zero,
          "range policy must bind exclusively to NUMA node 0");
}
inline void check_pages(void* ptr, size_t bytes, size_t page) {
  constexpr size_t batch = 512;
  for (size_t offset = 0; offset < bytes;) {
    size_t n = std::min(batch, (bytes - offset) / page);
    std::vector<void*> pages(n);
    std::vector<int> status(n, -9999);
    for (size_t i = 0; i < n; ++i) pages[i] = static_cast<char*>(ptr) + offset + i * page;
    long rc = syscall(SYS_move_pages, 0, n, pages.data(), nullptr, status.data(), 0);
    require(rc == 0, "move_pages query failed (all pages must be queryable)");
    for (int node : status) require(node == 0, "page not resident on CPU NUMA node 0");
    offset += n * page;
  }
}
inline cudaError_t allocate(void** out, size_t bytes) {
  *out = nullptr;
  auto& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  check_hardware(s);
  if (!bytes) return cudaSuccess;
  if (bytes > SIZE_MAX - (s.page - 1)) return cudaErrorMemoryAllocation;
  size_t rounded = (bytes + s.page - 1) & ~(s.page - 1);
  if (rounded > SIZE_MAX - (s.page - 1)) return cudaErrorMemoryAllocation;
  size_t charge = rounded + s.page - 1;
  if (charge > s.limit || s.live > s.limit - charge) {
    std::fprintf(s.log, "{\"event\":\"budget_reject\",\"bytes\":%zu,\"live\":%zu}\n", bytes, s.live);
    return cudaErrorMemoryAllocation;
  }
  void* base = std::malloc(charge);
  if (!base) return cudaErrorMemoryAllocation;
  uintptr_t address = (reinterpret_cast<uintptr_t>(base) + s.page - 1) & ~(uintptr_t(s.page - 1));
  void* ptr = reinterpret_cast<void*>(address);
  try {
    unsigned long mask = 1;
    long rc = syscall(SYS_mbind, ptr, rounded, MPOL_BIND, &mask, 8UL * sizeof(mask), 0);
    require(rc == 0, "mbind failed, errno=" + std::to_string(errno));
    // Populate and verify before exposing the block to any GPU operation.
    // Ranges comprise complete pages entirely inside this malloc allocation.
    std::memset(ptr, 0, rounded);
    check_policy(ptr);
    check_pages(ptr, rounded, s.page);
    s.blocks.emplace(address, Block{base, rounded, charge});
  } catch (...) {
    // Do not return a policy-modified range to libc unless reset succeeds.
    // On reset failure, deliberately retain it until this failing process exits.
    if (syscall(SYS_mbind, ptr, rounded, MPOL_DEFAULT, nullptr, 0UL, 0) == 0)
      std::free(base);
    throw;
  }
  s.live += charge;
  std::fprintf(s.log, "{\"event\":\"alloc\",\"ptr\":%llu,\"base\":%llu,\"bytes\":%zu,\"mapped_bytes\":%zu,\"charge\":%zu,\"live\":%zu,\"node\":0}\n",
               static_cast<unsigned long long>(address), static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(base)),
               bytes, rounded, charge, s.live);
  *out = ptr;
  return cudaSuccess;
}
inline bool contains_locked(const State& s, const void* ptr, size_t bytes) {
  auto address = reinterpret_cast<uintptr_t>(ptr);
  auto it = s.blocks.upper_bound(address);
  if (it == s.blocks.begin()) return false;
  --it;
  size_t offset = address - it->first;
  return offset < it->second.bytes && bytes <= it->second.bytes - offset;
}
inline cudaError_t release(void* ptr) {
  if (!ptr) return cudaSuccess;
  // cudaFree used to provide implicit synchronization on actual segment
  // release. Preserve safety before replacing it with non-synchronizing free.
  guard(nullptr);
  cuda_ok(cudaDeviceSynchronize(), "synchronize before free");
  auto& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  auto it = s.blocks.find(reinterpret_cast<uintptr_t>(ptr));
  require(it != s.blocks.end(), "attempt to free an unknown host block");
  check_policy(ptr);
  check_pages(ptr, it->second.bytes, s.page);
  auto block = it->second;
  // Remove VMA policy before libc can reuse the allocation's pages.
  long rc = syscall(SYS_mbind, ptr, block.bytes, MPOL_DEFAULT, nullptr, 0UL, 0);
  require(rc == 0, "cannot reset range policy before free");
  s.blocks.erase(it);
  s.live -= block.charge;
  std::free(block.base);
  std::fprintf(s.log, "{\"event\":\"free\",\"ptr\":%llu,\"live\":%zu,\"node\":0}\n",
               static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(ptr)), s.live);
  return cudaSuccess;
}
inline cudaError_t copy(void* dst, const void* src, size_t bytes, cudaStream_t stream) {
  guard(stream);
  cuda_ok(cudaStreamSynchronize(stream), "synchronize before host-backed copy");
  auto& s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  require(!bytes || (contains_locked(s, dst, bytes) && contains_locked(s, src, bytes)),
          "CUDA-to-CUDA copy includes external/non-host-backed storage");
  if (bytes) std::memmove(dst, src, bytes);
  std::fprintf(s.log, "{\"event\":\"host_copy\",\"bytes\":%zu}\n", bytes);
  return cudaSuccess;
}
} // namespace thesis_host_malloc
