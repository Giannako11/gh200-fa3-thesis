// Το TMA και το barrier βασίζονται στο  CUDA 12.8.1, ενότητα 7.29.2:
// https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#using-tma-to-transfer-multi-dimensional-arrays
// Host allocations:
// https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#mapped-memory
// Απλό malloc σε συστήματα με ATS:
// https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#unified-memory-programming
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/barrier>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <linux/mempolicy.h>
#include <map>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

using Barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;
constexpr const char* names[] = {"loads_direct", "loads_shared", "tma_shared"};
constexpr const char* header = "input_bytes,tile_rows,tile_cols,tile_bytes,source,kernel,gpu_threads,first_sample_position,kernel_ms,effective_input_gb_s,status,cpu_node,mbind_success,cpu_residency_verified,pages_before,on_cpu_before,unknown_before,query_errno_before,pages_before_samples,on_cpu_before_samples,unknown_before_samples,query_errno_before_samples,pages_after,on_cpu_after,unknown_after,query_errno_after,pages_after_verification,on_cpu_after_verification,unknown_after_verification,query_errno_after_verification,warmup_launches,sample_count,timing_statistic,min_ms,p25_ms,p75_ms,p95_ms,max_ms,mean_ms,stddev_ms,cv,first_sample_ms,message";

struct DeviceFault : std::runtime_error { using std::runtime_error::runtime_error; };
void check(cudaError_t e) {
    if (e == cudaSuccess) return;
    if (e == cudaErrorMemoryAllocation || e == cudaErrorInvalidValue || e == cudaErrorNotSupported)
        throw std::runtime_error(cudaGetErrorString(e));
    throw DeviceFault(cudaGetErrorString(e));
}
void stage(const char* s) { std::cout << "STAGE " << s << std::endl; }
uint32_t value(size_t i) { return uint32_t(i * 1664525ULL + 1013904223ULL); }

// Κάθε GPU block χειρίζεται ένα συνεχόμενο tile. Η εκτελεση γινεται απο GPU thread και η εκκίνηση παει απο CPU
template<int E>
__global__ void loads_direct(const uint32_t* src, uint32_t* out) {
    size_t base = size_t(blockIdx.x) * E;
    for (int i = threadIdx.x; i < E; i += blockDim.x) out[base+i] = src[base+i];
}

// Άλλο GPU warp γράφει και άλλο διαβάζει κάθε τμήμα του tile.
// Έτσι χρειάζεται πραγματικά η shared memory και ο συγχρονισμός. Αλλιως μπορεί να ειχα δεδομενα να μενουν στους ρετζιστερς
template<int E>
__device__ void store_tile(const uint32_t* tile, uint32_t* out, size_t base) {
    int first = (threadIdx.x + 32) % blockDim.x;
    for (int i = first; i < E; i += blockDim.x) out[base+i] = tile[i];
}

// Εδω προσθετω ενδιάμεσο buffer για να τα στειλω στην shared mem
// και αντιστοιχα τα σωστα syncs
template<int E>
__global__ void loads_shared(const uint32_t* src, uint32_t* out) {
    extern __shared__ __align__(128) uint32_t tile[];
    size_t base = size_t(blockIdx.x) * E;
    for (int i = threadIdx.x; i < E; i += blockDim.x) tile[i] = src[base+i];
    __syncthreads();
    store_tile<E>(tile, out, base);
}
// TMA Implementation
template<int H, int W> // tile size
__global__ void tma_shared(const __grid_constant__ CUtensorMap map, uint32_t* out) { //readonly __grid constant
    extern __shared__ __align__(128) uint32_t tile[]; //
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ Barrier bar;
    if (threadIdx.x == 0) {  // edw ειναι η αρχικοποιηση, γινεται απο ενα thread μονο διοτι το bar απο πανω ειναι shared
        init(&bar, blockDim.x);
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();
    Barrier::arrival_token token;
    if (threadIdx.x == 0) {
        // Μόνο το GPU thread 0 ζητά τη μεταφορά ολόκληρου του tile.
        cde::cp_async_bulk_tensor_2d_global_to_shared(tile, &map, 0, blockIdx.x * H, bar);
        token = cuda::device::barrier_arrive_tx(bar, 1, H*W*sizeof(uint32_t));
    } else {
        token = bar.arrive();
    }
    bar.wait(std::move(token));
    store_tile<H*W>(tile, out, size_t(blockIdx.x) * H * W);
    __syncthreads();
    if (threadIdx.x == 0) bar.~Barrier();
}


// Query both the calling thread policy and the allocation's effective policy.
void diag_policy(void* address) {
    int mode = -1;
    unsigned long mask[16] = {};
    errno = 0;
    long rc = syscall(SYS_get_mempolicy, &mode, mask, 1024UL, address,
                      address ? MPOL_F_ADDR : 0);
    int error = errno;
    std::cout << "DIAG_POLICY scope=" << (address ? "address" : "thread")
              << " rc=" << rc << " errno=" << (rc < 0 ? error : 0)
              << " mode=" << mode << " nodes=";
    for (unsigned i=0; i<1024; ++i)
        if (mask[i/(8*sizeof(unsigned long))] & (1UL << (i%(8*sizeof(unsigned long)))))
            std::cout << i << ':';
    std::cout << std::endl;
}


void require_cpu_policy(void* address, unsigned node) {
    int mode = -1;
    unsigned long mask[16] = {};
    if (syscall(SYS_get_mempolicy, &mode, mask, 1024UL, address, MPOL_F_ADDR) != 0)
        throw std::runtime_error(std::string("get_mempolicy: ")+std::strerror(errno));
    if (mode != MPOL_BIND) throw std::runtime_error("Allocation is not MPOL_BIND");
    for (unsigned i=0; i<1024; ++i) {
        bool set = mask[i/(8*sizeof(unsigned long))] & (1UL << (i%(8*sizeof(unsigned long))));
        if (set != (i == node)) throw std::runtime_error("Allocation policy is not bound exclusively to CPU node");
    }
}

struct Location {
    size_t pages = 0, on_cpu = 0, unknown = 0;
    int error = 0;
    bool verified() const { return pages && pages == on_cpu && !unknown; }
};

// Ρωτάμε το Linux για όλες τις σελίδες, σε μικρές ομάδες. Δεν ζητάμε μετακίνηση.
// βλεπω δηλαδή απο που προέρχεται-υπαρχει η σελίδα
Location locate(void* ptr, size_t bytes, unsigned node) {
    Location r;
    std::map<int,size_t> node_counts, negative_status;
    size_t page = size_t(sysconf(_SC_PAGESIZE));
    uintptr_t begin = reinterpret_cast<uintptr_t>(ptr) & ~(uintptr_t(page)-1);
    uintptr_t end = reinterpret_cast<uintptr_t>(ptr) + bytes;
    while (begin < end) {
        std::vector<void*> addresses;
        for (; begin < end && addresses.size() < 4096; begin += page)
            addresses.push_back(reinterpret_cast<void*>(begin));
        std::vector<int> status(addresses.size(), -1);
        r.pages += addresses.size();
        if (syscall(SYS_move_pages, 0, addresses.size(), addresses.data(), nullptr,
                    status.data(), 0) < 0) {
            if (!r.error) r.error = errno;
            r.unknown += addresses.size();
        } else {
            for (int n : status) {
                if (n < 0) { ++r.unknown; ++negative_status[n]; }
                else { ++node_counts[n]; if (unsigned(n) == node) ++r.on_cpu; }
            }
        }
    }
    std::cout << "DIAG_LOCATION pages=" << r.pages << " nodes=";
    for (const auto& v : node_counts) std::cout << v.first << ":" << v.second << ";";
    std::cout << " negative_status=";
    for (const auto& v : negative_status) std::cout << v.first << ":" << v.second << ";";
    std::cout << " query_errno=" << r.error << std::endl;
    return r;
}

struct Memory {
    std::string source;
    size_t bytes;
    uint32_t *host = nullptr, *src = nullptr, *out = nullptr;
    bool registered = false;
    int bound = -1; // -1 σημαίνει ότι δεν ζητήθηκε mbind.
    Memory(std::string s, size_t n) : source(std::move(s)), bytes(n) {}
    ~Memory() {
        if (out) cudaFree(out);
        if (source == "hbm" && src) cudaFree(src);
        if (source == "pinned" && host) cudaFreeHost(host);
        if (source == "malloc_bound" && host) std::free(host);
        if (source == "registered" && host) {
            if (registered) cudaHostUnregister(host);
            munmap(host, bytes);
        }
    }
    void allocate(unsigned node) {
        size_t free_bytes = 0, total = 0;
        check(cudaMemGetInfo(&free_bytes, &total));
        size_t needed = source == "hbm" ? 2*bytes : bytes;
        if (free_bytes < needed + (256ULL << 20))
            throw std::runtime_error("Not enough free HBM for input/output and headroom");
        if (source != "hbm") {
            std::ifstream info("/proc/meminfo");
            std::string line;
            while (std::getline(info, line)) if (line.rfind("MemAvailable:", 0) == 0) {
                std::istringstream fields(line);
                std::string key;
                size_t available_kib = 0;
                fields >> key >> available_kib;
                if (available_kib*1024 < bytes + (256ULL << 20))
                    throw std::runtime_error("Not enough available CPU RAM for host input");
            }
        }
        if (source == "hbm") {
            check(cudaMalloc(reinterpret_cast<void**>(&src), bytes));
            // Γεμίζουμε την HBM σε κομμάτια, χωρίς δεύτερο τεράστιο πίνακα στην CPU.
            std::vector<uint32_t> part(std::min(bytes/4, size_t(1 << 20)));
            for (size_t start = 0; start < bytes/4; start += part.size()) {
                size_t n = std::min(part.size(), bytes/4-start);
                for (size_t i = 0; i < n; ++i) part[i] = value(start+i);
                check(cudaMemcpy(src+start, part.data(), n*4, cudaMemcpyHostToDevice));
            }
        } else {
            if (source == "pinned") {
                check(cudaHostAlloc(reinterpret_cast<void**>(&host), bytes, cudaHostAllocMapped));
            } else if (source == "malloc_bound") {
                int pageable = 0, host_tables = 0;
                check(cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, 0));
                check(cudaDeviceGetAttribute(&host_tables, cudaDevAttrPageableMemoryAccessUsesHostPageTables, 0));
                std::cout << "pageableMemoryAccess=" << pageable
                          << " pageableMemoryAccessUsesHostPageTables=" << host_tables << std::endl;
                if (!pageable || !host_tables)
                    throw std::runtime_error("malloc case requires GPU access to system memory with ATS");
                // Απλό malloc: δεν κάνουμε mbind, register ή αντιγραφή της εισόδου στην HBM.

                std::string mode = "rangebind";
                if (mode == "plain" || mode == "taskbind") {
                    host = static_cast<uint32_t*>(std::malloc(bytes));
                    if (!host) throw std::runtime_error("malloc failed");
                } else {
                    void* allocation = nullptr;
                    int rc = posix_memalign(&allocation, size_t(sysconf(_SC_PAGESIZE)), bytes);
                    if (rc) throw std::runtime_error(std::string("posix_memalign: ")+std::strerror(rc));
                    host = static_cast<uint32_t*>(allocation);
                }
                if (mode == "rangebind") {
                    if (node >= 8*sizeof(unsigned long)) throw std::runtime_error("CPU node exceeds diagnostic mask");
                    unsigned long mask = 1UL << node;
                    bound = syscall(SYS_mbind, host, bytes, MPOL_BIND, &mask,
                                    8*sizeof(unsigned long), 0) == 0;
                    if (!bound) throw std::runtime_error(std::string("diagnostic mbind: ")+std::strerror(errno));
                }
                if (mode == "preferred") {
                    check(cudaMemAdvise(host, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
                    std::cout << "DIAG_PREFERRED_HOST success=1" << std::endl;
                }

            } else {
                void* p = mmap(nullptr, bytes, PROT_READ|PROT_WRITE,
                               MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
                if (p == MAP_FAILED) throw std::runtime_error("mmap failed");
                host = static_cast<uint32_t*>(p);
                std::vector<unsigned long> mask(node/(8*sizeof(unsigned long))+1, 0);
                mask[node/(8*sizeof(unsigned long))] |= 1UL << (node%(8*sizeof(unsigned long)));
                bound = syscall(SYS_mbind, host, bytes, MPOL_BIND, mask.data(),
                                mask.size()*8*sizeof(unsigned long)+1UL, 0) == 0;
                if (!bound) throw std::runtime_error(std::string("mbind: ")+std::strerror(errno));
            }
            // Το CPU thread γράφει τα δεδομένα στην ίδια host allocation που θα διαβάσει η GPU.
            for (size_t i = 0; i < bytes/4; ++i) host[i] = value(i);
            if (source == "registered") {
                check(cudaHostRegister(host, bytes, cudaHostRegisterMapped));
                registered = true;
            }
            if (source == "malloc_bound") src = host;
            else check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&src), host, 0));
        }
        cudaPointerAttributes a{};
        check(cudaPointerGetAttributes(&a, src));
        cudaMemoryType expected_type = source == "malloc_bound" ? cudaMemoryTypeUnregistered :
                                       (host ? cudaMemoryTypeHost : cudaMemoryTypeDevice);
        if (a.type != expected_type)
            throw std::runtime_error("Unexpected CUDA memory type");
        check(cudaMalloc(reinterpret_cast<void**>(&out), bytes));
    }
};

// Η επαλήθευση γίνεται μετά τον χρονισμό, σε κομμάτια των 4 MiB.
void verify(const Memory& m) {
    std::vector<uint32_t> part(std::min(m.bytes/4, size_t(1 << 20)));
    for (size_t start = 0; start < m.bytes/4; start += part.size()) {
        size_t n = std::min(part.size(), m.bytes/4-start);
        check(cudaMemcpy(part.data(), m.out+start, n*4, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < n; ++i)
            if (part[i] != value(start+i))
                throw std::runtime_error("Output mismatch at element "+std::to_string(start+i));
    }
}

struct Row {
    std::string status = "not_run", message;
    float ms = -1;
    int order = -1;
    Location before, before_samples, after, after_verification;
    int warmup = 0;
    std::vector<double> samples;
    std::vector<int> positions;
};
struct Events {
    cudaEvent_t begin{}, end{};
    ~Events() { if (begin) cudaEventDestroy(begin); if (end) cudaEventDestroy(end); }
};

// Host-side measurement and CSV saving for the warmed/sampled sweep.
// Inserted after Row/Events into a generated copy of the verified CUDA benchmark.
// The three GPU kernels and their tiling are unchanged.
std::string quoted(std::string s);

constexpr int sampled_orders[6][3] = {
    {0,1,2}, {1,2,0}, {2,0,1}, {2,1,0}, {1,0,2}, {0,2,1}
};

double sample_quantile(std::vector<double> v, double q) {
    if (v.empty()) return -1;
    std::sort(v.begin(), v.end());
    double position = q * (v.size()-1);
    size_t lo = size_t(position), hi = std::min(lo+1, v.size()-1);
    return v[lo] + (v[hi]-v[lo]) * (position-lo);
}

double sample_mean(const std::vector<double>& v) {
    double total = 0;
    for (double x : v) total += x;
    return v.empty() ? -1 : total/v.size();
}

double sample_stddev(const std::vector<double>& v) {
    if (v.size()<2) return 0;
    double mean = sample_mean(v), total = 0;
    for (double x : v) total += (x-mean)*(x-mean);
    return std::sqrt(total/(v.size()-1));
}

// GPU eviction conditioning, not a portable hardware cache-invalidation API.
// Scratch traffic is explicit .ca loads + .wb stores. Each CTA touches >=512 KiB,
// total scratch >=8*reported L2. Every conditioning call checks SM coverage.
// It never evicts Grace CPU caches or claims guaranteed LPDDR service.
constexpr unsigned controlled_sm_slots=4096;

struct ControlledBuffer {
    void* ptr=nullptr;
    ~ControlledBuffer() { if (ptr) cudaFree(ptr); }
    void allocate(size_t bytes) { check(cudaMalloc(&ptr,bytes)); }
};

__global__ void controlled_evict(uint32_t* data,size_t words_per_block,unsigned* visited) {
    unsigned sm;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
    if (threadIdx.x==0) atomicExch(visited+(sm<controlled_sm_slots ? sm : controlled_sm_slots),1U);
    size_t base=size_t(blockIdx.x)*words_per_block;
    for (size_t i=threadIdx.x;i<words_per_block;i+=blockDim.x) {
        uint32_t v;
        uint32_t* address=data+base+i;
        asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(v) : "l"(address) : "memory");
        v=v*1664525U+1013904223U;
        asm volatile("st.global.wb.u32 [%0], %1;" :: "l"(address),"r"(v) : "memory");
    }
}

struct ControlledEviction {
    cudaDeviceProp prop{};
    ControlledBuffer scratch,coverage;
    size_t block_bytes=0,bytes=0;
    unsigned blocks=0;
    int passes=2;
    unsigned long long calls=0,launches=0;
    unsigned minimum_coverage=controlled_sm_slots;
    ControlledEviction() {
        check(cudaGetDeviceProperties(&prop,0));
        if (prop.l2CacheSize<=0 || prop.multiProcessorCount<=0 || prop.multiProcessorCount>=int(controlled_sm_slots))
            throw std::runtime_error("Cannot size eviction buffers from device properties");
        block_bytes=std::max(size_t(512<<10),2*size_t(prop.sharedMemPerMultiprocessor));
        block_bytes=(block_bytes+127)/128*128;
        blocks=unsigned(std::max(size_t(prop.multiProcessorCount)*8,
                                (8*size_t(prop.l2CacheSize)+block_bytes-1)/block_bytes));
        bytes=size_t(blocks)*block_bytes;
        scratch.allocate(bytes);coverage.allocate((controlled_sm_slots+1)*sizeof(unsigned));
        check(cudaMemset(scratch.ptr,0,bytes));
        check(cudaDeviceSynchronize());
        std::cout << "CONDITIONING l2_bytes=" << prop.l2CacheSize << " scratch_bytes=" << bytes
                  << " block_bytes=" << block_bytes << " blocks=" << blocks
                  << " expected_sms=" << prop.multiProcessorCount << " passes=" << passes << std::endl;
    }
    void run() {
        ++calls;
        check(cudaMemset(coverage.ptr,0,(controlled_sm_slots+1)*sizeof(unsigned)));
        std::vector<unsigned> flags(controlled_sm_slots+1);
        for (int attempt=0;attempt<3;++attempt) {
            for (int pass=0;pass<passes;++pass) {
                controlled_evict<<<blocks,256>>>(static_cast<uint32_t*>(scratch.ptr),block_bytes/4,
                                                static_cast<unsigned*>(coverage.ptr));
                check(cudaGetLastError());++launches;
            }
            check(cudaDeviceSynchronize());
            check(cudaMemcpy(flags.data(),coverage.ptr,flags.size()*sizeof(unsigned),cudaMemcpyDeviceToHost));
            unsigned count=0;
            for (unsigned i=0;i<controlled_sm_slots;++i) count+=flags[i]!=0;
            if (flags.back()) throw std::runtime_error("SM id exceeds eviction coverage buffer");
            if (count==unsigned(prop.multiProcessorCount)) {
                minimum_coverage=std::min(minimum_coverage,count);
                return;
            }
        }
        throw std::runtime_error("Eviction did not visit every reported SM");
    }
    void save(const std::string& path,const std::string& protocol) {
        std::ofstream f(path);
        f << "{\"protocol\":\"" << protocol << "\",\"l2_bytes\":" << prop.l2CacheSize
          << ",\"eviction_bytes\":" << bytes << ",\"block_bytes\":" << block_bytes
          << ",\"blocks\":" << blocks << ",\"expected_sms\":" << prop.multiProcessorCount
          << ",\"minimum_covered_sms\":" << minimum_coverage << ",\"passes\":" << passes
          << ",\"eviction_calls\":" << calls << ",\"eviction_kernel_launches\":" << launches
          << ",\"primers_per_sample\":" << (protocol=="self_primed" ? 1 : 0)
          << ",\"hardware_invalidation_claim\":false,\"cpu_cache_flush\":false}\n";
        if (!f) throw std::runtime_error("Cannot save conditioning metadata");
    }
};

int controlled_cache_check(const std::string& path) {
    check(cudaSetDeviceFlags(cudaDeviceMapHost));check(cudaSetDevice(0));
    cudaDeviceProp prop{};check(cudaGetDeviceProperties(&prop,0));
    if (prop.major!=9 || prop.minor!=0) throw std::runtime_error("Requires Hopper 9.0");
    ControlledEviction eviction;
    size_t bytes=(size_t(prop.l2CacheSize)/4/4096)*4096;
    if (bytes<4096) throw std::runtime_error("L2 too small for probe");
    ControlledBuffer input,output;input.allocate(bytes);output.allocate(bytes);
    check(cudaMemset(input.ptr,0x3c,bytes));
    Events e;check(cudaEventCreate(&e.begin));check(cudaEventCreate(&e.end));
    auto launch=[&]() {
        loads_direct<1024><<<unsigned(bytes/4096),128>>>(static_cast<uint32_t*>(input.ptr),static_cast<uint32_t*>(output.ptr));
        check(cudaGetLastError());
    };
    // Exercise the conditioning path and GPU clocks before the paired probe.
    for (int i=0;i<10;++i) { eviction.run();launch(); }
    check(cudaDeviceSynchronize());
    std::vector<double> warm,cold;
    for (int round=0;round<12;++round) {
        for (int position=0;position<2;++position) {
            bool prime=((position+round)%2)==0;
            eviction.run();
            if (prime) { launch();check(cudaDeviceSynchronize()); }
            check(cudaEventRecord(e.begin));launch();check(cudaEventRecord(e.end));
            check(cudaEventSynchronize(e.end));
            float ms=0;check(cudaEventElapsedTime(&ms,e.begin,e.end));
            if (!(ms>0) || !std::isfinite(ms)) throw std::runtime_error("Invalid cache probe time");
            (prime ? warm : cold).push_back(ms);
        }
    }
    std::vector<uint32_t> result(bytes/4);
    check(cudaMemcpy(result.data(),output.ptr,bytes,cudaMemcpyDeviceToHost));
    for (auto x:result) if (x!=0x3c3c3c3cU) throw std::runtime_error("Cache probe output mismatch");
    double warm_med=sample_quantile(warm,.5),cold_med=sample_quantile(cold,.5);
    bool separated=cold_med>warm_med*1.05 && sample_quantile(cold,.25)>sample_quantile(warm,.75);
    std::ofstream f(path);f << std::setprecision(12);
    f << "{\"probe_bytes\":" << bytes << ",\"l2_bytes\":" << prop.l2CacheSize
      << ",\"eviction_bytes\":" << eviction.bytes << ",\"expected_sms\":" << prop.multiProcessorCount
      << ",\"minimum_covered_sms\":" << eviction.minimum_coverage
      << ",\"self_primed_median_ms\":" << warm_med << ",\"gpu_evicted_median_ms\":" << cold_med
      << ",\"evicted_over_primed\":" << cold_med/warm_med
      << ",\"probe_separates_conditions\":" << (separated ? "true" : "false")
      << ",\"not_a_hardware_invalidation_proof\":true,\"self_primed_samples_ms\":[";
    for (size_t i=0;i<warm.size();++i) { if (i) f << ',';f << warm[i]; }
    f << "],\"gpu_evicted_samples_ms\":[";
    for (size_t i=0;i<cold.size();++i) { if (i) f << ',';f << cold[i]; }
    f << "]}\n";f.close();
    if (!f) throw std::runtime_error("Cannot save cache probe");
    if (!separated) throw std::runtime_error("Eviction sanity check inconclusive: inspect cache_check.json; full sweep not started");
    return 0;
}

template<int H,int W>
void measure(Memory& m,int threads,int rotation,unsigned node,std::array<Row,3>& rows) {
    constexpr int elements=H*W,shared_bytes=elements*4;
    unsigned blocks=unsigned(m.bytes/shared_bytes);
    int warmup=std::getenv("TMA_SAMPLE_WARMUP") ? std::stoi(std::getenv("TMA_SAMPLE_WARMUP")) : 10;
    int samples=std::getenv("TMA_SAMPLE_COUNT") ? std::stoi(std::getenv("TMA_SAMPLE_COUNT")) : 30;
    std::string protocol=std::getenv("TMA_CONTROLLED_PROTOCOL") ? std::getenv("TMA_CONTROLLED_PROTOCOL") : "";
    const char* case_dir=std::getenv("TMA_CONTROLLED_CASE_DIR");
    if (warmup<1 || samples<6 || samples%6 || !case_dir ||
        (protocol!="self_primed" && protocol!="gpu_evicted"))
        throw std::runtime_error("Invalid controlled measurement configuration");
    if (shared_bytes>48*1024) {
        check(cudaFuncSetAttribute(loads_shared<elements>,cudaFuncAttributeMaxDynamicSharedMemorySize,shared_bytes));
        check(cudaFuncSetAttribute(tma_shared<H,W>,cudaFuncAttributeMaxDynamicSharedMemorySize,shared_bytes));
    }
    CUtensorMap map{};
    uint64_t dims[]={W,m.bytes/(4*W)},strides[]={W*4};
    uint32_t box[]={W,H},steps[]={1,1};
    if (reinterpret_cast<uintptr_t>(m.src)%16) throw std::runtime_error("Unaligned TMA input");
    CUresult encoded=cuTensorMapEncodeTiled(&map,CU_TENSOR_MAP_DATA_TYPE_UINT32,2,m.src,dims,strides,box,steps,
        CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (encoded!=CUDA_SUCCESS) throw std::runtime_error("Tensor map encoding failed");
    auto launch=[&](int method) {
        if (method==0) loads_direct<elements><<<blocks,threads>>>(m.src,m.out);
        if (method==1) loads_shared<elements><<<blocks,threads,shared_bytes>>>(m.src,m.out);
        if (method==2) tma_shared<H,W><<<blocks,threads,shared_bytes>>>(map,m.out);
        check(cudaGetLastError());
    };
    ControlledEviction eviction;
    Events events;check(cudaEventCreate(&events.begin));check(cudaEventCreate(&events.end));
    check(cudaEventRecord(events.begin));check(cudaEventRecord(events.end));check(cudaEventSynchronize(events.end));
    for (int position=0;position<3;++position) {
        int method=(position+rotation)%3;
        Row& row=rows[method];row.status="running";row.warmup=warmup;
        stage(names[method]);
        if (m.host) { stage("placement_before_warmup");row.before=locate(m.host,m.bytes,node); }
        stage("correctness_before_sampling");
        check(cudaMemset(m.out,0,m.bytes));launch(method);check(cudaDeviceSynchronize());verify(m);
        stage("warmup");
        for (int round=0;round<warmup;++round) launch(method);
        check(cudaDeviceSynchronize());
        if (m.host) { stage("placement_before_samples");row.before_samples=locate(m.host,m.bytes,node); }
        stage("conditioned_samples");
        for (int sample=0;sample<samples;++sample) {
            // Same conditioning in both modes. CPU coverage queries, GPU scratch
            // work, and optional self-primer are all OUTSIDE the event interval.
            eviction.run();
            if (protocol=="self_primed") { launch(method);check(cudaDeviceSynchronize()); }
            check(cudaEventRecord(events.begin));launch(method);check(cudaEventRecord(events.end));
            check(cudaEventSynchronize(events.end));
            float ms=0;check(cudaEventElapsedTime(&ms,events.begin,events.end));
            if (!(ms>0) || !std::isfinite(ms)) throw std::runtime_error("Invalid controlled sample time");
            row.samples.push_back(ms);row.positions.push_back(position);
        }
        if (m.host) { stage("placement_after_samples_before_verification");row.after=locate(m.host,m.bytes,node); }
        stage("correctness_after_sampling");
        check(cudaMemset(m.out,0,m.bytes));launch(method);check(cudaDeviceSynchronize());verify(m);
        if (m.host) { stage("placement_after_verification");row.after_verification=locate(m.host,m.bytes,node); }
        row.ms=float(sample_quantile(row.samples,.5));row.order=position;row.status="ok";
    }
    eviction.save(std::string(case_dir)+"/conditioning.json",protocol);
}

void save(const std::string& path,const Memory& m,int h,int w,int threads,
          unsigned node,const std::array<Row,3>& rows) {
    std::ofstream csv(path+".tmp");
    csv << header << '\n' << std::setprecision(12);
    auto location=[&](const Location& l) {
        csv << l.pages << ',' << l.on_cpu << ',' << l.unknown << ',' << l.error << ',';
    };
    for (int i=0; i<3; ++i) {
        const Row& r=rows[i];
        int resident=m.source=="hbm" ? -1 : int(r.before.verified() && r.before_samples.verified() &&
                                               r.after.verified() && r.after_verification.verified());
        double median=sample_quantile(r.samples,0.5), mean=sample_mean(r.samples);
        double sd=sample_stddev(r.samples);
        csv << m.bytes << ',' << h << ',' << w << ',' << h*w*4 << ',' << m.source << ','
            << names[i] << ',' << threads << ',' << r.order << ',';
        if (r.status=="ok") csv << median;
        csv << ',';
        if (r.status=="ok") csv << m.bytes/(median*1e6);
        csv << ',' << r.status << ',' << node << ',' << m.bound << ',' << resident << ',';
        location(r.before); location(r.before_samples); location(r.after); location(r.after_verification);
        csv << r.warmup << ',' << r.samples.size() << ",median,";
        if (r.status=="ok")
            csv << sample_quantile(r.samples,0) << ',' << sample_quantile(r.samples,0.25) << ','
                << sample_quantile(r.samples,0.75) << ',' << sample_quantile(r.samples,0.95) << ','
                << sample_quantile(r.samples,1) << ',' << mean << ',' << sd << ',' << sd/mean << ','
                << r.samples.front();
        else csv << ",,,,,,,,";
        csv << ',' << quoted(r.message) << '\n';
    }
    csv.flush();
    if (!csv) throw std::runtime_error("Cannot write aggregate CSV");
    csv.close();
    if (std::rename((path+".tmp").c_str(),path.c_str())) throw std::runtime_error("Cannot finish aggregate CSV");
    std::string raw_path=path.substr(0,path.find_last_of('/')+1)+"samples.csv";
    std::ofstream raw(raw_path+".tmp");
    raw << "input_bytes,tile_rows,tile_cols,source,kernel,sample_index,position,kernel_ms,effective_input_gb_s\n"
        << std::setprecision(12);
    for (int method=0; method<3; ++method)
        for (size_t sample=0; sample<rows[method].samples.size(); ++sample) {
            double ms=rows[method].samples[sample];
            raw << m.bytes << ',' << h << ',' << w << ',' << m.source << ',' << names[method] << ','
                << sample+1 << ',' << rows[method].positions[sample] << ',' << ms << ',' << m.bytes/(ms*1e6) << '\n';
        }
    raw.flush();
    if (!raw) throw std::runtime_error("Cannot write raw samples");
    raw.close();
    if (std::rename((raw_path+".tmp").c_str(),raw_path.c_str())) throw std::runtime_error("Cannot finish samples CSV");
}


// Το message είναι το τελευταίο CSV πεδίο. Κρατάμε σωστά τυχόν κόμματα και εισαγωγικά.
std::string quoted(std::string s) {
    std::string out = "\"";
    for (char c : s) { if (c == '"') out += '"'; out += (c == '\n' || c == '\r') ? ' ' : c; }
    return out+'"';
}


int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "--cache-check") return controlled_cache_check(argv[2]);
        if (argc == 2 && std::string(argv[1]) == "--header") { std::cout << header << '\n'; return 0; }
        bool failure = argc == 10 && std::string(argv[1]) == "--failure";
        if (!failure && argc != 8) throw std::runtime_error("Usage: binary bytes rows cols source GPU_threads rotation csv");
        int offset = failure ? 1 : 0;
        size_t bytes = std::stoull(argv[1+offset]);
        int h = std::stoi(argv[2+offset]), w = std::stoi(argv[3+offset]);
        std::string source = argv[4+offset];
        int threads = std::stoi(argv[5+offset]), rotation = std::stoi(argv[6+offset]);
        std::string path = argv[7+offset];
        const std::vector<std::pair<int,int>> tiles = {{32,32},{32,128},{64,64},{128,32},{64,128},{128,64},{128,128}};
        if (std::find(tiles.begin(), tiles.end(), std::make_pair(h,w)) == tiles.end() ||
            bytes < size_t(h*w*4) || bytes > (16ULL << 30) || bytes % (h*w*4) ||
            (source != "hbm" && source != "pinned" && source != "registered" && source != "malloc_bound") ||
            (threads != 64 && threads != 128 && threads != 256) || rotation < 0 || rotation > 2)
            throw std::runtime_error("Invalid case parameters");
        Memory m(source, bytes);
        std::array<Row, 3> rows;
        unsigned cpu = 0, node = 0;
        int code = 0;
        if (failure) {
            for (auto& r : rows) { r.status = "worker_error"; r.message = argv[9]; }
            save(path, m, h, w, threads, node, rows);
            return 0;
        }
        try {
            stage("device_check");
            check(cudaSetDeviceFlags(cudaDeviceMapHost));
            check(cudaSetDevice(0));
            cudaDeviceProp p{};
            check(cudaGetDeviceProperties(&p, 0));
            if (p.major != 9 || p.minor != 0) throw std::runtime_error("Requires Hopper 9.0");
            if (syscall(SYS_getcpu, &cpu, &node, nullptr)) throw std::runtime_error("getcpu failed");
            std::cout << "GPU=" << p.name << " CPU=" << cpu << " NUMA=" << node << std::endl;
            stage("allocation");
            diag_policy(nullptr);
            m.allocate(node);
            if (m.host) diag_policy(m.host);
            if (m.source == "malloc_bound") require_cpu_policy(m.host, node);
            stage("measure");
            if (h == 32 && w == 32) measure<32,32>(m, threads, rotation, node, rows);
            if (h == 32 && w == 128) measure<32,128>(m, threads, rotation, node, rows);
            if (h == 64 && w == 64) measure<64,64>(m, threads, rotation, node, rows);
            if (h == 128 && w == 32) measure<128,32>(m, threads, rotation, node, rows);
            if (h == 64 && w == 128) measure<64,128>(m, threads, rotation, node, rows);
            if (h == 128 && w == 64) measure<128,64>(m, threads, rotation, node, rows);
            if (h == 128 && w == 128) measure<128,128>(m, threads, rotation, node, rows);
        } catch (const DeviceFault& e) {
            code = 3;
            for (auto& r : rows) if (r.status != "ok") { r.message = e.what(); r.status = "device_fault"; }
        } catch (const std::exception& e) {
            code = 2;
            for (auto& r : rows) if (r.status != "ok") {
                r.message = e.what();
                r.status = r.status == "running" ? "error" : "not_run";
            }
        }
        for (const auto& r : rows) if (r.status != "ok" && code == 0) code = 2;
        save(path, m, h, w, threads, node, rows);
        stage("finished");
        return code;
    } catch (const std::exception& e) {
        std::cerr << "ERROR " << e.what() << std::endl;
        return 2;
    }
}
