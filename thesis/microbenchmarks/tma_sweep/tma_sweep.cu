// Συγκρίνουμε τρεις τρόπους ανάγνωσης της ίδιας εισόδου από την GPU.
// Το TMA και το barrier βασίζονται στο παράδειγμα CUDA 12.8.1, ενότητα 7.29.2:
// https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#using-tma-to-transfer-multi-dimensional-arrays
// Host allocations:
// https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#mapped-memory
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/barrier>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
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
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

using Barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;
constexpr const char* names[] = {"loads_direct", "loads_shared", "tma_shared"};
constexpr const char* header =
    "input_bytes,tile_rows,tile_cols,tile_bytes,source,kernel,gpu_threads,launch_order,"
    "kernel_ms,effective_input_gb_s,status,cpu_node,mbind_success,cpu_residency_verified,"
    "pages_before,on_cpu_before,unknown_before,query_errno_before,"
    "pages_after,on_cpu_after,unknown_after,query_errno_after,message";

struct DeviceFault : std::runtime_error { using std::runtime_error::runtime_error; };
void check(cudaError_t e) {
    if (e == cudaSuccess) return;
    if (e == cudaErrorMemoryAllocation || e == cudaErrorInvalidValue || e == cudaErrorNotSupported)
        throw std::runtime_error(cudaGetErrorString(e));
    throw DeviceFault(cudaGetErrorString(e));
}
void stage(const char* s) { std::cout << "STAGE " << s << std::endl; }
uint32_t value(size_t i) { return uint32_t(i * 1664525ULL + 1013904223ULL); }

// Κάθε GPU block χειρίζεται ένα συνεχόμενο tile.
template<int E>
__global__ void loads_direct(const uint32_t* src, uint32_t* out) {
    size_t base = size_t(blockIdx.x) * E;
    for (int i = threadIdx.x; i < E; i += blockDim.x) out[base+i] = src[base+i];
}

// Άλλο GPU warp γράφει και άλλο διαβάζει κάθε τμήμα του tile.
// Έτσι χρειάζεται πραγματικά η shared memory και ο συγχρονισμός.
template<int E>
__device__ void store_tile(const uint32_t* tile, uint32_t* out, size_t base) {
    int first = (threadIdx.x + 32) % blockDim.x;
    for (int i = first; i < E; i += blockDim.x) out[base+i] = tile[i];
}

template<int E>
__global__ void loads_shared(const uint32_t* src, uint32_t* out) {
    extern __shared__ __align__(128) uint32_t tile[];
    size_t base = size_t(blockIdx.x) * E;
    for (int i = threadIdx.x; i < E; i += blockDim.x) tile[i] = src[base+i];
    __syncthreads();
    store_tile<E>(tile, out, base);
}

template<int H, int W>
__global__ void tma_shared(const __grid_constant__ CUtensorMap map, uint32_t* out) {
    extern __shared__ __align__(128) uint32_t tile[];
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ Barrier bar;
    if (threadIdx.x == 0) {
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

struct Location {
    size_t pages = 0, on_cpu = 0, unknown = 0;
    int error = 0;
    bool verified() const { return pages && pages == on_cpu && !unknown; }
};

// Ρωτάμε το Linux για όλες τις σελίδες, σε μικρές ομάδες. Δεν ζητάμε μετακίνηση.
Location locate(void* ptr, size_t bytes, unsigned node) {
    Location r;
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
                if (n < 0) ++r.unknown;
                else if (unsigned(n) == node) ++r.on_cpu;
            }
        }
    }
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
            check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&src), host, 0));
        }
        cudaPointerAttributes a{};
        check(cudaPointerGetAttributes(&a, src));
        if (a.type != (host ? cudaMemoryTypeHost : cudaMemoryTypeDevice))
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
    Location before, after;
};
struct Events {
    cudaEvent_t begin{}, end{};
    ~Events() { if (begin) cudaEventDestroy(begin); if (end) cudaEventDestroy(end); }
};

template<int H, int W>
void measure(Memory& m, int threads, int rotation, unsigned node, std::array<Row, 3>& rows) {
    constexpr int elements = H*W, shared_bytes = elements*4;
    unsigned blocks = unsigned(m.bytes/shared_bytes);
    // Τα μεγάλα tiles χρειάζονται ρητή άδεια για dynamic shared memory πάνω από 48 KiB.
    if (shared_bytes > 48*1024) {
        check(cudaFuncSetAttribute(loads_shared<elements>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
        check(cudaFuncSetAttribute(tma_shared<H,W>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
    }
    CUtensorMap map{};
    uint64_t dims[] = {W, m.bytes/(4*W)}, strides[] = {W*4};
    uint32_t box[] = {W, H}, steps[] = {1, 1};
    CUresult encoded = cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_UINT32, 2,
        m.src, dims, strides, box, steps, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (encoded != CUDA_SUCCESS) throw std::runtime_error("Tensor map encoding failed: "+std::to_string(encoded));
    Events e;
    check(cudaEventCreate(&e.begin));
    check(cudaEventCreate(&e.end));
    for (int position = 0; position < 3; ++position) {
        int method = (rotation+position)%3;
        Row& row = rows[method];
        row.order = position;
        row.status = "running";
        stage(names[method]);
        if (m.host) row.before = locate(m.host, m.bytes, node);
        check(cudaMemset(m.out, 0, m.bytes));
        check(cudaDeviceSynchronize());
        // Μία χρονισμένη εκτέλεση. Χωρίς warmup ή αλλαγή της cache.
        check(cudaEventRecord(e.begin));
        if (method == 0) loads_direct<elements><<<blocks, threads>>>(m.src, m.out);
        if (method == 1) loads_shared<elements><<<blocks, threads, shared_bytes>>>(m.src, m.out);
        if (method == 2) tma_shared<H,W><<<blocks, threads, shared_bytes>>>(map, m.out);
        check(cudaGetLastError());
        check(cudaEventRecord(e.end));
        check(cudaEventSynchronize(e.end));
        check(cudaEventElapsedTime(&row.ms, e.begin, e.end));
        if (!(row.ms > 0) || !std::isfinite(row.ms)) throw std::runtime_error("Invalid event time");
        verify(m);
        if (m.host) row.after = locate(m.host, m.bytes, node);
        row.status = "ok";
    }
}

// Το message είναι το τελευταίο CSV πεδίο. Κρατάμε σωστά τυχόν κόμματα και εισαγωγικά.
std::string quoted(std::string s) {
    std::string out = "\"";
    for (char c : s) { if (c == '"') out += '"'; out += (c == '\n' || c == '\r') ? ' ' : c; }
    return out+'"';
}

void save(const std::string& path, const Memory& m, int h, int w, int threads,
          unsigned node, const std::array<Row, 3>& rows) {
    std::ofstream csv(path+".tmp");
    csv << header << '\n' << std::setprecision(9);
    for (int i = 0; i < 3; ++i) {
        const Row& r = rows[i];
        int resident = m.source == "hbm" ? -1 : int(r.before.verified() && r.after.verified());
        csv << m.bytes << ',' << h << ',' << w << ',' << h*w*4 << ',' << m.source << ','
            << names[i] << ',' << threads << ',' << r.order << ',';
        if (r.status == "ok") csv << r.ms;
        csv << ',';
        if (r.status == "ok") csv << m.bytes/(double(r.ms)*1e6);
        csv << ',' << r.status << ',' << node << ',' << m.bound << ',' << resident << ','
            << r.before.pages << ',' << r.before.on_cpu << ',' << r.before.unknown << ',' << r.before.error << ','
            << r.after.pages << ',' << r.after.on_cpu << ',' << r.after.unknown << ',' << r.after.error << ','
            << quoted(r.message) << '\n';
    }
    csv.flush();
    if (!csv) throw std::runtime_error("Cannot write CSV");
    csv.close();
    if (std::rename((path+".tmp").c_str(), path.c_str())) throw std::runtime_error("Cannot finish CSV");
}

int main(int argc, char** argv) {
    try {
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
            (source != "hbm" && source != "pinned" && source != "registered") ||
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
            m.allocate(node);
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
        save(path, m, h, w, threads, node, rows);
        stage("finished");
        return code;
    } catch (const std::exception& e) {
        std::cerr << "ERROR " << e.what() << std::endl;
        return 2;
    }
}
