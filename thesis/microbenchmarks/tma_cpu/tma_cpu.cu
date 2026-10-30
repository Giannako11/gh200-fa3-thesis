// TMA mechanism probe, following the CUDA 12.8 Programming Guide section 7.29.2.
// Χάρτης πειράματος: 1 = CPU allocation/γνωστές τιμές, 2 = κανονικά GPU loads,
// 3 = TMA ανάγνωση, 4 = σύγκριση αποτελέσματος, 5 = CPU NUMA residency.
// Τα στάδια 2-4 ελέγχονται επίσης με πηγή HBM ως δοκιμή αναφοράς.
// Το στάδιο 5 γίνεται πριν και μετά τους kernels, όχι μόνο στο τέλος.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/barrier>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <cerrno>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <linux/mempolicy.h>

using Barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;
constexpr int WIDTH = 32, TILE = 1024, THREADS = 128;

void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
void stage(const char* name) { std::cout << "STAGE " << name << std::endl; }
// ΣΤΑΔΙΟ 1 — Γνωστή, επαναλήψιμη τιμή για κάθε θέση i της εισόδου.
// Η ίδια συνάρτηση δίνει την αναμενόμενη τιμή στη σύγκριση του σταδίου 4.
uint32_t value(size_t i) { return uint32_t(i * 1664525ULL + 1013904223ULL); }

// ΣΤΑΔΙΟ 2 — Κανονικά GPU loads: διαβάζουμε την πηγή χωρίς TMA.
// Τα threads μοιράζονται τα στοιχεία και τα γράφουν στο ξεχωριστό output.
__global__ void load_copy(const uint32_t* src, uint32_t* out) {
    size_t base = size_t(blockIdx.x) * TILE;
    for (int i = threadIdx.x; i < TILE; i += blockDim.x) out[base+i] = src[base+i];
}

// ΣΤΑΔΙΟ 3 — Ο TMA kernel: πηγή του descriptor -> shared memory -> output HBM.
__global__ void tma_copy(const __grid_constant__ CUtensorMap map, uint32_t* out) {
    // Ένα tile 32x32 uint32_t = 4096 bytes, με alignment 128 bytes.
    __shared__ alignas(128) uint32_t tile[TILE];
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ Barrier barrier;
    if (threadIdx.x == 0) {
        // Αρχικοποίηση του barrier για όλα τα threads και δημοσίευση στο async proxy.
        init(&barrier, blockDim.x);
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();
    Barrier::arrival_token token;
    if (threadIdx.x == 0) {
        // ΣΤΑΔΙΟ 3Α — Ένα thread ζητά TMA μεταφορά του tile στη shared memory.
        cde::cp_async_bulk_tensor_2d_global_to_shared(tile, &map, 0, blockIdx.x * WIDTH, barrier);
        // Δηλώνει μία άφιξη thread και τα 4096 bytes που πρέπει να ολοκληρωθούν.
        token = cuda::device::barrier_arrive_tx(barrier, 1, sizeof(tile));
    } else {
        // Τα υπόλοιπα threads δηλώνουν τη δική τους άφιξη στο barrier.
        token = barrier.arrive();
    }
    // ΣΤΑΔΙΟ 3Β — Περιμένουμε threads και μεταφορά πριν διαβάσουμε το tile.
    barrier.wait(std::move(token));
    size_t base = size_t(blockIdx.x) * TILE;
    // ΣΤΑΔΙΟ 3Γ — Γράφουμε όσα έφερε το TMA σε output HBM για επαλήθευση.
    for (int i = threadIdx.x; i < TILE; i += blockDim.x) out[base+i] = tile[i];
    __syncthreads();
    if (threadIdx.x == 0) barrier.~Barrier();
}

struct Residency {
    long pages = 0, on_cpu_node = 0, unknown = 0;
    int query_errno = 0;
    bool verified() const { return pages > 0 && on_cpu_node == pages && unknown == 0; }
};

// ΣΤΑΔΙΟ 5 — Ρωτάμε το Linux σε ποιον NUMA node βρίσκεται κάθε σελίδα πηγής.
// Το move_pages με nodes=nullptr ζητά πληροφορία, όχι μετακίνηση σελίδων.
// Αποτυχία query σημαίνει άγνωστη τοποθέτηση, όχι επιβεβαίωση CPU residency.
Residency residency(void* ptr, size_t bytes, unsigned cpu_node) {
    Residency r;
    size_t page = size_t(sysconf(_SC_PAGESIZE));
    uintptr_t start = reinterpret_cast<uintptr_t>(ptr) & ~(uintptr_t(page)-1);
    uintptr_t end = reinterpret_cast<uintptr_t>(ptr) + bytes;
    std::vector<void*> addresses;
    for (uintptr_t p = start; p < end; p += page) addresses.push_back(reinterpret_cast<void*>(p));
    r.pages = addresses.size();
    std::vector<int> status(addresses.size(), -1);
    if (syscall(SYS_move_pages, 0, addresses.size(), addresses.data(), nullptr, status.data(), 0) < 0) {
        r.query_errno = errno; r.unknown = r.pages; return r;
    }
    for (int n : status) {
        if (n < 0) ++r.unknown;
        else if (unsigned(n) == cpu_node) ++r.on_cpu_node;
    }
    return r;
}

void print_residency(const char* when, const Residency& r, unsigned node) {
    std::cout << "RESIDENCY " << when << " cpu_node=" << node << " pages=" << r.pages
              << " on_cpu_node=" << r.on_cpu_node << " unknown=" << r.unknown
              << " query_errno=" << r.query_errno << std::endl;
}

// ΣΤΑΔΙΟ 4 — Επιστρέφουμε το output στην CPU και ελέγχουμε κάθε στοιχείο.
// Αυτή η αντιγραφή αφορά μόνο την επαλήθευση του output, όχι staging της εισόδου.
void verify(uint32_t* out, size_t count) {
    std::vector<uint32_t> result(count);
    check(cudaMemcpy(result.data(), out, count*sizeof(uint32_t), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < count; ++i)
        if (result[i] != value(i)) throw std::runtime_error("Output mismatch at element " + std::to_string(i));
}

int main(int argc, char** argv) {
    try {
        if (argc != 5) throw std::runtime_error("Usage: binary hbm|pinned|registered loads|tma bytes repetitions");
        std::string source = argv[1], method = argv[2];
        size_t bytes = std::stoull(argv[3]);
        int repetitions = std::stoi(argv[4]);
        if ((source != "hbm" && source != "pinned" && source != "registered") ||
            (method != "loads" && method != "tma") || bytes < 4096 || bytes > 64*1024*1024 ||
            bytes % 4096 || repetitions < 1 || repetitions > 1000)
            throw std::runtime_error("Invalid arguments (4KiB multiples, <=64MiB, repetitions 1..1000)");
        stage("device_check");
        check(cudaSetDeviceFlags(cudaDeviceMapHost));
        check(cudaSetDevice(0));
        cudaDeviceProp props{}; check(cudaGetDeviceProperties(&props, 0));
        if (props.major != 9 || props.minor != 0) throw std::runtime_error("Requires Hopper capability 9.0");
        unsigned cpu = 0, node = 0;
        if (syscall(SYS_getcpu, &cpu, &node, nullptr) != 0) throw std::runtime_error("getcpu failed");
        // The OS node of the running Grace CPU is used for placement checks.
        std::cout << "CPU node=" << node << " cpu=" << cpu << std::endl;
        stage("allocation");
        uint32_t *host = nullptr, *src = nullptr, *out = nullptr;
        bool bound = false;
        size_t count = bytes / sizeof(uint32_t);
        if (source == "hbm") {
            // Δοκιμή αναφοράς: μόνο εδώ αντιγράφουμε την είσοδο σε GPU allocation.
            std::vector<uint32_t> initial(count);
            for (size_t i = 0; i < count; ++i) initial[i] = value(i);
            check(cudaMalloc(reinterpret_cast<void**>(&src), bytes));
            check(cudaMemcpy(src, initial.data(), bytes, cudaMemcpyHostToDevice));
        } else {
            // ΣΤΑΔΙΟ 1Α — Δεσμεύουμε host μνήμη με έναν από τους δύο τρόπους.
            if (source == "pinned") {
                // Host allocation από τη CUDA, mapped για πρόσβαση από GPU.
                check(cudaHostAlloc(reinterpret_cast<void**>(&host), bytes, cudaHostAllocMapped));
            } else {
                // Host allocation από το λειτουργικό, που θα καταχωριστεί στη CUDA.
                host = static_cast<uint32_t*>(mmap(nullptr, bytes, PROT_READ|PROT_WRITE,
                                                 MAP_PRIVATE|MAP_ANONYMOUS, -1, 0));
                if (host == MAP_FAILED) throw std::runtime_error("mmap failed");
                // Ζητάμε τον CPU NUMA node πριν από το πρώτο γράψιμο (first touch).
                // Καταγράφουμε αν το mbind πέτυχε· η πραγματική θέση ελέγχεται στο στάδιο 5.
                std::vector<unsigned long> mask(node/(8*sizeof(unsigned long))+1, 0);
                mask[node/(8*sizeof(unsigned long))] |= 1UL << (node%(8*sizeof(unsigned long)));
                bound = syscall(SYS_mbind, host, bytes, MPOL_BIND, mask.data(),
                                node+1UL, 0) == 0;
                std::cout << "MBIND success=" << bound << " errno=" << (bound ? 0 : errno) << std::endl;
            }
            // ΣΤΑΔΙΟ 1Β — Η CPU γεμίζει την πηγή με τις γνωστές τιμές value(i).
            for (size_t i = 0; i < count; ++i) host[i] = value(i);
            // Καταχώριση της mmap allocation και ανάκτηση GPU-visible διεύθυνσης.
            // Το src δείχνει στην ίδια host allocation· δεν αντιγράφουμε είσοδο σε HBM.
            if (source == "registered") check(cudaHostRegister(host, bytes, cudaHostRegisterMapped));
            check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&src), host, 0));
        }
        // Έλεγχος CUDA memory type, χωριστός από τον έλεγχο φυσικής NUMA θέσης.
        cudaPointerAttributes attributes{}; check(cudaPointerGetAttributes(&attributes, src));
        if ((host && attributes.type != cudaMemoryTypeHost) ||
            (!host && attributes.type != cudaMemoryTypeDevice))
            throw std::runtime_error("Unexpected source memory type");
        std::cout << "POINTER cuda_type=" << int(attributes.type) << " device=" << attributes.device << std::endl;
        Residency before, after;
        // ΣΤΑΔΙΟ 5Α — Πρώτη μέτρηση της θέσης όλων των host σελίδων, πριν τους kernels.
        if (host) { before = residency(host, bytes, node); print_residency("before", before, node); }
        // Ξεχωριστό output στην HBM. Η host είσοδος παραμένει η πηγή src.
        check(cudaMalloc(reinterpret_cast<void**>(&out), bytes));
        unsigned blocks = unsigned(count / TILE);
        // ΣΤΑΔΙΟ 2 — Πρώτα ελέγχουμε την ΙΔΙΑ allocation με κανονικά GPU loads.
        // Περιμένουμε την ολοκλήρωση και επαληθεύουμε το αποτέλεσμα πριν συνεχίσουμε.
        stage("ordinary_load_control");
        check(cudaMemset(out, 0, bytes));
        load_copy<<<blocks, THREADS>>>(src, out);
        check(cudaGetLastError()); check(cudaDeviceSynchronize()); verify(out, count);
        CUtensorMap map{};
        if (method == "tma") {
            // ΣΤΑΔΙΟ 3 — Προετοιμασία: ο descriptor περιγράφει το ίδιο src και tile 32x32.
            // Το encoding περιγράφει τη μεταφορά· η εκτέλεσή της γίνεται στην tma_copy().
            stage("tensor_map_encode");
            uint64_t dimensions[] = {WIDTH, count/WIDTH}, strides[] = {WIDTH*sizeof(uint32_t)};
            uint32_t box[] = {WIDTH, WIDTH}, element_strides[] = {1, 1};
            CUresult e = cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_UINT32, 2, src,
                dimensions, strides, box, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
            if (e != CUDA_SUCCESS) {
                const char* name = nullptr; const char* message = nullptr;
                cuGetErrorName(e, &name); cuGetErrorString(e, &message);
                throw std::runtime_error(std::string(name ? name : "driver_error") + ": " + (message ? message : ""));
            }
        }
        auto launch = [&]() {
            // Επιλέγουμε TMA ή ordinary-load kernel για την τρέχουσα δοκιμή.
            if (method == "tma") tma_copy<<<blocks, THREADS>>>(map, out);
            else load_copy<<<blocks, THREADS>>>(src, out);
            check(cudaGetLastError());
        };
        stage("correctness");
        // ΣΤΑΔΙΑ 3-4 — Σβήνουμε το παλιό output, εκτελούμε τον επιλεγμένο kernel,
        // περιμένουμε και συγκρίνουμε όλα τα στοιχεία. Το προηγούμενο control
        // δεν μπορεί έτσι να αφήσει σωστό output που θα κάλυπτε αποτυχία εγγραφής.
        check(cudaMemset(out, 0, bytes)); launch(); check(cudaDeviceSynchronize()); verify(out, count);
        // Πρόσθετος χρονισμός του kernel: επαναχρησιμοποιεί την ίδια πηγή,
        // άρα οι αναγνώσεις μπορεί να εξυπηρετούνται από GPU caches.
        stage("timing");
        cudaEvent_t begin, end; check(cudaEventCreate(&begin)); check(cudaEventCreate(&end));
        std::vector<float> samples;
        for (int i = 0; i < repetitions; ++i) {
            check(cudaEventRecord(begin)); launch(); check(cudaEventRecord(end));
            check(cudaEventSynchronize(end)); float ms = 0;
            check(cudaEventElapsedTime(&ms, begin, end)); samples.push_back(ms);
        }
        // ΣΤΑΔΙΟ 4 — Επαναλαμβάνουμε τη σύγκριση και μετά τις timed επαναλήψεις.
        verify(out, count);
        // ΣΤΑΔΙΟ 5Β — Δεύτερη μέτρηση σελίδων μετά τους kernels.
        // Τα δύο checkpoints δεν αποτελούν συνεχή παρακολούθηση ή μέτρηση DRAM traffic.
        if (host) { after = residency(host, bytes, node); print_residency("after", after, node); }
        std::sort(samples.begin(), samples.end());
        float median = samples[samples.size()/2];
        stage("cleanup");
        check(cudaEventDestroy(begin)); check(cudaEventDestroy(end)); check(cudaFree(out));
        if (source == "hbm") check(cudaFree(src));
        else if (source == "pinned") check(cudaFreeHost(host));
        else { check(cudaHostUnregister(host)); munmap(host, bytes); }
        // Επιτυχία αντιγραφής μόνο αφού περάσουν οι συγκρίσεις και οι CUDA έλεγχοι.
        // Το CPU residency αναφέρεται χωριστά και απαιτεί επιτυχία και στα δύο checkpoints.
        std::cout << "RESULT {\"status\":\"correctness_pass\",\"source\":\"" << source
            << "\",\"method\":\"" << method << "\",\"bytes\":" << bytes
            << ",\"repetitions\":" << repetitions << ",\"median_kernel_ms\":" << median
            << ",\"cpu_node\":" << node << ",\"mbind_success\":" << (bound ? "true" : "false")
            << ",\"cpu_residency_verified_before_and_after\":"
            << ((host && before.verified() && after.verified()) ? "true" : "false") << "}" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR " << e.what() << std::endl;
        return 2; // Worker exit releases allocations even when cleanup was interrupted.
    }
}
