// Ελέγχουμε αν η GPU διαβάζει σωστά μνήμη της CPU με απλά loads και με TMA.
// Η tma_copy() και ο συγχρονισμός της βασίζονται στο παράδειγμα της NVIDIA,
// CUDA 12.8.1 Programming Guide, ενότητα 7.29.2:
// https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#using-tma-to-transfer-multi-dimensional-arrays
// Εδώ προσθέσαμε host allocations, σύγκριση τιμών και έλεγχο της θέσης των σελίδων.
// Οι κλήσεις για πρόσβαση της GPU στη host μνήμη περιγράφονται εδώ:
// https://docs.nvidia.com/cuda/archive/12.8.1/cuda-c-programming-guide/index.html#mapped-memory
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

// Με το barrier περιμένουν μαζί τα GPU threads του ίδιου GPU block.
using Barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;
// Ένα GPU block δουλεύει σε ένα tile 32x32, με 128 GPU threads.
constexpr int WIDTH = 32, TILE = 1024, THREADS = 128;

void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
void stage(const char* name) { std::cout << "STAGE " << name << std::endl; }
// Στάδιο 1: βγάζουμε πάντα την ίδια τιμή για κάθε θέση i.
// Έτσι ξέρουμε τι πρέπει να βρούμε στο output.
uint32_t value(size_t i) { return uint32_t(i * 1664525ULL + 1013904223ULL); }

// Στάδιο 2: τα GPU threads διαβάζουν από το src με κανονικά loads.
// Το src μπορεί να δείχνει σε host μνήμη ή σε HBM. Το output είναι στην HBM.
__global__ void load_copy(const uint32_t* src, uint32_t* out) {
    size_t base = size_t(blockIdx.x) * TILE;
    for (int i = threadIdx.x; i < TILE; i += blockDim.x) out[base+i] = src[base+i];
}

// Στάδιο 3: το TMA φέρνει ένα tile στη shared memory της GPU.
__global__ void tma_copy(const __grid_constant__ CUtensorMap map, uint32_t* out) {
    // Προσωρινός χώρος για 1024 αριθμούς, δηλαδή 4 KiB.
    // Η αρχή του πρέπει να είναι σε διεύθυνση πολλαπλάσια των 128 bytes.
    __shared__ alignas(128) uint32_t tile[TILE];
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ Barrier barrier;
    if (threadIdx.x == 0) {
        // Το GPU thread 0 ετοιμάζει το barrier για όλα τα GPU threads του GPU block.
        init(&barrier, blockDim.x);
        // Κάνουμε την αρχικοποίηση ορατή και στον μηχανισμό TMA.
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();
    Barrier::arrival_token token;
    if (threadIdx.x == 0) {
        // Το GPU thread 0 ζητά τη μεταφορά ολόκληρου του tile.
        cde::cp_async_bulk_tensor_2d_global_to_shared(tile, &map, 0, blockIdx.x * WIDTH, barrier);
        // Δηλώνει ότι έφτασε στο barrier και ότι περιμένουμε 4096 bytes.
        token = cuda::device::barrier_arrive_tx(barrier, 1, sizeof(tile));
    } else {
        // Τα υπόλοιπα GPU threads δηλώνουν ότι έφτασαν και αυτά στο barrier.
        token = barrier.arrive();
    }
    // Κανένα GPU thread δεν διαβάζει το tile πριν ολοκληρωθεί η μεταφορά.
    barrier.wait(std::move(token));
    size_t base = size_t(blockIdx.x) * TILE;
    // Τα GPU threads γράφουν το tile στο output, για να το ελέγξει μετά η CPU.
    for (int i = threadIdx.x; i < TILE; i += blockDim.x) out[base+i] = tile[i];
    __syncthreads();
    if (threadIdx.x == 0) barrier.~Barrier();
}

struct Residency {
    long pages = 0, on_cpu_node = 0, unknown = 0;
    int query_errno = 0;
    bool verified() const { return pages > 0 && on_cpu_node == pages && unknown == 0; }
};

// Στάδιο 5: βρίσκουμε σε ποιον NUMA node είναι οι σελίδες της εισόδου.
// Συγκρίνουμε τον αριθμό του node με αυτόν της Grace CPU.
// Εδώ ρωτάμε μόνο τη θέση των σελίδων, χωρίς να ζητάμε μετακίνησή τους.
// Αν το Linux δεν απαντήσει, κρατάμε τη θέση ως άγνωστη.
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

// Στάδιο 4: φέρνουμε το output στην CPU και συγκρίνουμε κάθε τιμή με τη value(i).
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
        // Κρατάμε τον NUMA node όπου εκτελείται τώρα το CPU thread.
        std::cout << "CPU node=" << node << " cpu=" << cpu << std::endl;
        stage("allocation");
        uint32_t *host = nullptr, *src = nullptr, *out = nullptr;
        bool bound = false;
        size_t count = bytes / sizeof(uint32_t);
        if (source == "hbm") {
            // Δοκιμή αναφοράς με είσοδο στην HBM: ετοιμάζουμε τιμές στην CPU
            // και τις αντιγράφουμε σε ξεχωριστή GPU allocation.
            std::vector<uint32_t> initial(count);
            for (size_t i = 0; i < count; ++i) initial[i] = value(i);
            check(cudaMalloc(reinterpret_cast<void**>(&src), bytes));
            check(cudaMemcpy(src, initial.data(), bytes, cudaMemcpyHostToDevice));
        } else {
            // Στάδιο 1: δεσμεύουμε χώρο για την είσοδο στην πλευρά της CPU.
            if (source == "pinned") {
                // Ζητάμε από τη CUDA host μνήμη που θα μπορεί να διαβάζει η GPU.
                check(cudaHostAlloc(reinterpret_cast<void**>(&host), bytes, cudaHostAllocMapped));
            } else {
                // Εδώ δεσμεύουμε πρώτα τη μνήμη μέσω Linux.
                host = static_cast<uint32_t*>(mmap(nullptr, bytes, PROT_READ|PROT_WRITE,
                                                 MAP_PRIVATE|MAP_ANONYMOUS, -1, 0));
                if (host == MAP_FAILED) throw std::runtime_error("mmap failed");
                // Ζητάμε τον CPU NUMA node πριν γράψουμε τιμές στη μνήμη.
                // Αν αποτύχει, το καταγράφουμε. Αργότερα ελέγχουμε πού μπήκαν οι σελίδες.
                std::vector<unsigned long> mask(node/(8*sizeof(unsigned long))+1, 0);
                mask[node/(8*sizeof(unsigned long))] |= 1UL << (node%(8*sizeof(unsigned long)));
                bound = syscall(SYS_mbind, host, bytes, MPOL_BIND, mask.data(),
                                mask.size()*8*sizeof(unsigned long)+1UL, 0) == 0;
                std::cout << "MBIND success=" << bound << " errno=" << (bound ? 0 : errno) << std::endl;
            }
            // Το CPU thread γράφει τις γνωστές τιμές στην είσοδο.
            for (size_t i = 0; i < count; ++i) host[i] = value(i);
            // Δηλώνουμε στη CUDA τη μνήμη που πήραμε με mmap.
            if (source == "registered") check(cudaHostRegister(host, bytes, cudaHostRegisterMapped));
            // Παίρνουμε διεύθυνση που μπορεί να χρησιμοποιήσει ο GPU kernel.
            // Είναι η ίδια host μνήμη, χωρίς νέο αντίγραφο της εισόδου στην HBM.
            check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&src), host, 0));
        }
        // Ελέγχουμε αν η CUDA βλέπει host ή device μνήμη στη διεύθυνση src.
        // Αυτό από μόνο του δεν μας λέει σε ποιον NUMA node είναι οι σελίδες.
        cudaPointerAttributes attributes{}; check(cudaPointerGetAttributes(&attributes, src));
        if ((host && attributes.type != cudaMemoryTypeHost) ||
            (!host && attributes.type != cudaMemoryTypeDevice))
            throw std::runtime_error("Unexpected source memory type");
        std::cout << "POINTER cuda_type=" << int(attributes.type) << " device=" << attributes.device << std::endl;
        Residency before, after;
        // Στάδιο 5, πριν: ρωτάμε πού είναι οι host σελίδες πριν διαβάσει η GPU.
        if (host) { before = residency(host, bytes, node); print_residency("before", before, node); }
        // Δεσμεύουμε ξεχωριστό output στην HBM. Δεν αλλάζουμε την είσοδο src.
        check(cudaMalloc(reinterpret_cast<void**>(&out), bytes));
        unsigned blocks = unsigned(count / TILE);
        // Στάδιο 2: δοκιμάζουμε πρώτα κανονικά GPU loads στην ίδια είσοδο.
        // Ελέγχουμε τις τιμές πριν προχωρήσουμε στη δοκιμή που ζητήθηκε.
        stage("ordinary_load_control");
        check(cudaMemset(out, 0, bytes));
        load_copy<<<blocks, THREADS>>>(src, out);
        check(cudaGetLastError()); check(cudaDeviceSynchronize()); verify(out, count);
        CUtensorMap map{};
        if (method == "tma") {
            // Στάδιο 3: λέμε στο TMA πού είναι η είσοδος και τι σχήμα έχει.
            // Ο descriptor είναι αυτή η περιγραφή. Δεν μεταφέρει ο ίδιος δεδομένα.
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
            // Ξεκινάμε τον GPU kernel που αντιστοιχεί στη δοκιμή: TMA ή απλά loads.
            if (method == "tma") tma_copy<<<blocks, THREADS>>>(map, out);
            else load_copy<<<blocks, THREADS>>>(src, out);
            check(cudaGetLastError());
        };
        stage("correctness");
        // Στάδια 3-4: σβήνουμε το παλιό output, τρέχουμε τον GPU kernel και ελέγχουμε τις τιμές.
        // Το σβήσιμο χρειάζεται για να μην ελέγξουμε κατά λάθος το output της προηγούμενης δοκιμής.
        check(cudaMemset(out, 0, bytes)); launch(); check(cudaDeviceSynchronize()); verify(out, count);
        // Μετράμε τον χρόνο του GPU kernel με την ίδια είσοδο σε κάθε επανάληψη.
        // Κάποια δεδομένα μπορεί να είναι ήδη στη GPU cache.
        stage("timing");
        cudaEvent_t begin, end; check(cudaEventCreate(&begin)); check(cudaEventCreate(&end));
        std::vector<float> samples;
        for (int i = 0; i < repetitions; ++i) {
            check(cudaEventRecord(begin)); launch(); check(cudaEventRecord(end));
            check(cudaEventSynchronize(end)); float ms = 0;
            check(cudaEventElapsedTime(&ms, begin, end)); samples.push_back(ms);
        }
        // Στάδιο 4: ελέγχουμε ξανά τις τιμές μετά τις επαναλήψεις.
        verify(out, count);
        // Στάδιο 5, μετά: ρωτάμε ξανά πού είναι οι host σελίδες.
        // Οι δύο έλεγχοι δεν δείχνουν αν κάθε ανάγνωση ήρθε από DRAM ή από cache.
        if (host) { after = residency(host, bytes, node); print_residency("after", after, node); }
        std::sort(samples.begin(), samples.end());
        float median = samples[samples.size()/2];
        stage("cleanup");
        check(cudaEventDestroy(begin)); check(cudaEventDestroy(end)); check(cudaFree(out));
        if (source == "hbm") check(cudaFree(src));
        else if (source == "pinned") check(cudaFreeHost(host));
        else { check(cudaHostUnregister(host)); munmap(host, bytes); }
        // Φτάνουμε εδώ μόνο αν ολοκληρώθηκαν οι CUDA κλήσεις και οι συγκρίσεις.
        // Γράφουμε χωριστά αν βρήκαμε τις host σελίδες στον CPU node και στους δύο ελέγχους.
        std::cout << "RESULT {\"status\":\"correctness_pass\",\"source\":\"" << source
            << "\",\"method\":\"" << method << "\",\"bytes\":" << bytes
            << ",\"repetitions\":" << repetitions << ",\"median_kernel_ms\":" << median
            << ",\"cpu_node\":" << node << ",\"mbind_success\":" << (bound ? "true" : "false")
            << ",\"cpu_residency_verified_before_and_after\":"
            << ((host && before.verified() && after.verified()) ? "true" : "false") << "}" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR " << e.what() << std::endl;
        return 2; // Με τον τερματισμό της διεργασίας αποδεσμεύονται και οι allocations της.
    }
}
