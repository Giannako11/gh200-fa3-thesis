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
