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

template<int H, int W>
void measure(Memory& m, int threads, int rotation, unsigned node, std::array<Row,3>& rows) {
    constexpr int elements=H*W, shared_bytes=elements*4;
    unsigned blocks=unsigned(m.bytes/shared_bytes);
    const char* warm_text=std::getenv("TMA_SAMPLE_WARMUP");
    const char* sample_text=std::getenv("TMA_SAMPLE_COUNT");
    int warmup=warm_text ? std::stoi(warm_text) : 10;
    int samples=sample_text ? std::stoi(sample_text) : 30;
    if (warmup<1 || samples<6 || samples%6)
        throw std::runtime_error("Require warmup>=1 and samples>=6, divisible by 6");
    if (shared_bytes>48*1024) {
        check(cudaFuncSetAttribute(loads_shared<elements>,cudaFuncAttributeMaxDynamicSharedMemorySize,shared_bytes));
        check(cudaFuncSetAttribute(tma_shared<H,W>,cudaFuncAttributeMaxDynamicSharedMemorySize,shared_bytes));
    }
    CUtensorMap map{};
    uint64_t dims[]={W,m.bytes/(4*W)}, strides[]={W*4};
    uint32_t box[]={W,H}, steps[]={1,1};
    if (reinterpret_cast<uintptr_t>(m.src)%16)
        throw std::runtime_error("TMA input is not 16-byte aligned");
    CUresult encoded=cuTensorMapEncodeTiled(&map,CU_TENSOR_MAP_DATA_TYPE_UINT32,2,
        m.src,dims,strides,box,steps,CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (encoded!=CUDA_SUCCESS)
        throw std::runtime_error("Tensor map encoding failed: "+std::to_string(encoded));
    auto launch=[&](int method) {
        if (method==0) loads_direct<elements><<<blocks,threads>>>(m.src,m.out);
        if (method==1) loads_shared<elements><<<blocks,threads,shared_bytes>>>(m.src,m.out);
        if (method==2) tma_shared<H,W><<<blocks,threads,shared_bytes>>>(map,m.out);
        check(cudaGetLastError());
    };
    // Prime event creation/record/synchronization before any timed samples.
    Events events;
    check(cudaEventCreate(&events.begin));
    check(cudaEventCreate(&events.end));
    check(cudaEventRecord(events.begin));
    check(cudaEventRecord(events.end));
    check(cudaEventSynchronize(events.end));
    for (auto& row:rows) { row.status="running"; row.warmup=warmup; }
    if (m.host) {
        stage("placement_before_warmup");
        Location location=locate(m.host,m.bytes,node);
        for (auto& row:rows) row.before=location;
    }
    // Pre/post correctness checks are outside the timed samples, one per kernel.
    stage("correctness_before_sampling");
    for (int method=0; method<3; ++method) {
        check(cudaMemset(m.out,0,m.bytes));
        launch(method);
        check(cudaDeviceSynchronize());
        verify(m);
    }
    stage("warmup");
    for (int round=0; round<warmup; ++round)
        for (int position=0; position<3; ++position)
            launch(sampled_orders[(round+rotation)%6][position]);
    check(cudaDeviceSynchronize());
    if (m.host) {
        stage("placement_after_warmup_before_samples");
        Location location=locate(m.host,m.bytes,node);
        for (auto& row:rows) row.before_samples=location;
    }
    stage("samples");
    for (int sample=0; sample<samples; ++sample) {
        for (int position=0; position<3; ++position) {
            int method=sampled_orders[(sample+rotation)%6][position];
            check(cudaEventRecord(events.begin));
            launch(method);
            check(cudaEventRecord(events.end));
            check(cudaEventSynchronize(events.end));
            float ms=0;
            check(cudaEventElapsedTime(&ms,events.begin,events.end));
            if (!(ms>0) || !std::isfinite(ms)) throw std::runtime_error("Invalid sample time");
            rows[method].samples.push_back(ms);
            rows[method].positions.push_back(position);
        }
    }
    if (m.host) {
        stage("placement_after_samples_before_verification");
        Location location=locate(m.host,m.bytes,node);
        for (auto& row:rows) row.after=location;
    }
    stage("correctness_after_sampling");
    for (int method=0; method<3; ++method) {
        check(cudaMemset(m.out,0,m.bytes));
        launch(method);
        check(cudaDeviceSynchronize());
        verify(m);
    }
    if (m.host) {
        stage("placement_after_verification");
        Location location=locate(m.host,m.bytes,node);
        for (auto& row:rows) row.after_verification=location;
    }
    for (auto& row:rows) {
        row.ms=float(sample_quantile(row.samples,0.5));
        row.order=row.positions.front();
        row.status="ok";
    }
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
