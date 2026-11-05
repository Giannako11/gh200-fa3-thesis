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
