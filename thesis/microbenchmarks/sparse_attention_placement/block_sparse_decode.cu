// One query per head. Read selected KV blocks directly, then merge their
// softmax partials. Grace DRAM and HBM use the same kernel and launch geometry.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {
constexpr int head_dim = 128;
constexpr int block_tokens = 128;
constexpr int max_selected = 64;
constexpr int warp = 32;
constexpr int threads = 128;

__global__ void block_partials(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    const int32_t* selected, int tokens, int count,
    float* partial_max, float* partial_sum, float* partial_v) {
    const int part = blockIdx.x;
    const int head = blockIdx.y;
    const int dim = threadIdx.x;
    const int lane = dim % warp;
    const int warp_id = dim / warp;
    const int start = selected[part] * block_tokens;
    const int row = (head * tokens + start) * head_dim;
    const int partial = head * count + part;
    __shared__ float weights[block_tokens];

    const auto* query = q + head * head_dim;
    float query_values[head_dim / warp];
    #pragma unroll
    for (int component = 0; component < head_dim / warp; ++component) {
        query_values[component] = __bfloat162float(query[component * warp + lane]);
    }

    for (int token = warp_id; token < block_tokens; token += threads / warp) {
        const auto* key = k + row + token * head_dim;
        float dot = 0.0f;
        #pragma unroll
        for (int component = 0; component < head_dim / warp; ++component) {
            dot = fmaf(query_values[component], __bfloat162float(key[component * warp + lane]), dot);
        }
        for (int shift = warp / 2; shift > 0; shift /= 2) {
            dot += __shfl_down_sync(0xffffffff, dot, shift);
        }
        if (lane == 0) {
            weights[token] = dot * rsqrtf(float(head_dim));
        }
    }
    __syncthreads();

    if (dim == 0) {
        float maximum = weights[0];
        for (int token = 1; token < block_tokens; ++token) {
            maximum = fmaxf(maximum, weights[token]);
        }
        float total = 0.0f;
        for (int token = 0; token < block_tokens; ++token) {
            const float weight = expf(weights[token] - maximum);
            weights[token] = weight;
            total += weight;
        }
        partial_max[partial] = maximum;
        partial_sum[partial] = total;
    }
    __syncthreads();

    float weighted_v = 0.0f;
    for (int token = 0; token < block_tokens; ++token) {
        weighted_v = fmaf(weights[token], __bfloat162float(v[row + token * head_dim + dim]),
                          weighted_v);
    }
    partial_v[partial * head_dim + dim] = weighted_v;
}

__global__ void merge_partials(const float* partial_max, const float* partial_sum,
                               const float* partial_v, int count, float* out) {
    const int head = blockIdx.x;
    const int dim = threadIdx.x;
    const int offset = head * count;
    __shared__ float scale[max_selected];
    if (dim == 0) {
        float maximum = partial_max[offset];
        for (int part = 1; part < count; ++part) {
            maximum = fmaxf(maximum, partial_max[offset + part]);
        }
        float denominator = 0.0f;
        for (int part = 0; part < count; ++part) {
            const float factor = expf(partial_max[offset + part] - maximum);
            scale[part] = factor;
            denominator = fmaf(factor, partial_sum[offset + part], denominator);
        }
        for (int part = 0; part < count; ++part) {
            scale[part] /= denominator;
        }
    }
    __syncthreads();

    float value = 0.0f;
    for (int part = 0; part < count; ++part) {
        value = fmaf(scale[part], partial_v[(offset + part) * head_dim + dim], value);
    }
    out[head * head_dim + dim] = value;
}
}  // namespace

namespace {
cudaError_t launch_decode(const void* q, const void* k, const void* v,
                             const int32_t* selected, int heads, int tokens, int count,
                             float* partial_max, float* partial_sum, float* partial_v,
                             float* output, cudaStream_t stream) {
    if (heads <= 0 || tokens <= 0 || tokens % block_tokens != 0
        || count <= 0 || count > max_selected || count > tokens / block_tokens) {
        return cudaErrorInvalidValue;
    }
    block_partials<<<dim3(count, heads), threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q), static_cast<const __nv_bfloat16*>(k),
        static_cast<const __nv_bfloat16*>(v), selected, tokens, count, partial_max,
        partial_sum, partial_v);
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }
    merge_partials<<<heads, threads, 0, stream>>>(
        partial_max, partial_sum, partial_v, count, output);
    return cudaGetLastError();
}

class Event {
public:
    cudaEvent_t handle = nullptr;
    ~Event() {
        if (handle != nullptr) {
            cudaEventDestroy(handle);
        }
    }
};
}  // namespace

extern "C" int sparse_decode(const void* q, const void* k, const void* v,
                             const int32_t* selected, int heads, int tokens, int count,
                             float* partial_max, float* partial_sum, float* partial_v,
                             float* output, cudaStream_t stream, float* measured_ms) {
    Event start;
    Event end;
    cudaError_t status = cudaSuccess;
    if (measured_ms) {
        if ((status = cudaEventCreate(&start.handle)) != cudaSuccess
            || (status = cudaEventCreate(&end.handle)) != cudaSuccess
            || (status = cudaEventRecord(start.handle, stream)) != cudaSuccess) {
            return status;
        }
    }

    status = launch_decode(q, k, v, selected, heads, tokens, count,
                           partial_max, partial_sum, partial_v, output, stream);
    if (status != cudaSuccess || !measured_ms) {
        return status;
    }
    if ((status = cudaEventRecord(end.handle, stream)) != cudaSuccess
        || (status = cudaEventSynchronize(end.handle)) != cudaSuccess) {
        return status;
    }
    return cudaEventElapsedTime(measured_ms, start.handle, end.handle);
}
