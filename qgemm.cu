#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <vector>
#include <stdint.h>
#include <assert.h>

// ------------------- Constants -------------------
#define TILE_SIZE 32
#define REG_TILE_N 4
#define REG_TILE_M 4

// ------------------- User kernels (mostly as provided) -------------------

__global__ void absmax_rowwise_kernel(const float* __restrict__ A, float* rowScales, int M, int K) {
    int m = blockIdx.x;
    if (m < M) {
        extern __shared__ float sdata[];
        int tid = threadIdx.x;
        float local_max = 0.f;
        for (int k = tid; k < K; k += blockDim.x) {
            float val = fabsf(A[m * K + k]);
            local_max = fmaxf(local_max, val);
        }
        sdata[tid] = local_max;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
            }
            __syncthreads();
        }
        if (tid == 0) {
            rowScales[m] = fmaxf(sdata[0] / 127.f, 1e-8f);
        }
    }
}

__global__ void colwise_minmax(const float* __restrict__ A, float* colMins, float* colMaxs, int M, int N) {
    int n = blockIdx.x;
    if (n < N) {
        extern __shared__ float sdata[];
        float* smins = sdata;
        float* smaxs = sdata + blockDim.x;

        int tid = threadIdx.x;
        float local_min = 1e30f;
        float local_max = -1e30f;
        for (int i = tid; i < M; i += blockDim.x) {
            float val = A[i * N + n];
            local_min = fminf(local_min, val);
            local_max = fmaxf(local_max, val);
        }
        smins[tid] = local_min;
        smaxs[tid] = local_max;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                smins[tid] = fminf(smins[tid], smins[tid + s]);
                smaxs[tid] = fmaxf(smaxs[tid], smaxs[tid + s]);
            }
            __syncthreads();
        }
        if (tid == 0) {
            colMins[n] = smins[0];
            colMaxs[n] = smaxs[0];
        }
    }
}

__global__ void quantize_weights_rowwise(const float* __restrict__ A, int8_t* Aq,
                                        const float* __restrict__ rowScales, int M, int K) {
    int m = blockIdx.x;
    int k = threadIdx.x;
    if (m < M && k < K) {
        float scale = rowScales[m];
        float val = A[m * K + k] / scale;
        val = fmaxf(fminf(val, 127.f), -127.f);
        Aq[m * K + k] = static_cast<int8_t>(rintf(val));
    }
}

__global__ void quantize_activations_colwise(const float* __restrict__ X, uint8_t* Xq,
                                            const float* __restrict__ colMins,
                                            const float* __restrict__ colMaxs,
                                            float* colScales, int* colZPs,
                                            int M, int N) {
    int n = blockIdx.x;
    int m = threadIdx.x;
    if (n < N && m < M) {
        float minv = colMins[n];
        float maxv = colMaxs[n];
        float scale = fmaxf((maxv - minv) / 255.f, 1e-8f);
        int zp = static_cast<int>(rintf(-minv / scale));
        colScales[n] = scale;
        colZPs[n] = zp;

        float val = X[m * N + n] / scale + zp;
        val = fmaxf(fminf(val, 255.f), 0.f);
        Xq[m * N + n] = static_cast<uint8_t>(rintf(val));
    }
}

__global__ void qgemm_kernel(const int8_t* __restrict__ Wq, const float* __restrict__ Sw,
                            const uint8_t* __restrict__ Xq, const float* __restrict__ Sx,
                            const int* __restrict__ Zx, float* Y, int M, int K, int N) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m < M && n < N) {
        int32_t acc = 0;
        for (int k = 0; k < K; k++) {
            int32_t w = static_cast<int32_t>(Wq[m * K + k]);
            int32_t x = static_cast<int32_t>(Xq[k * N + n]) - Zx[n];
            acc += w * x;
        }
        float result = acc * (Sw[m] * Sx[n]);
        Y[m * N + n] = result;
    }
}

// ------------------- C++ / PyTorch wrappers -------------------

inline void check_contiguous_cuda(const torch::Tensor &t, const char* name) {
    TORCH_CHECK(t.device().is_cuda(), "%s must be a CUDA tensor", name);
    TORCH_CHECK(t.is_contiguous(), "%s must be contiguous", name);
}

torch::Tensor absmax_rowwise(torch::Tensor A) {
    check_contiguous_cuda(A, "A");
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    int64_t M = A.size(0);
    int64_t K = A.size(1);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(A.device());
    torch::Tensor rowScales = torch::empty({M}, options);

    // launch kernel: grid.x = M, block.x = min(1024, K) but >= 1
    int threads = static_cast<int>(std::min<int64_t>(1024, K > 0 ? K : 1));
    if (threads < 1) threads = 1;
    dim3 blocks(static_cast<uint32_t>(M));
    size_t shared_bytes = threads * sizeof(float);

    float* A_ptr = A.data_ptr<float>();
    float* out_ptr = rowScales.data_ptr<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    absmax_rowwise_kernel<<<blocks, threads, shared_bytes, stream>>>(A_ptr, out_ptr, (int)M, (int)K);
    AT_CUDA_CHECK(cudaGetLastError());
    return rowScales;
}

std::vector<torch::Tensor> colwise_minmax_wrapper(torch::Tensor A) {
    check_contiguous_cuda(A, "A");
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    int64_t M = A.size(0);
    int64_t N = A.size(1);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(A.device());
    torch::Tensor colMins = torch::empty({N}, options);
    torch::Tensor colMaxs = torch::empty({N}, options);

    int threads = static_cast<int>(std::min<int64_t>(1024, M > 0 ? M : 1));
    if (threads < 1) threads = 1;
    dim3 blocks(static_cast<uint32_t>(N));
    size_t shared_bytes = threads * 2 * sizeof(float);

    float* A_ptr = A.data_ptr<float>();
    float* mins_ptr = colMins.data_ptr<float>();
    float* maxs_ptr = colMaxs.data_ptr<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    colwise_minmax<<<blocks, threads, shared_bytes, stream>>>(A_ptr, mins_ptr, maxs_ptr, (int)M, (int)N);
    AT_CUDA_CHECK(cudaGetLastError());
    return {colMins, colMaxs};
}

torch::Tensor quantize_weights_rowwise_wrapper(torch::Tensor A, torch::Tensor rowScales) {
    check_contiguous_cuda(A, "A");
    check_contiguous_cuda(rowScales, "rowScales");
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(rowScales.dim() == 1, "rowScales must be 1D");
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    TORCH_CHECK(rowScales.size(0) == M, "rowScales length must equal M");

    auto options = torch::TensorOptions().dtype(torch::kInt8).device(A.device());
    torch::Tensor Aq = torch::empty({M, K}, options);

    int threads = static_cast<int>(std::min<int64_t>(1024, K > 0 ? K : 1));
    if (threads < 1) threads = 1;
    dim3 blocks(static_cast<uint32_t>(M));

    float* A_ptr = A.data_ptr<float>();
    int8_t* Aq_ptr = reinterpret_cast<int8_t*>(Aq.data_ptr<int8_t>());
    float* scales_ptr = rowScales.data_ptr<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    quantize_weights_rowwise<<<blocks, threads, 0, stream>>>(A_ptr, Aq_ptr, scales_ptr, (int)M, (int)K);
    AT_CUDA_CHECK(cudaGetLastError());
    return Aq;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> quantize_activations_colwise_wrapper(torch::Tensor X) {
    check_contiguous_cuda(X, "X");
    TORCH_CHECK(X.dim() == 2, "X must be 2D");
    int64_t M = X.size(0);
    int64_t N = X.size(1);

    auto options_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(X.device());
    auto options_f = torch::TensorOptions().dtype(torch::kFloat32).device(X.device());
    auto options_i = torch::TensorOptions().dtype(torch::kInt32).device(X.device());

    torch::Tensor Xq = torch::empty({M, N}, options_u8);
    torch::Tensor colScales = torch::empty({N}, options_f);
    torch::Tensor colZPs = torch::empty({N}, options_i);

    int threads = static_cast<int>(std::min<int64_t>(1024, M > 0 ? M : 1));
    if (threads < 1) threads = 1;
    dim3 blocks(static_cast<uint32_t>(N));

    float* X_ptr = X.data_ptr<float>();
    uint8_t* Xq_ptr = Xq.data_ptr<uint8_t>();
    float* mins_ptr; // we need mins/maxs to compute scales -> call colwise_minmax first
    float* maxs_ptr;

    // compute col mins/maxs
    torch::Tensor colMins = torch::empty({N}, options_f);
    torch::Tensor colMaxs = torch::empty({N}, options_f);
    {
        int threads_mm = static_cast<int>(std::min<int64_t>(1024, M > 0 ? M : 1));
        size_t shared_bytes = threads_mm * 2 * sizeof(float);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
        colwise_minmax<<<dim3((uint32_t)N), threads_mm, shared_bytes, stream>>>(X_ptr, colMins.data_ptr<float>(), colMaxs.data_ptr<float>(), (int)M, (int)N);
        AT_CUDA_CHECK(cudaGetLastError());
    }

    mins_ptr = colMins.data_ptr<float>();
    maxs_ptr = colMaxs.data_ptr<float>();
    float* scales_ptr = colScales.data_ptr<float>();
    int* zps_ptr = colZPs.data_ptr<int>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    quantize_activations_colwise<<<dim3((uint32_t)N), threads, 0, stream>>>(X_ptr, Xq_ptr, mins_ptr, maxs_ptr, scales_ptr, zps_ptr, (int)M, (int)N);
    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(Xq, colScales, colZPs);
}

torch::Tensor qgemm_wrapper(torch::Tensor Wq, torch::Tensor Sw, torch::Tensor Xq, torch::Tensor Sx, torch::Tensor Zx, int M, int K, int N) {
    // Validate
    check_contiguous_cuda(Wq, "Wq");
    check_contiguous_cuda(Sw, "Sw");
    check_contiguous_cuda(Xq, "Xq");
    check_contiguous_cuda(Sx, "Sx");
    check_contiguous_cuda(Zx, "Zx");

    TORCH_CHECK(Wq.dim() == 2 && Wq.size(0) == M && Wq.size(1) == K, "Wq must be MxK int8");
    TORCH_CHECK(Sw.dim() == 1 && Sw.size(0) == M, "Sw must be length M");
    TORCH_CHECK(Xq.dim() == 2 && Xq.size(0) == K && Xq.size(1) == N, "Xq must be KxN uint8");
    TORCH_CHECK(Sx.dim() == 1 && Sx.size(0) == N, "Sx must be length N");
    TORCH_CHECK(Zx.dim() == 1 && Zx.size(0) == N, "Zx must be length N");

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(Wq.device());
    torch::Tensor Y = torch::zeros({M, N}, options);

    // Use simple 2D grid launch instead of register tiling for now
    dim3 block(16, 16);  // 256 threads per block
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    // pointers
    const int8_t* Wq_ptr = reinterpret_cast<const int8_t*>(Wq.data_ptr<int8_t>());
    const float* Sw_ptr = Sw.data_ptr<float>();
    const uint8_t* Xq_ptr = Xq.data_ptr<uint8_t>();
    const float* Sx_ptr = Sx.data_ptr<float>();
    const int* Zx_ptr = Zx.data_ptr<int>();
    float* Y_ptr = Y.data_ptr<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    qgemm_kernel<<<grid, block, 0, stream>>>(Wq_ptr, Sw_ptr, Xq_ptr, Sx_ptr, Zx_ptr, Y_ptr, M, K, N);
    AT_CUDA_CHECK(cudaGetLastError());
    return Y;
}

// ------------------- Python binding -------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Quantization ops (CUDA) - wrappers for custom kernels";

    m.def("absmax_rowwise", &absmax_rowwise, "Compute row-wise absolute max and produce rowScales (float32) (CUDA)");
    m.def("colwise_minmax", &colwise_minmax_wrapper, "Compute column min/max (returns (mins,maxs)) (CUDA)");
    m.def("quantize_weights_rowwise", &quantize_weights_rowwise_wrapper, "Quantize weights rowwise to int8 (CUDA)");
    m.def("quantize_activations_colwise", &quantize_activations_colwise_wrapper, "Quantize activations columnwise -> returns (Xq, colScales, colZPs) (CUDA)");
    m.def("qgemm", &qgemm_wrapper, "Quantized GEMM with register tiling (CUDA). Call as qgemm(Wq, Sw, Xq, Sx, Zx, M, K, N)");
}
