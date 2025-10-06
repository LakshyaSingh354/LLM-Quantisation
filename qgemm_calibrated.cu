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
    if (m < M) {
        float scale = rowScales[m];
        for (int k = threadIdx.x; k < K; k += blockDim.x) {
            float val = A[m * K + k] / scale;
            val = fmaxf(fminf(val, 127.f), -127.f);
            Aq[m * K + k] = static_cast<int8_t>(rintf(val));
        }
    }
}

__global__ void quantize_activations_colwise(
    const float* __restrict__ X, uint8_t* Xq,
    const float* __restrict__ calib_mins,  // Pass in pre-calculated mins
    const float* __restrict__ calib_maxs,  // Pass in pre-calculated maxs
    float* colScales, int* colZPs,
    int K, // The number of rows
    int N) // The number of columns
{
    // Each block processes one column 'n'
    int n = blockIdx.x;

    if (n < N) {
        float minv = calib_mins[n];
        float maxv = calib_maxs[n];

        if (threadIdx.x == 0) {
            float scale = fmaxf((maxv - minv) / 255.f, 1e-8f);
            int zp = static_cast<int>(roundf(-minv / scale));
            // Clamp the zero-point to be a valid uint8 value
            zp = max(0, min(255, zp));
            colScales[n] = scale;
            colZPs[n] = zp;
        }

        __syncthreads();

        float scale = colScales[n];
        int zp = colZPs[n];

        for (int m = threadIdx.x; m < K; m += blockDim.x) {
            float val = X[m * N + n];

            val = fmaxf(fminf(val, maxv), minv);

            float quantized_val = val / scale + zp;
            Xq[m * N + n] = static_cast<uint8_t>(roundf(quantized_val));
        }
    }
}

__global__ void qgemm_kernel(const int8_t* __restrict__ Wq,
                                                 const float* __restrict__ Sw,
                                                 const uint8_t* __restrict__ Xq,
                                                 const float* __restrict__ Sx,
                                                 const int* __restrict__ Zx,
                                                 float* Y, int M, int K, int N) {
    extern __shared__ int8_t smem_raw[]; // optional if dynamic shared needed
    // static shared arrays (compile-time sizes) - preferred
    __shared__ int8_t  Wq_tile[TILE_SIZE][TILE_SIZE];
    __shared__ int32_t Xq_tile[TILE_SIZE][TILE_SIZE];

    const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;

    const int base_row = by * TILE_SIZE;
    const int base_col = bx * TILE_SIZE;

    // thread-owned offset within the tile
    const int thread_row_in_tile = ty * REG_TILE_M; // 0..TILE_SIZE-REG_TILE_M
    const int thread_col_in_tile = tx * REG_TILE_N; // 0..TILE_SIZE-REG_TILE_N

    const int global_row = base_row + thread_row_in_tile;
    const int global_col = base_col + thread_col_in_tile;

    int32_t acc[REG_TILE_M][REG_TILE_N];
    #pragma unroll
    for (int i = 0; i < REG_TILE_M; ++i)
        #pragma unroll
        for (int j = 0; j < REG_TILE_N; ++j)
            acc[i][j] = 0;

    int8_t w_reg[REG_TILE_M];

    const int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        // how many K values are valid in this tile
        const int K_tile_valid = min(TILE_SIZE, K - t * TILE_SIZE);

        // Cooperative load: each thread writes REG_TILE_M x REG_TILE_N entries
        #pragma unroll
        for (int lm = 0; lm < REG_TILE_M; ++lm) {
            #pragma unroll
            for (int ln = 0; ln < REG_TILE_N; ++ln) {
                int tile_r = thread_row_in_tile + lm;   // 0..TILE_SIZE-1
                int tile_c = thread_col_in_tile + ln;   // 0..TILE_SIZE-1

                int Wq_row = base_row + tile_r;         // global m
                int Wq_col = t * TILE_SIZE + tile_c;    // global k

                if (Wq_row < M && Wq_col < K) {
                    Wq_tile[tile_r][tile_c] = Wq[Wq_row * K + Wq_col];
                } else {
                    // MUST zero any shared cell that could be read later
                    Wq_tile[tile_r][tile_c] = 0;
                }

                int Xq_row = t * TILE_SIZE + tile_r;    // global k
                int Xq_col = base_col + tile_c;         // global n
                if (Xq_row < K && Xq_col < N) {
                    int32_t xval = static_cast<int32_t>(Xq[Xq_row * N + Xq_col]);
                    Xq_tile[tile_r][tile_c] = xval - Zx[Xq_col];
                } else {
                    Xq_tile[tile_r][tile_c] = 0;
                }
            }
        }

        __syncthreads();


        for (int k = 0; k < TILE_SIZE; ++k) {
            #pragma unroll
            for (int i = 0; i < REG_TILE_M; ++i) {
                w_reg[i] = Wq_tile[thread_row_in_tile + i][k];
            }

            #pragma unroll
            for (int i = 0; i < REG_TILE_M; ++i) {
                #pragma unroll
                for (int j = 0; j < REG_TILE_N; ++j) {
                    acc[i][j] += static_cast<int32_t>(w_reg[i]) * Xq_tile[k][thread_col_in_tile + j];
                }
            }
        }

        __syncthreads();
    }

    // Write back with scaling
    #pragma unroll
    for (int i = 0; i < REG_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < REG_TILE_N; ++j) {
            int out_r = global_row + i;
            int out_c = global_col + j;
            if (out_r < M && out_c < N) {
                Y[out_r * N + out_c] = static_cast<float>(acc[i][j]) * Sw[out_r] * Sx[out_c];
            }
        }
    }
}

__global__ void dequantize_rowwise_kernel(
    const int8_t* __restrict__ Wq, 
    const float* __restrict__ row_scales,
    float* __restrict__ W_dequant, // The FP32 output tensor
    int M, int K) 
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < M * K; 
         idx += gridDim.x * blockDim.x) {
        
        int row = idx / K;
        
        W_dequant[idx] = static_cast<float>(Wq[idx]) * row_scales[row];
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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> quantize_activations_colwise_wrapper(
    torch::Tensor X, 
    torch::Tensor calib_min,  // Accepts the pre-calculated min value
    torch::Tensor calib_max   // Accepts the pre-calculated max value
) {
    // --- Input Checks ---
    check_contiguous_cuda(X, "X");
    check_contiguous_cuda(calib_min, "calib_min");
    check_contiguous_cuda(calib_max, "calib_max");
    TORCH_CHECK(X.dim() == 2, "X must be 2D");

    int64_t K = X.size(0); // rows
    int64_t N = X.size(1); // Columns

    // --- Output Tensor Allocation ---
    auto options_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(X.device());
    auto options_f = torch::TensorOptions().dtype(torch::kFloat32).device(X.device());
    auto options_i = torch::TensorOptions().dtype(torch::kInt32).device(X.device());

    torch::Tensor Xq = torch::empty({K, N}, options_u8);
    torch::Tensor colScales = torch::empty({N}, options_f);
    torch::Tensor colZPs = torch::empty({N}, options_i);

    // --- Kernel Launch Configuration ---
    // Use a robust, fixed number of threads
    const int threads = 256;
    dim3 blocks(static_cast<uint32_t>(N));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // --- Get Data Pointers ---
    const float* X_ptr = X.data_ptr<float>();
    uint8_t* Xq_ptr = Xq.data_ptr<uint8_t>();
    float* scales_ptr = colScales.data_ptr<float>();
    int* zps_ptr = colZPs.data_ptr<int>();

    const float* calib_mins_ptr = calib_min.data_ptr<float>();
    const float* calib_maxs_ptr = calib_max.data_ptr<float>();

    quantize_activations_colwise<<<blocks, threads, 0, stream>>>(
        X_ptr, Xq_ptr, 
        calib_mins_ptr, calib_maxs_ptr, 
        scales_ptr, zps_ptr, 
        static_cast<int>(K), static_cast<int>(N)
    );
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


    const int threads_per_block_x = TILE_SIZE / REG_TILE_N;  // 32/4 = 8
    const int threads_per_block_y = TILE_SIZE / REG_TILE_M;  // 32/4 = 8
    
    dim3 block(threads_per_block_x, threads_per_block_y);  // 8x8 = 64 threads per block
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    size_t shared_mem_size = TILE_SIZE * TILE_SIZE * sizeof(int8_t) +  // Wq_tile
                            TILE_SIZE * TILE_SIZE * sizeof(int32_t);   // Xq_tile

    const int8_t* Wq_ptr = reinterpret_cast<const int8_t*>(Wq.data_ptr<int8_t>());
    const float* Sw_ptr = Sw.data_ptr<float>();
    const uint8_t* Xq_ptr = Xq.data_ptr<uint8_t>();
    const float* Sx_ptr = Sx.data_ptr<float>();
    const int* Zx_ptr = Zx.data_ptr<int>();
    float* Y_ptr = Y.data_ptr<float>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    qgemm_kernel<<<grid, block, shared_mem_size, stream>>>(Wq_ptr, Sw_ptr, Xq_ptr, Sx_ptr, Zx_ptr, Y_ptr, M, K, N);
    AT_CUDA_CHECK(cudaGetLastError());
    return Y;
}

torch::Tensor dequantize_rowwise_wrapper(torch::Tensor Wq, torch::Tensor row_scales) {
    TORCH_CHECK(Wq.is_cuda(), "Wq must be a CUDA tensor");
    TORCH_CHECK(Wq.is_contiguous(), "Wq must be contiguous");
    TORCH_CHECK(row_scales.is_cuda(), "row_scales must be a CUDA tensor");
    TORCH_CHECK(row_scales.is_contiguous(), "row_scales must be contiguous");
    TORCH_CHECK(Wq.dtype() == torch::kInt8, "Wq must be an INT8 tensor");

    int M = Wq.size(0);
    int K = Wq.size(1);

    // Create the output tensor in FP32
    auto options_f = torch::TensorOptions().dtype(torch::kFloat32).device(Wq.device());
    torch::Tensor W_dequant = torch::empty({M, K}, options_f);

    // Standard CUDA launch configuration
    const int threads = 256;
    const int blocks = (M * K + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Launch the kernel
    dequantize_rowwise_kernel<<<blocks, threads, 0, stream>>>(
        Wq.data_ptr<int8_t>(),
        row_scales.data_ptr<float>(),
        W_dequant.data_ptr<float>(),
        M, K
    );
    AT_CUDA_CHECK(cudaGetLastError());
    
    return W_dequant;
}

// ------------------- Python binding -------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Quantization ops (CUDA) - wrappers for custom kernels";

    m.def("absmax_rowwise", &absmax_rowwise, "Compute row-wise absolute max and produce rowScales (float32) (CUDA)");
    m.def("colwise_minmax", &colwise_minmax_wrapper, "Compute column min/max (returns (mins,maxs)) (CUDA)");
    m.def("quantize_weights_rowwise", &quantize_weights_rowwise_wrapper, "Quantize weights rowwise to int8 (CUDA)");
    m.def("quantize_activations_colwise", &quantize_activations_colwise_wrapper, "Quantize activations columnwise -> returns (Xq, colScales, colZPs) (CUDA)");
    m.def("qgemm", &qgemm_wrapper, "Quantized GEMM with register tiling (CUDA). Call as qgemm(Wq, Sw, Xq, Sx, Zx, M, K, N)");
    m.def("dequantize_rowwise", &dequantize_rowwise_wrapper, "Fast row-wise dequantization from INT8 to FP32");
}