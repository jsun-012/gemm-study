#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// --- Error Handling ---
#define CHECK_CUDA(func) { \
    cudaError_t status = (func); \
    if (status != cudaSuccess) { \
        printf("CUDA Error: %s : %d\n", cudaGetErrorString(status), __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUBLAS(func) { \
    cublasStatus_t status = (func); \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        printf("CUBLAS Error: %d at line %d\n", status, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

int main() {
    // 1. Matrix Dimensions
    int M = 4096;
    int N = 4096;
    int K = 4096;

    // Scalars (must be passed by reference to cuBLAS)
    float alpha = 1.0f;
    float beta = 0.0f;

    printf("Benchmarking cuBLAS SGEMM [M=%d, N=%d, K=%d]\n", M, N, K);

    // 2. Setup Handles
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // 3. Allocate Memory
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void **)&d_A, size_A));
    CHECK_CUDA(cudaMalloc((void **)&d_B, size_B));
    CHECK_CUDA(cudaMalloc((void **)&d_C, size_C));

    // Initialize data (Skipping host init for brevity, just filling device with random/zero)
    // Note: In a real app, use cudaMemcpy from Host.
    // We just memset here to ensure no NaN propagation for benchmark stability.
    CHECK_CUDA(cudaMemset(d_A, 0x40, size_A)); // ~ 3.0f as int representation
    CHECK_CUDA(cudaMemset(d_B, 0x40, size_B)); 
    CHECK_CUDA(cudaMemset(d_C, 0, size_C));

    // 4. Warm-up
    // Note on Arguments for Row-Major C++:
    // We effectively compute C = A * B.
    // cuBLAS sees this as C' = B' * A' (where ' is transpose/col-major view)
    // So we pass:
    // Leading Dimension (LDA) = Stride to next column (which is Row Width in C++)
    
    CHECK_CUBLAS(cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N,          // m (rows of B^T) -> N
        M,          // n (cols of A^T) -> M
        K,          // k
        &alpha,
        d_B, N,     // Matrix B first, Leading Dim = N
        d_A, K,     // Matrix A second, Leading Dim = K
        &beta,
        d_C, N      // Matrix C, Leading Dim = N
    ));
    CHECK_CUDA(cudaDeviceSynchronize());

    // 5. Benchmark
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int n_iter = 50; 

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < n_iter; i++) {
        cublasSgemm(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K, &alpha,
            d_B, N,
            d_A, K, &beta,
            d_C, N
        );
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    // 6. Metrics
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    
    float avg_latency_ms = milliseconds / n_iter;
    
    // 2*M*N*K for Multiply + Add
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double tflops = (flops) / (avg_latency_ms / 1000.0) / 1e12;

    printf("--------------------------------------------------\n");
    printf("Iterations:      %d\n", n_iter);
    printf("Avg Latency:     %.4f ms\n", avg_latency_ms);
    printf("Throughput:      %.4f TFLOPS\n", tflops);
    printf("--------------------------------------------------\n");

    // Cleanup
    CHECK_CUBLAS(cublasDestroy(handle));
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}