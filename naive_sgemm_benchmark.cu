#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// --- Error Handling ---
#define CHECK_CUDA(func) { \
    cudaError_t status = (func); \
    if (status != cudaSuccess) { \
        printf("CUDA Error: %s : %d\n", cudaGetErrorString(status), __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

__global__ void naive_sgemm(int M, int N, int K, float alpha, const float* A,
                            const float* B, float beta, float* C) {
    // compute position in C that this thread will compute
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    // if condition is necessary if matrix dimensions are not multiples of block size
    if (x < N && y < M) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[x * K + k] * B[k * N + y];
        }

        C[x * N + y] = alpha * sum + beta * C[x * N + y];
    }                           
}

int main() {
    // 1. Matrix Dimensions
    int M = 4096;
    int N = 4096;
    int K = 4096;

    float alpha = 1.0f;
    float beta = 0.0f;

    printf("Benchmarking Naive SGEMM [M=%d, N=%d, K=%d]\n", M, N, K);

    // Grid config: x covers N (cols), y covers M (rows) to match kernel checks
    dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 32), 1);
    dim3 blockDim(32, 32, 1);

    // 2. Allocate Memory
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void **)&d_A, size_A));
    CHECK_CUDA(cudaMalloc((void **)&d_B, size_B));
    CHECK_CUDA(cudaMalloc((void **)&d_C, size_C));

    // 3. Initialize Data
    // We strictly use cudaMemset for simple deterministic initialization
    CHECK_CUDA(cudaMemset(d_A, 0x40, size_A)); 
    CHECK_CUDA(cudaMemset(d_B, 0x40, size_B)); 
    CHECK_CUDA(cudaMemset(d_C, 0, size_C));

    // 4. Warm-up
    naive_sgemm<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // 5. Benchmark
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int n_iter = 50; 

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < n_iter; i++) {
        naive_sgemm<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
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
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
