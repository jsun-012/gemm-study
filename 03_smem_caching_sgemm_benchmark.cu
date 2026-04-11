#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#define BLOCKSIZE 32


// --- Error Handling ---
#define CHECK_CUDA(func) { \
    cudaError_t status = (func); \
    if (status != cudaSuccess) { \
        printf("CUDA Error: %s : %d\n", cudaGetErrorString(status), __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

__global__ void smem_caching_sgemm(int M, int N, int K, float alpha, const float* A,
                                   const float* B, float beta, float* C) {
    // load chunk of A and a chunk of B from global memory into shared memory. 
    // each thread still being assigned one entry of C
    // move the chunk along the columns of A and the rows of B performing partial sums on C until result is computed


    // compute tiling position
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    // allow global memory access coalescing
    const uint threadRow = threadIdx.y;
    const uint threadCol = threadIdx.x;

    // advance pointers to the starting positions.
    A += cRow * BLOCKSIZE * K;
    B += cCol * BLOCKSIZE;
    C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;

    // define shared memory As, Bs
    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];
        
    // each iteration only calculate partial value of an element
    float tmp = 0.0;

    // the outer loop advances A along the columns and B along the rows
    // until we have fully calculated result in C
    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
        // have each thread load one of the elements in A & B from
        // global memory into shared memory
        // make threadCol (=threadIdx.x) the consecutive index
        // to allow global memory access coalescing
        As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
        Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

        // block threads in this block until cache is fully populated
        __syncthreads();

        // advance pointers onto next chunk
        A += BLOCKSIZE;
        B += BLOCKSIZE * N;

        // execute the matmul on the current cached tile/block
        for (int k = 0; k < BLOCKSIZE; ++k) {
            tmp += As[threadRow * BLOCKSIZE + k] * Bs[k * BLOCKSIZE + threadCol];
        }

        // need sync again at the end, to avoid faster threads
        // fetching next block into the cache before slower threads are done
        __syncthreads();
    }
    
    C[threadRow * N + threadCol] = alpha * tmp + beta * C[threadRow * N + threadCol];
}

int main() {
    // 1. Matrix Dimensions
    int M = 4096;
    int N = 4096;
    int K = 4096;

    float alpha = 1.0f;
    float beta = 0.0f;

    printf("Benchmarking smem caching SGEMM [M=%d, N=%d, K=%d]\n", M, N, K);

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
    smem_caching_sgemm<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // 5. Benchmark
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    int n_iter = 50; 

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < n_iter; i++) {
        smem_caching_sgemm<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
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
