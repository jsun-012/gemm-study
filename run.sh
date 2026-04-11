mkdir -p build

nvcc -O3 -arch=sm_90 -lcublas cublas_benchmark.cu -o build/cublas_benchmark
nvcc -O3 -arch=sm_90 -lcublas 01_naive_sgemm_benchmark.cu -o build/naive_sgemm_benchmark
nvcc -O3 -arch=sm_90 -lcublas 02_coalescing_sgemm_benchmark.cu -o build/coalescing_sgemm_benchmark
nvcc -O3 -arch=sm_90 -lcublas 03_smem_caching_sgemm_benchmark.cu -o build/caching_sgemm_benchmark
./build/cublas_benchmark
./build/naive_sgemm_benchmark
./build/coalescing_sgemm_benchmark
./build/caching_sgemm_benchmark