mkdir -p build

nvcc -O3 -arch=sm_80 -lcublas cublas_benchmark.cu -o build/cublas_benchmark
nvcc -O3 -arch=sm_80 -lcublas naive_sgemm_benchmark.cu -o build/naive_sgemm_benchmark
nvcc -O3 -arch=sm_80 -lcublas coalescing_sgemm_benchmark.cu -o build/coalescing_sgemm_benchmark
./build/cublas_benchmark
./build/naive_sgemm_benchmark
./build/coalescing_sgemm_benchmark