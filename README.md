## SGEMM Optimization on GPU

I provided step-wise optimizations on Single Precision General Matrix/Matrix Multiplication (SGEMM) on with CUDA. Our step-wise optimizations reach 70%-80% performance of cuBLAS SGEMM on my test platform - RTX 2080 Super (tu104).

## Descriptions on each kernel

Kernel1: naive implementation.

Kernel2: kernel1 + basic coalescing

Kernel3: kernel2 + blocking

Kernel4: kernel3 + register blocking on C

Kernel5: kernel4 + more workloads per thread

Kernel6: kernel5 + vectorized memory access

## TODOs

1. wrapper layer for edge case - irregular input shapes.

2. float2 vec load to improve register bank conflict issues to further improve the performance.

3. multi-layer shared memory to avoid reduce unnecessary.

4. prefetching to further hide the latency.