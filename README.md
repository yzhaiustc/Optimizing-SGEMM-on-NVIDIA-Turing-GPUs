# How to optimize SGEMM on NVIDIA GPUs

"HPC is all about reducing data movement". Optimizing GEMM on GPU and CPU platforms share the same idea: to hide the memory latency with massive parallelism, cache-/register-level data re-use, and manual prefetching. On CPUs, both instruction-level and data-level parallelisms are exploited as well as delicate prefetching schemes are designed to hide the memory latency. Meanwhile, we partition the input matrices and pack them before computing to ensure a "smooth and low-latancy" computing kernel. The prefetching for matrix ```C``` is especially critical to the CPU GEMM performance.

On GPU, we also need to take advantage of the low-latency cache. There are rich opportunities on GPUs for us to exploit cache-level data re-use. More details could be found in the official document of [CUTLASS](https://github.com/NVIDIA/cutlass/blob/master/media/docs/efficient_gemm.md).

All questions are encouraged to sent to [yujiazhai94@gmail.com](mailto:yujiazhai94@gmail.com).

# Hardware platforms and software configurations

* We compiled the program with ```gcc 7.3.0``` under Ubuntu 18.04.5 LTS.
* NVIDIA cuBLAS version: ```CUDA cuBLAS 11.3.1.68```.

# How to run
Just three steps.
* We first modify the path of ```nvcc``` in ```Makefile```.
* Second, type in ```make``` to compile. A binary executable ```sgemm_gpu``` will be generated.
* Third, run the binary using ```./sgemm_gpu [kernel_number]```, where ```kernel_number``` selects the kernel for benchmark. ```0``` represents NVIDIA cuBLAS and ```1-11``` represent 11 kernels demonstrating the optimizing strategies.

# Step-wise Optimizations

Here we take the column-major implemetation for SGEMM. Both A and B are not transposed.

## Kernel 1 (naive version)
[source code](https://www.cs.ucr.edu/~yzhai015/GPU_GEMM/kernel1.cuh)

Kernel 1 is the most naive implementation of SGEMM in CUDA. This is the triple-for-loop implementation with register re-use when updating ```C(i,j)```. In this version, each threa block (TB) is responsible for a ```32x32``` sub-block of ```C```, and each thread computes only a single element of the ```C``` matrix.

## Kernel 2 (Kernel1 + 32x32x32 tiling)
[source code](https://www.cs.ucr.edu/~yzhai015/GPU_GEMM/kernel2.cuh)

Kernel2 partitions the matrix ```A``` and matrix ```B``` into ```32x32``` blocks. These ```32x32``` blocks are loaded into shared memory before being loaded for GEMM computation. When loading the data into shared memory (this is called as packing in CPU GEMM), each thread is responsible to load/store one element and we set 1024 threads per TB using ```__launch_bounds__(1024)```. After packing is completed, all the threads are synchronized and then start to compute for their own element. Since each TB is still to compute a ```32x32``` matrix ```C```, each thread remains to take a single element of ```C```.
In short, this version adds cache blocking upon [the previous version](https://www.cs.ucr.edu/~yzhai015/GPU_GEMM/kernel1.cuh), with the parameter set ```{Ms,Ns,Ks}={32,32,32}```.

![image](https://www.cs.ucr.edu/~yzhai015/GPU_GEMM/Kernel1.png)

## Kernel 3 (minor update on Kernel2)
[source code](https://www.cs.ucr.edu/~yzhai015/GPU_GEMM/kernel3.cuh)

We bring a simple optimization upon [kernel 2](https://www.cs.ucr.edu/~yzhai015/GPU_GEMM/kernel2.cuh) here: storing ```threadIdx.x``` before re-using it massively, in order to reduce living registers and benefit the compiler optimization. The performance slightly improves in this step.

![image](https://www.cs.ucr.edu/~yzhai015/GPU_GEMM/Kernel2.png)

## Kernel 4 (kernel 3 + coalescing)
[source code](https://www.cs.ucr.edu/~yzhai015/GPU_GEMM/kernel4.cuh)

In the previous version, the memory access on the shared memory is not coalesced. We re-ordered the memory access pattern on the shared memory: making the shared memory col-major but transposing matrix ```B``` when packing it into the shared memory. This doubles the performance.

![image](https://www.cs.ucr.edu/~yzhai015/GPU_GEMM/Kernel3.png)

## Kernel 5 (kernel4 + 4x1 micro kernel)
[source code](https://www.cs.ucr.edu/~yzhai015/GPU_GEMM/kernel5.cuh)

In this step, we ask each thread to compute 4 elements for the ```C``` matrix. Therefore, we now have 256 threads in a TB to compute the ```32x32``` matrix ```C``` that the TB is responsible for. Using the CPU-GEMM language, the micro kernel's shape is: ```4x1```: that is to say, after the packing routine completes, each thread loads a ```4x1``` A and an ```1x1``` B and computes ```C(4x1)``` += ```A(4x1)*B(1x1)```.
Starting from this step, we restrict 256 threads for each TB.

![image](https://www.cs.ucr.edu/~yzhai015/GPU_GEMM/Kernel4.png)

## Kernel 6 (kernel5 + vectorized load/store)
[source code](https://www.cs.ucr.edu/~yzhai015/GPU_GEMM/kernel6.cuh)

Since our target machine supports a 128-bit transaction from the DRAM, we can apply the vectorized load operation using the ```float4``` data type.

![image](https://www.cs.ucr.edu/~yzhai015/GPU_GEMM/Kernel5.png)

## Kernel 7 ({Ms,Ns,Ks}={64,64,16}, {Mr,Nr}={4,4})
[source code](https://www.cs.ucr.edu/~yzhai015/GPU_GEMM/kernel7.cuh)

Considering there are sufficient registers (64K) for a TB while we only assign 256 threads for each TB, it should be safe, in terms of the performance, for us to assign more workloads to each thread. Now we ask each thread to compute a ```4x4``` sub-matrix of ```C``` so we gain massive data re-use at the register level compared with the previous step.

Additionally, when the input matrices are large, we can increase ```Ms``` and ```Ns``` and maintain enough TBs to map to streaming multiprocessors. Here we increase ```{Ms,Ns}``` from the previous ```{32,32}``` to ```{64,64}``` but decreased the ```Ks``` from ```32``` to ```16``` to maintain the same shared memory consumption. Since everything but the two parameters ```Ms,Ns``` are different, we deduce that the asking TBs to do more jobs benefits the performance when the input matrices are large enough.

![image](https://www.cs.ucr.edu/~yzhai015/GPU_GEMM/Kernel6.png)

## Kernel 8 ({Ms,Ns,Ks}={128,128,8}, {Mr,Nr}={8,8})
[source code](https://www.cs.ucr.edu/~yzhai015/GPU_GEMM/kernel8.cuh)

Assign more workloads for each TB AND each thread.

![image](https://www.cs.ucr.edu/~yzhai015/GPU_GEMM/Kernel7.png)

## Kernel 9 (Kernel 8 + warp-level tiling/parallelism)
[source code](https://www.cs.ucr.edu/~yzhai015/GPU_GEMM/kernel9.cuh)

Since each warp contains 32 threads and the memory accesses to the same memory address in shared memory within the same warp can be coalesced, we introduce a ```{Mw,Nw}```=```{4xMr,8xNr}``` to benefit the warp-level parallelism. We refer readers to [(Huang, 2018)](https://arxiv.org/abs/1808.07984) for more details.

![image](https://www.cs.ucr.edu/~yzhai015/GPU_GEMM/Kernel8.png)

## Kernel 10 (Kernel9 + prefetching ([Huang, 2018](https://arxiv.org/abs/1808.07984)))
[source code](https://www.cs.ucr.edu/~yzhai015/GPU_GEMM/kernel10.cuh)

![image](https://www.cs.ucr.edu/~yzhai015/GPU_GEMM/Kernel9.png)

## Kernel 11 (Kernel10 + double buffer to cancel a sync.)
[source code](https://www.cs.ucr.edu/~yzhai015/GPU_GEMM/kernel11.cuh)

We introduce the double buffer strategy for the shared memory buffers to cancel an unnecessary syncthreads inside the loop body, pushing the performance to the limit - mostly same as the close-sourced NVIDIA cuBLAS.

![image](https://www.cs.ucr.edu/~yzhai015/GPU_GEMM/Kernel10.png)

## Compare with cuBLAS

![image](https://www.cs.ucr.edu/~yzhai015/GPU_GEMM/Kernel11.png)
