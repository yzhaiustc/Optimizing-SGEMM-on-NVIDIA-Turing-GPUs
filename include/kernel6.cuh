#include<stdio.h>
#include<stdlib.h>
#define A(i,j) A[(i) + (j)*lda]
#define B(i,j) B[(i) + (j)*ldb]
#define C(i,j) C[(i) + (j)*ldc]
#define sa6(i,j) sa6[((j)<<5) + (i)]
#define sb6(i,j) sb6[((i)<<5) + (j)]
#define MS 32
#define NS 32
#define KS 32
// cache blocking version, without register-level data re-used
// with memory coelascing on shared memory
// more workloads per thread. 4x1 micro kernel.
// adopt vetorized load/store
__global__  __launch_bounds__(256)
void mysgemm_v6(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C){
    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x;
    int bx = blockIdx.x, by = blockIdx.y;
    int row1 = (tx&7)<<2, row2 = row1+1, row3 = row1+2, row4 = row1+3, col = tx>>3;
    A = &A((bx<<5),0);
    B = &B(0,(by<<5));
    C = &C((bx<<5),(by<<5));
    __shared__ float sa6[MS*KS];
    __shared__ float sb6[KS*NS];
    float4 Av, Bv, Cv, Cres;
    Cres.x = 0., Cres.y = 0., Cres.z = 0., Cres.w = 0.;
    float b00;
    for (int k_count = 0; k_count<K; k_count+=KS){
        Av = *((float4 *)(&A(row1,col)));
        Bv = *((float4 *)(&B(row1,col)));
        ((float4 *)sa6)[tx] = Av;
        sb6(col,row1)=Bv.x;
        sb6(col,row2)=Bv.y;
        sb6(col,row3)=Bv.z;
        sb6(col,row4)=Bv.w;
        A+=(lda<<5);B+=32;
        __syncthreads();
        #pragma unroll
        for (int inner_k_count=0;inner_k_count<KS;inner_k_count++){
            b00 = sb6(col,inner_k_count);
            Cres.x += sa6(row1,inner_k_count) * b00;
            Cres.y += sa6(row2,inner_k_count) * b00;
            Cres.z += sa6(row3,inner_k_count) * b00;
            Cres.w += sa6(row4,inner_k_count) * b00;
        }
        __syncthreads();
    }
    Cv = *((float4 *)(&C(row1,col)));
    Cres.x = alpha * Cres.x + beta * Cv.x;
    Cres.y = alpha * Cres.y + beta * Cv.y;
    Cres.z = alpha * Cres.z + beta * Cv.z;
    Cres.w = alpha * Cres.w + beta * Cv.w;
    *(float4 *)(&(C(row1,col))) = Cres;
}