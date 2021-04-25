#include<stdio.h>
#include<stdlib.h>
#define A(i,j) A[(i) + (j)*lda]
#define B(i,j) B[(i) + (j)*ldb]
#define C(i,j) C[(i) + (j)*ldc]
#define sa5(i,j) sa5[((j)<<5) + (i)]
#define sb5(i,j) sb5[((j)<<5) + (i)]
#define MS 32
#define NS 32
#define KS 32
// cache blocking version, without register-level data re-use
// with memory coelascing on shared memory
// more workloads per thread. 4x1 micro kernel.
__global__  __launch_bounds__(256)
void mysgemm_v5(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C){
    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x;
    int bx = blockIdx.x, by = blockIdx.y;
    int row1 = (tx&7)<<2, row2 = row1+1, row3 = row1+2, row4 = row1+3, col = tx>>3;
    A = &A((bx<<5),0);
    B = &B(0,(by<<5));
    C = &C((bx<<5),(by<<5));
    __shared__ float sa5[MS*KS];
    __shared__ float sb5[KS*NS];
    float Cres[4] = {0., 0., 0., 0.};
    float b00;
    for (int k_count = 0; k_count<K; k_count+=KS){
        sa5(row1,col)=A(row1,col);
        sa5(row2,col)=A(row2,col);
        sa5(row3,col)=A(row3,col);
        sa5(row4,col)=A(row4,col);
        sb5(col,row1)=B(row1,col);
        sb5(col,row2)=B(row2,col);
        sb5(col,row3)=B(row3,col);
        sb5(col,row4)=B(row4,col);
        A+=(lda<<5);B+=32;
        __syncthreads();
        #pragma unroll
        for (int inner_k_count=0;inner_k_count<KS;inner_k_count++){
            b00 = sb5(col,inner_k_count);
            Cres[0] += sa5(row1,inner_k_count) * b00;
            Cres[1] += sa5(row2,inner_k_count) * b00;
            Cres[2] += sa5(row3,inner_k_count) * b00;
            Cres[3] += sa5(row4,inner_k_count) * b00;
        }
        __syncthreads();
    }
    C(row1,col) = alpha * Cres[0] + beta*C(row1,col);
    C(row2,col) = alpha * Cres[1] + beta*C(row2,col);
    C(row3,col) = alpha * Cres[2] + beta*C(row3,col);
    C(row4,col) = alpha * Cres[3] + beta*C(row4,col);
}