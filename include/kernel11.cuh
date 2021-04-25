#include<stdio.h>
#include<stdlib.h>
#define A(i,j) A[(i) + (j)*lda]
#define B(i,j) B[(i) + (j)*ldb]
#define ptr_A(i,j) ptr_A[(i) + (j)*lda]
#define ptr_B(i,j) ptr_B[(i) + (j)*ldb]
#define C(i,j) C[(i) + (j)*ldc]
#define sa11(i,j) sa11[((j)<<7) + (i)]
#define sb11(i,j) sb11[((j)<<7) + (i)]
#define ptr_sa11(i,j) ptr_sa11[((j)<<7) + (i)]
#define ptr_sb11(i,j) ptr_sb11[((j)<<7) + (i)]
#define MS_11 128
#define NS_11 128
#define KS_11 8
//v1 += v2 * s3, vector scaling
#define vscal(v1, v2, s3)\
    v1.x+=v2.x*s3;\
    v1.y+=v2.y*s3;\
    v1.z+=v2.z*s3;\
    v1.w+=v2.w*s3;
//v1 = alpha * v2 + beta * v3, simd fma
#define simd_axpby(v1, alpha, v2, beta, v3)\
    v1.x=alpha*v2.x+beta*v3.x;\
    v1.y=alpha*v2.y+beta*v3.y;\
    v1.z=alpha*v2.z+beta*v3.z;\
    v1.w=alpha*v2.w+beta*v3.w;
#define vload(v1,addr)\
    v1 = *((float4 *)(addr));
#define vstore(addr,v1)\
    *((float4 *)(addr)) = v1;
// cache blocking version, without register-level data re-use
// with memory coelascing on shared memory
// more workloads per thread. 8x8 micro kernel.
// adopt vetorized load/store
__global__  __launch_bounds__(256)
void mysgemm_v11(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C){
    int lda = M, ldb = K, ldc = M;
    int tx = threadIdx.x;
    int bx = blockIdx.x, by = blockIdx.y;
    int warp_id = tx>>5;
    int lane_id = tx&31;
    int warp_row = warp_id & 3, warp_col = warp_id >> 2;
    int row_w = lane_id&3, col_w = lane_id>>2;
    int row_b = (tx&1)<<2, col_b = tx>>1;
    int lda8 = lda<<3;
    int row_c = (warp_row<<5) + (row_w<<3), col_c = (warp_col<<6) + (col_w<<3);
    int row_a = (tx&31)<<2, col_a = tx>>5;
    int K_upper = K>>3;
    A = &A((bx<<7),0);
    B = &B(0,(by<<7));
    C = &C((bx<<7),(by<<7));//the TB size is 128.
    __shared__ float sa11[2][1024];
    __shared__ float sb11[2][1024];
    float *ptr_sa11, *ptr_sb11;
    ptr_sa11 = (float*)sa11;
    ptr_sb11 = (float*)sb11;
    float4 Av1[2], Av2[2], Bv1[2], Bv2[2], Cv[16], Cres[16];
    float4 pref_Av, pref_Bv;
    float* ptr_A, *ptr_B;
    memset(Cres, 0, sizeof(Cres));//clear registers
    vload(pref_Av, &A(row_a,col_a))
    vload(pref_Bv, &B(row_b,col_b))
    ((float4 *)ptr_sa11)[tx] = pref_Av;
    ptr_sb11(col_b,row_b)=pref_Bv.x;
    ptr_sb11(col_b,row_b+1)=pref_Bv.y;
    ptr_sb11(col_b,row_b+2)=pref_Bv.z;
    ptr_sb11(col_b,row_b+3)=pref_Bv.w;
    __syncthreads();
    vload(Av1[0], &ptr_sa11(row_c,0))
    vload(Av2[0], &ptr_sa11(row_c+4,0))
    vload(Bv1[0], &ptr_sb11(col_c,0))
    vload(Bv2[0], &ptr_sb11(col_c+4,0))
    for (int k_count = 0; k_count<K_upper; k_count++){
        /*packing A and B into shared memory*/
        int inc = (k_count+1)%K_upper;
        int offset = ((k_count+1)&1)<<10;
        ptr_A = A + inc * lda8;
        ptr_B = B + inc * 8;
        vload(pref_Av, &ptr_A(row_a,col_a))
        vload(pref_Bv, &ptr_B(row_b,col_b))
        #pragma unroll
        for (int inner_k_count=0;inner_k_count<KS_11;inner_k_count++){
            int next_inner_k_count = (inner_k_count+1)&7;
            vload(Av1[(inner_k_count+1)&1], &ptr_sa11(row_c,next_inner_k_count))
            vload(Av2[(inner_k_count+1)&1], &ptr_sa11(row_c+4,next_inner_k_count))
            vload(Bv1[(inner_k_count+1)&1], &ptr_sb11(col_c,next_inner_k_count))
            vload(Bv2[(inner_k_count+1)&1], &ptr_sb11(col_c+4,next_inner_k_count))
            vscal(Cres[0], Av1[(inner_k_count)&1], Bv1[(inner_k_count)&1].x)
            vscal(Cres[1], Av2[(inner_k_count)&1], Bv1[(inner_k_count)&1].x)
            vscal(Cres[2], Av1[(inner_k_count)&1], Bv1[(inner_k_count)&1].y)
            vscal(Cres[3], Av2[(inner_k_count)&1], Bv1[(inner_k_count)&1].y)
            vscal(Cres[4], Av1[(inner_k_count)&1], Bv1[(inner_k_count)&1].z)
            vscal(Cres[5], Av2[(inner_k_count)&1], Bv1[(inner_k_count)&1].z)
            vscal(Cres[6], Av1[(inner_k_count)&1], Bv1[(inner_k_count)&1].w)
            vscal(Cres[7], Av2[(inner_k_count)&1], Bv1[(inner_k_count)&1].w)
            vscal(Cres[8], Av1[(inner_k_count)&1], Bv2[(inner_k_count)&1].x)
            vscal(Cres[9], Av2[(inner_k_count)&1], Bv2[(inner_k_count)&1].x)
            vscal(Cres[10], Av1[(inner_k_count)&1], Bv2[(inner_k_count)&1].y)
            vscal(Cres[11], Av2[(inner_k_count)&1], Bv2[(inner_k_count)&1].y)
            vscal(Cres[12], Av1[(inner_k_count)&1], Bv2[(inner_k_count)&1].z)
            vscal(Cres[13], Av2[(inner_k_count)&1], Bv2[(inner_k_count)&1].z)
            vscal(Cres[14], Av1[(inner_k_count)&1], Bv2[(inner_k_count)&1].w)
            vscal(Cres[15], Av2[(inner_k_count)&1], Bv2[(inner_k_count)&1].w)
        }
        ptr_sa11 = (float*)sa11 + offset;
        ptr_sb11 = (float*)sb11 + offset;
        ((float4 *)ptr_sa11)[tx] = pref_Av;
        ptr_sb11(col_b,row_b)=pref_Bv.x;
        ptr_sb11(col_b,row_b+1)=pref_Bv.y;
        ptr_sb11(col_b,row_b+2)=pref_Bv.z;
        ptr_sb11(col_b,row_b+3)=pref_Bv.w;
        __syncthreads();
        vload(Av1[0], &ptr_sa11(row_c,0))
        vload(Av2[0], &ptr_sa11(row_c+4,0))
        vload(Bv1[0], &ptr_sb11(col_c,0))
        vload(Bv2[0], &ptr_sb11(col_c+4,0))
    }
    vload(Cv[0], &C(row_c,col_c))
    vload(Cv[1], &C(row_c+4,col_c))
    vload(Cv[2], &C(row_c,col_c+1))
    vload(Cv[3], &C(row_c+4,col_c+1))
    vload(Cv[4], &C(row_c,col_c+2))
    vload(Cv[5], &C(row_c+4,col_c+2))
    vload(Cv[6], &C(row_c,col_c+3))
    vload(Cv[7], &C(row_c+4,col_c+3))
    vload(Cv[8], &C(row_c,col_c+4))
    vload(Cv[9], &C(row_c+4,col_c+4))
    vload(Cv[10], &C(row_c,col_c+5))
    vload(Cv[11], &C(row_c+4,col_c+5))
    vload(Cv[12], &C(row_c,col_c+6))
    vload(Cv[13], &C(row_c+4,col_c+6))
    vload(Cv[14], &C(row_c,col_c+7))
    vload(Cv[15], &C(row_c+4,col_c+7))
    
    simd_axpby(Cres[0],alpha,Cres[0],beta,Cv[0])
    simd_axpby(Cres[1],alpha,Cres[1],beta,Cv[1])
    simd_axpby(Cres[2],alpha,Cres[2],beta,Cv[2])
    simd_axpby(Cres[3],alpha,Cres[3],beta,Cv[3])

    simd_axpby(Cres[4],alpha,Cres[4],beta,Cv[4])
    simd_axpby(Cres[5],alpha,Cres[5],beta,Cv[5])
    simd_axpby(Cres[6],alpha,Cres[6],beta,Cv[6])
    simd_axpby(Cres[7],alpha,Cres[7],beta,Cv[7])

    simd_axpby(Cres[8],alpha,Cres[8],beta,Cv[8])
    simd_axpby(Cres[9],alpha,Cres[9],beta,Cv[9])
    simd_axpby(Cres[10],alpha,Cres[10],beta,Cv[10])
    simd_axpby(Cres[11],alpha,Cres[11],beta,Cv[11])

    simd_axpby(Cres[12],alpha,Cres[12],beta,Cv[12])
    simd_axpby(Cres[13],alpha,Cres[13],beta,Cv[13])
    simd_axpby(Cres[14],alpha,Cres[14],beta,Cv[14])
    simd_axpby(Cres[15],alpha,Cres[15],beta,Cv[15])

    vstore(&C(row_c,col_c), Cres[0])
    vstore(&C(row_c+4,col_c), Cres[1])
    vstore(&C(row_c,col_c+1), Cres[2])
    vstore(&C(row_c+4,col_c+1), Cres[3])
    vstore(&C(row_c,col_c+2), Cres[4])
    vstore(&C(row_c+4,col_c+2), Cres[5])
    vstore(&C(row_c,col_c+3), Cres[6])
    vstore(&C(row_c+4,col_c+3), Cres[7])
    vstore(&C(row_c,col_c+4), Cres[8])
    vstore(&C(row_c+4,col_c+4), Cres[9])
    vstore(&C(row_c,col_c+5), Cres[10])
    vstore(&C(row_c+4,col_c+5), Cres[11])
    vstore(&C(row_c,col_c+6), Cres[12])
    vstore(&C(row_c+4,col_c+6), Cres[13])
    vstore(&C(row_c,col_c+7), Cres[14])
    vstore(&C(row_c+4,col_c+7), Cres[15])
}