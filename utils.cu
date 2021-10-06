#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "utils.cuh"
#include "kernels.cuh"
#include <helper_string.h> 
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#define FLOAT float
#define INT int
#define CEIL_DIV(m,n) ( (m) + (n) - 1 ) / (n)
void print_matrix(const FLOAT *A, int m, int n){
    int i;
    printf("[");
    for (i = 0; i < m * n; i++){
        if ((i + 1) % n == 0) printf("%5.2f ", A[i]);
        else printf("%5.2f, ", A[i]);
        if ((i + 1) % n == 0){
            if (i + 1 < m * n) printf(";\n");
        }
    }
    printf("]\n");
}

void randomize_matrix(FLOAT* mat, int N){
    srand(time(NULL)); int i;
    for (i = 0; i < N; i++) {
        FLOAT tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
        tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
        //tmp = i;
        mat[i] = tmp;
    }
}

double get_sec(){
    struct timeval time;
    gettimeofday(&time, NULL); 
    return (time.tv_sec + 1e-6 * time.tv_usec);
}

bool verify_matrix(FLOAT *mat1, FLOAT *mat2, int n){
    double diff = 0.0;
    int i;
    for (i = 0; mat1 + i && mat2 + i && i < n; i++){
        diff = fabs( (double)mat1[i] - (double)mat2[i] );
        if (diff > 1e-2) {
            printf("error. %5.2f,%5.2f,%d\n", mat1[i],mat2[i],i);
            return false;
        }
    }
    return true;
}

void copy_matrix(FLOAT *src, FLOAT *dest, int n){
    int i;
    for (i = 0; src + i && dest + i && i < n; i++) *(dest + i) = *(src + i);
    if (i != n) printf("copy failed at %d while there are %d elements in total.\n", i, n);
}



void test_cublas(cublasHandle_t err, INT M, INT N, INT K, FLOAT alpha, FLOAT *A, FLOAT *B, FLOAT beta, FLOAT *C){
    cudaDeviceSynchronize();
    cublasSgemm(err, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, A, M, B, K, &beta, C, M);
    cudaDeviceSynchronize();
}

void test_mysgemm_v1(INT M, INT N, INT K, FLOAT alpha, FLOAT *A, FLOAT *B, FLOAT beta, FLOAT *C){
    cudaDeviceSynchronize();
    dim3 blockDim(32,32);
    dim3 gridDim(CEIL_DIV(M,32),CEIL_DIV(N,32));
    mysgemm_v1<<<gridDim, blockDim>>>(M,N,K,alpha,A,B,beta,C);
    cudaDeviceSynchronize();
}

void test_mysgemm_v2(INT M, INT N, INT K, FLOAT alpha, FLOAT *A, FLOAT *B, FLOAT beta, FLOAT *C){
    cudaDeviceSynchronize();
    dim3 blockDim(32,32);
    dim3 gridDim(CEIL_DIV(M,32),CEIL_DIV(N,32));
    mysgemm_v2<<<gridDim, blockDim>>>(M,N,K,alpha,A,B,beta,C);
    cudaDeviceSynchronize();
}

void test_mysgemm_v3(INT M, INT N, INT K, FLOAT alpha, FLOAT *A, FLOAT *B, FLOAT beta, FLOAT *C){
    cudaDeviceSynchronize();
    dim3 blockDim(1024);
    dim3 gridDim(CEIL_DIV(M,32),CEIL_DIV(N,32));
    mysgemm_v3<<<gridDim, blockDim>>>(M,N,K,alpha,A,B,beta,C);
    cudaDeviceSynchronize();
}

void test_mysgemm_v4(INT M, INT N, INT K, FLOAT alpha, FLOAT *A, FLOAT *B, FLOAT beta, FLOAT *C){
    cudaDeviceSynchronize();
    dim3 blockDim(1024);
    dim3 gridDim(CEIL_DIV(M,32),CEIL_DIV(N,32));
    mysgemm_v4<<<gridDim, blockDim>>>(M,N,K,alpha,A,B,beta,C);
    cudaDeviceSynchronize();
}

void test_mysgemm_v5(INT M, INT N, INT K, FLOAT alpha, FLOAT *A, FLOAT *B, FLOAT beta, FLOAT *C){
    cudaDeviceSynchronize();
    dim3 blockDim(256);
    dim3 gridDim(CEIL_DIV(M,32),CEIL_DIV(N,32));
    mysgemm_v5<<<gridDim, blockDim>>>(M,N,K,alpha,A,B,beta,C);
    cudaDeviceSynchronize();
}

void test_mysgemm_v6(INT M, INT N, INT K, FLOAT alpha, FLOAT *A, FLOAT *B, FLOAT beta, FLOAT *C){
    cudaDeviceSynchronize();
    dim3 blockDim(256);
    dim3 gridDim(CEIL_DIV(M,32),CEIL_DIV(N,32));
    mysgemm_v6<<<gridDim, blockDim>>>(M,N,K,alpha,A,B,beta,C);
    cudaDeviceSynchronize();
}

void test_mysgemm_v7(INT M, INT N, INT K, FLOAT alpha, FLOAT *A, FLOAT *B, FLOAT beta, FLOAT *C){
    cudaDeviceSynchronize();
    dim3 blockDim(256);
    dim3 gridDim(CEIL_DIV(M,64),CEIL_DIV(N,64));
    mysgemm_v7<<<gridDim, blockDim>>>(M,N,K,alpha,A,B,beta,C);
    cudaDeviceSynchronize();
}

void test_mysgemm_v8(INT M, INT N, INT K, FLOAT alpha, FLOAT *A, FLOAT *B, FLOAT beta, FLOAT *C){
    cudaDeviceSynchronize();
    dim3 blockDim(256);
    dim3 gridDim(CEIL_DIV(M,128),CEIL_DIV(N,128));
    mysgemm_v8<<<gridDim, blockDim>>>(M,N,K,alpha,A,B,beta,C);
    cudaDeviceSynchronize();
}

void test_mysgemm_v9(INT M, INT N, INT K, FLOAT alpha, FLOAT *A, FLOAT *B, FLOAT beta, FLOAT *C){
    cudaDeviceSynchronize();
    dim3 blockDim(256);
    dim3 gridDim(CEIL_DIV(M,128),CEIL_DIV(N,128));
    mysgemm_v9<<<gridDim, blockDim>>>(M,N,K,alpha,A,B,beta,C);
    cudaDeviceSynchronize();
}

void test_mysgemm_v10(INT M, INT N, INT K, FLOAT alpha, FLOAT *A, FLOAT *B, FLOAT beta, FLOAT *C){
    cudaDeviceSynchronize();
    dim3 blockDim(256);
    dim3 gridDim(CEIL_DIV(M,128),CEIL_DIV(N,128));
    mysgemm_v10<<<gridDim, blockDim>>>(M,N,K,alpha,A,B,beta,C);
    cudaDeviceSynchronize();
}

void test_mysgemm_v11(INT M, INT N, INT K, FLOAT alpha, FLOAT *A, FLOAT *B, FLOAT beta, FLOAT *C){
    cudaDeviceSynchronize();
    dim3 blockDim(256);
    dim3 gridDim(CEIL_DIV(M,128),CEIL_DIV(N,128));
    mysgemm_v11<<<gridDim, blockDim>>>(M,N,K,alpha,A,B,beta,C);
    cudaDeviceSynchronize();
}

void test_kernel(int kernel_num,INT m,INT n,INT k,FLOAT alpha,FLOAT *A,FLOAT *B,FLOAT beta,FLOAT *C, cublasHandle_t err){
    switch (kernel_num){
        case 0: test_cublas(err, m,n,k,alpha,A,B,beta,C); break;
        case 1: test_mysgemm_v1(m,n,k,alpha,A,B,beta,C); break;
        case 2: test_mysgemm_v2(m,n,k,alpha,A,B,beta,C); break;
        case 3: test_mysgemm_v3(m,n,k,alpha,A,B,beta,C); break;
        case 4: test_mysgemm_v4(m,n,k,alpha,A,B,beta,C); break;
        case 5: test_mysgemm_v5(m,n,k,alpha,A,B,beta,C); break;
        case 6: test_mysgemm_v6(m,n,k,alpha,A,B,beta,C); break;
        case 7: test_mysgemm_v7(m,n,k,alpha,A,B,beta,C); break;
        case 8: test_mysgemm_v8(m,n,k,alpha,A,B,beta,C); break;
        case 9: test_mysgemm_v9(m,n,k,alpha,A,B,beta,C); break;
        case 10: test_mysgemm_v10(m,n,k,alpha,A,B,beta,C); break;
        case 11: test_mysgemm_v11(m,n,k,alpha,A,B,beta,C); break;
        default: break;
    }
}