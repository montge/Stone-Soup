/* SPDX-FileCopyrightText: 2017-2025 Stone Soup contributors */
/* SPDX-License-Identifier: MIT */

/**
 * @file cuda_matrix.cu
 * @brief CUDA matrix operations for Stone Soup
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>

extern "C" {
#include "stonesoup/cuda.h"
}

/* Static cuBLAS handle */
static cublasHandle_t g_cublas_handle = NULL;

/**
 * @brief Initialize cuBLAS handle (lazy initialization)
 */
static int ensure_cublas_handle(void) {
    if (g_cublas_handle == NULL) {
        cublasStatus_t status = cublasCreate(&g_cublas_handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "cuBLAS initialization failed\n");
            return -1;
        }
    }
    return 0;
}

/**
 * @brief Check CUDA error
 */
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

/**
 * @brief Check cuBLAS error
 */
#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", \
                __FILE__, __LINE__, (int)status); \
        return -1; \
    } \
} while(0)

extern "C" {

int stonesoup_cuda_matrix_multiply(
    const ss_float_t *A, const ss_float_t *B, ss_float_t *C,
    int m, int k, int n)
{
    if (ensure_cublas_handle() != 0) {
        return -1;
    }

    /* Allocate device memory */
    ss_float_t *d_A, *d_B, *d_C;
    size_t size_A = (size_t)m * k * sizeof(ss_float_t);
    size_t size_B = (size_t)k * n * sizeof(ss_float_t);
    size_t size_C = (size_t)m * n * sizeof(ss_float_t);

    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    /* Copy input matrices to device */
    CUDA_CHECK(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice));

    /* Perform matrix multiplication: C = A * B */
    /* cuBLAS uses column-major, so we compute C^T = B^T * A^T */
    const ss_float_t alpha = 1.0;
    const ss_float_t beta = 0.0;

#if SS_FLOAT_PRECISION == 64
    CUBLAS_CHECK(cublasDgemm(g_cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, m, k,
        &alpha,
        d_B, n,
        d_A, k,
        &beta,
        d_C, n));
#else
    CUBLAS_CHECK(cublasSgemm(g_cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, m, k,
        &alpha,
        d_B, n,
        d_A, k,
        &beta,
        d_C, n));
#endif

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost));

    /* Free device memory */
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

int stonesoup_cuda_matrix_vector_multiply(
    const ss_float_t *A, const ss_float_t *x, ss_float_t *y,
    int m, int n)
{
    if (ensure_cublas_handle() != 0) {
        return -1;
    }

    /* Allocate device memory */
    ss_float_t *d_A, *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_A, (size_t)m * n * sizeof(ss_float_t)));
    CUDA_CHECK(cudaMalloc(&d_x, (size_t)n * sizeof(ss_float_t)));
    CUDA_CHECK(cudaMalloc(&d_y, (size_t)m * sizeof(ss_float_t)));

    /* Copy to device */
    CUDA_CHECK(cudaMemcpy(d_A, A, (size_t)m * n * sizeof(ss_float_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, x, (size_t)n * sizeof(ss_float_t), cudaMemcpyHostToDevice));

    /* y = A * x */
    const ss_float_t alpha = 1.0;
    const ss_float_t beta = 0.0;

#if SS_FLOAT_PRECISION == 64
    CUBLAS_CHECK(cublasDgemv(g_cublas_handle,
        CUBLAS_OP_T,
        n, m,
        &alpha,
        d_A, n,
        d_x, 1,
        &beta,
        d_y, 1));
#else
    CUBLAS_CHECK(cublasSgemv(g_cublas_handle,
        CUBLAS_OP_T,
        n, m,
        &alpha,
        d_A, n,
        d_x, 1,
        &beta,
        d_y, 1));
#endif

    /* Copy result back */
    CUDA_CHECK(cudaMemcpy(y, d_y, (size_t)m * sizeof(ss_float_t), cudaMemcpyDeviceToHost));

    /* Cleanup */
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}

/* Transpose kernel */
__global__ void transpose_kernel(
    const ss_float_t *A, ss_float_t *B,
    int m, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < m && j < n) {
        B[j * m + i] = A[i * n + j];
    }
}

int stonesoup_cuda_matrix_transpose(
    const ss_float_t *A, ss_float_t *B,
    int m, int n)
{
    /* Allocate device memory */
    ss_float_t *d_A, *d_B;
    size_t size_A = (size_t)m * n * sizeof(ss_float_t);
    size_t size_B = (size_t)n * m * sizeof(ss_float_t);

    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));

    /* Copy input to device */
    CUDA_CHECK(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice));

    /* Launch transpose kernel */
    dim3 block(16, 16);
    dim3 grid((m + block.x - 1) / block.x, (n + block.y - 1) / block.y);
    transpose_kernel<<<grid, block>>>(d_A, d_B, m, n);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Copy result back */
    CUDA_CHECK(cudaMemcpy(B, d_B, size_B, cudaMemcpyDeviceToHost));

    /* Cleanup */
    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}

} /* extern "C" */
