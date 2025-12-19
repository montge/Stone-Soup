/* SPDX-FileCopyrightText: 2017-2025 Stone Soup contributors */
/* SPDX-License-Identifier: MIT */

/**
 * @file cuda_kalman.cu
 * @brief CUDA Kalman filter operations for Stone Soup
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <string.h>

extern "C" {
#include "stonesoup/cuda.h"
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
 * @brief Batch Kalman predict kernel
 *
 * Each thread processes one state in the batch.
 * x_pred = F * x
 * P_pred = F * P * F^T + Q
 */
__global__ void batch_kalman_predict_kernel(
    ss_float_t *x_batch,
    ss_float_t *P_batch,
    const ss_float_t *F,
    const ss_float_t *Q,
    int batch_size,
    int state_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    /* Pointer to this state's data */
    ss_float_t *x = x_batch + idx * state_dim;
    ss_float_t *P = P_batch + idx * state_dim * state_dim;

    /* Temporary storage in shared memory would be better for larger dims */
    /* For now, use local arrays (small state dims only) */
    ss_float_t x_pred[16];  /* Max state dim 16 */
    ss_float_t P_pred[256]; /* Max 16x16 */
    ss_float_t FP[256];

    /* x_pred = F * x */
    for (int i = 0; i < state_dim; i++) {
        x_pred[i] = 0.0;
        for (int j = 0; j < state_dim; j++) {
            x_pred[i] += F[i * state_dim + j] * x[j];
        }
    }

    /* FP = F * P */
    for (int i = 0; i < state_dim; i++) {
        for (int j = 0; j < state_dim; j++) {
            FP[i * state_dim + j] = 0.0;
            for (int k = 0; k < state_dim; k++) {
                FP[i * state_dim + j] += F[i * state_dim + k] * P[k * state_dim + j];
            }
        }
    }

    /* P_pred = FP * F^T + Q */
    for (int i = 0; i < state_dim; i++) {
        for (int j = 0; j < state_dim; j++) {
            P_pred[i * state_dim + j] = Q[i * state_dim + j];
            for (int k = 0; k < state_dim; k++) {
                P_pred[i * state_dim + j] += FP[i * state_dim + k] * F[j * state_dim + k];
            }
        }
    }

    /* Write back */
    for (int i = 0; i < state_dim; i++) {
        x[i] = x_pred[i];
    }
    for (int i = 0; i < state_dim * state_dim; i++) {
        P[i] = P_pred[i];
    }
}

extern "C" {

int stonesoup_cuda_batch_kalman_predict(
    ss_float_t *x_batch,
    ss_float_t *P_batch,
    const ss_float_t *F,
    const ss_float_t *Q,
    int batch_size,
    int state_dim)
{
    if (state_dim > 16) {
        fprintf(stderr, "CUDA batch predict: state_dim > 16 not supported\n");
        return -1;
    }

    /* Allocate device memory */
    ss_float_t *d_x, *d_P, *d_F, *d_Q;
    size_t x_size = (size_t)batch_size * state_dim * sizeof(ss_float_t);
    size_t P_size = (size_t)batch_size * state_dim * state_dim * sizeof(ss_float_t);
    size_t F_size = (size_t)state_dim * state_dim * sizeof(ss_float_t);

    CUDA_CHECK(cudaMalloc(&d_x, x_size));
    CUDA_CHECK(cudaMalloc(&d_P, P_size));
    CUDA_CHECK(cudaMalloc(&d_F, F_size));
    CUDA_CHECK(cudaMalloc(&d_Q, F_size));

    /* Copy to device */
    CUDA_CHECK(cudaMemcpy(d_x, x_batch, x_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_P, P_batch, P_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_F, F, F_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Q, Q, F_size, cudaMemcpyHostToDevice));

    /* Launch kernel */
    int block_size = 256;
    int num_blocks = (batch_size + block_size - 1) / block_size;
    batch_kalman_predict_kernel<<<num_blocks, block_size>>>(
        d_x, d_P, d_F, d_Q, batch_size, state_dim);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Copy results back */
    CUDA_CHECK(cudaMemcpy(x_batch, d_x, x_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(P_batch, d_P, P_size, cudaMemcpyDeviceToHost));

    /* Cleanup */
    cudaFree(d_x);
    cudaFree(d_P);
    cudaFree(d_F);
    cudaFree(d_Q);

    return 0;
}

int stonesoup_cuda_kalman_predict(
    ss_float_t *x,
    ss_float_t *P,
    const ss_float_t *F,
    const ss_float_t *Q,
    int state_dim)
{
    /* Single state is just batch of 1 */
    return stonesoup_cuda_batch_kalman_predict(x, P, F, Q, 1, state_dim);
}

int stonesoup_cuda_kalman_update(
    ss_float_t *x,
    ss_float_t *P,
    const ss_float_t *z,
    const ss_float_t *H,
    const ss_float_t *R,
    int state_dim,
    int meas_dim)
{
    /* For single update, CPU is often faster due to GPU overhead */
    /* This is a placeholder - full implementation would use cuBLAS/cuSOLVER */

    /* Allocate temporary arrays */
    ss_float_t *y = (ss_float_t *)malloc((size_t)meas_dim * sizeof(ss_float_t));
    ss_float_t *S = (ss_float_t *)malloc((size_t)meas_dim * meas_dim * sizeof(ss_float_t));
    ss_float_t *K = (ss_float_t *)malloc((size_t)state_dim * meas_dim * sizeof(ss_float_t));
    ss_float_t *PH = (ss_float_t *)malloc((size_t)state_dim * meas_dim * sizeof(ss_float_t));

    if (!y || !S || !K || !PH) {
        free(y); free(S); free(K); free(PH);
        return -1;
    }

    /* y = z - H*x */
    for (int i = 0; i < meas_dim; i++) {
        y[i] = z[i];
        for (int j = 0; j < state_dim; j++) {
            y[i] -= H[i * state_dim + j] * x[j];
        }
    }

    /* PH = P * H^T */
    for (int i = 0; i < state_dim; i++) {
        for (int j = 0; j < meas_dim; j++) {
            PH[i * meas_dim + j] = 0.0;
            for (int k = 0; k < state_dim; k++) {
                PH[i * meas_dim + j] += P[i * state_dim + k] * H[j * state_dim + k];
            }
        }
    }

    /* S = H * P * H^T + R = H * PH^T + R */
    for (int i = 0; i < meas_dim; i++) {
        for (int j = 0; j < meas_dim; j++) {
            S[i * meas_dim + j] = R[i * meas_dim + j];
            for (int k = 0; k < state_dim; k++) {
                S[i * meas_dim + j] += H[i * state_dim + k] * PH[k * meas_dim + j];
            }
        }
    }

    /* K = PH * S^-1 (simplified for 1D measurement) */
    if (meas_dim == 1) {
        ss_float_t s_inv = 1.0 / S[0];
        for (int i = 0; i < state_dim; i++) {
            K[i] = PH[i] * s_inv;
        }
    } else {
        /* TODO: Proper matrix inversion for higher dimensions */
        free(y); free(S); free(K); free(PH);
        return -1;
    }

    /* x = x + K * y */
    for (int i = 0; i < state_dim; i++) {
        for (int j = 0; j < meas_dim; j++) {
            x[i] += K[i * meas_dim + j] * y[j];
        }
    }

    /* P = (I - K*H) * P (simplified Joseph form) */
    ss_float_t *KH = (ss_float_t *)malloc((size_t)state_dim * state_dim * sizeof(ss_float_t));
    ss_float_t *P_new = (ss_float_t *)malloc((size_t)state_dim * state_dim * sizeof(ss_float_t));

    if (KH && P_new) {
        /* KH = K * H */
        for (int i = 0; i < state_dim; i++) {
            for (int j = 0; j < state_dim; j++) {
                KH[i * state_dim + j] = 0.0;
                for (int k = 0; k < meas_dim; k++) {
                    KH[i * state_dim + j] += K[i * meas_dim + k] * H[k * state_dim + j];
                }
            }
        }

        /* P_new = (I - KH) * P */
        for (int i = 0; i < state_dim; i++) {
            for (int j = 0; j < state_dim; j++) {
                P_new[i * state_dim + j] = 0.0;
                for (int k = 0; k < state_dim; k++) {
                    ss_float_t factor = (i == k ? 1.0 : 0.0) - KH[i * state_dim + k];
                    P_new[i * state_dim + j] += factor * P[k * state_dim + j];
                }
            }
        }

        memcpy(P, P_new, (size_t)state_dim * state_dim * sizeof(ss_float_t));
    }

    free(y);
    free(S);
    free(K);
    free(PH);
    free(KH);
    free(P_new);

    return 0;
}

} /* extern "C" */
