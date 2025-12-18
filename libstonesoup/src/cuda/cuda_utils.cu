/* SPDX-FileCopyrightText: 2017-2025 Stone Soup contributors */
/* SPDX-License-Identifier: MIT */

/**
 * @file cuda_utils.cu
 * @brief CUDA utility functions for Stone Soup
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

extern "C" {
#include "stonesoup/cuda.h"
}

/**
 * @brief Check CUDA error and print message
 */
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

extern "C" {

int stonesoup_cuda_available(void) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        return 0;
    }
    return 1;
}

int stonesoup_cuda_device_count(void) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        return 0;
    }
    return device_count;
}

int stonesoup_cuda_device_name(int device, char *name, size_t max_len) {
    if (name == NULL || max_len == 0) {
        return -1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    strncpy(name, prop.name, max_len - 1);
    name[max_len - 1] = '\0';
    return 0;
}

int stonesoup_cuda_memory_info(int device, size_t *total_bytes, size_t *free_bytes) {
    if (total_bytes == NULL || free_bytes == NULL) {
        return -1;
    }

    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaMemGetInfo(free_bytes, total_bytes));
    return 0;
}

int stonesoup_cuda_set_device(int device) {
    CUDA_CHECK(cudaSetDevice(device));
    return 0;
}

} /* extern "C" */
