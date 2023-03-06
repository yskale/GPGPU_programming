//==============================================================
// Copyright © 2023 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// haowei.zhang@intel.com

// icpx -fiopenmp -fopenmp-targets=spir64 matrixMultiplyLLVMOMP.cpp
// clang++ -fiopenmp -fopenmp-targets=spir64 -I${oneAPIHome}/compiler/latest/linux/compiler/include/ matrixMultiplyLLVMOMP.cpp
// (use the opensource compiler) clang++ -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_70 --libomptarget-nvptx-bc-path=$DPCPP_HOME/llvm/build/install/lib -I${oneAPIHome}/compiler/latest/linux/compiler/include/ matrixMultiplyLLVMOMP.cpp
// LLVM SYCL compiler required, ref: https://github.com/intel/llvm/discussions/3759
// nvc++ -acc -mp=gpu -gpu=cc70 -Minfo=all matrixMultiplyLLVMOMP.cpp
#include <iostream>
#include <chrono>
#include <cmath>
#include <cstring>
#include <omp.h>
/*
DPCT1012:4: Detected kernel execution time measurement pattern and generated an
initial code for time measurements in SYCL. You can change the way time is
measured depending on your goals.
*/
#define __TIME_BEGIN \
    start_ct1 = std::chrono::steady_clock::now();
/*
DPCT1012:5: Detected kernel execution time measurement pattern and generated an
initial code for time measurements in SYCL. You can change the way time is
measured depending on your goals.
*/
#define __TIME_END                               \
    stop_ct1 = std::chrono::steady_clock::now(); \
    elapsedTime =                                \
        std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();

#define TILE_WIDTH 16

typedef float fp;

static int M = 2048;
static int K = 1024;
static int N = 512;

fp *arrayA_h, *arrayB_h, *arrayC_h, *arrayC_href;
std::chrono::time_point<std::chrono::steady_clock> start_ct1;
std::chrono::time_point<std::chrono::steady_clock> stop_ct1;

float elapsedTime;

void resources_init();
void result_reset();
void resources_free();
void print_matrix(const fp *arr, int M, int N);
bool compare_matrix(const fp *arr1, const fp *arr2, int M, int N);
void multiplyCpu(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N);
void multiplyGpuOmp(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N);

int main()
{
    resources_init();

    multiplyCpu(arrayA_h, arrayB_h, arrayC_href, M, K, N);

    multiplyGpuOmp(arrayA_h, arrayB_h, arrayC_h, M, K, N);

    multiplyGpuOmp(arrayA_h, arrayB_h, arrayC_h, M, K, N);

    resources_free();
    return 0;
}

void resources_init()
{
    arrayA_h = new fp[M * K]();
    arrayB_h = new fp[K * N]();
    arrayC_h = new fp[M * N]();
    arrayC_href = new fp[M * N]();

    for (int i = 0; i < M * K; i++)
        arrayA_h[i] = rand() / (fp)RAND_MAX * 1.0;

    for (int i = 0; i < K * N; i++)
        arrayB_h[i] = rand() / (fp)RAND_MAX * 1.0;

    memset(arrayC_h, 0, M * N * sizeof(fp));
    memset(arrayC_href, 0, M * N * sizeof(fp));
    printf("There are %d devices\n", omp_get_num_devices());
    omp_set_default_device(0);
}

void result_reset()
{
    memset(arrayC_h, 0, M * N * sizeof(fp));
}

void resources_free()
{
    delete[] arrayA_h;
    delete[] arrayB_h;
    delete[] arrayC_h;
    delete[] arrayC_href;
}

void print_matrix(const fp *arr, int M, int N)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            std::cout << arr[i * N + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

bool compare_matrix(const fp *arr1, const fp *arr2, int M, int N)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            fp err = fabs(arr1[i * N + j] - arr2[i * N + j]);
            if (err > fabs(arr1[i * N + j] * 1.E-5))
            {
                // print_matrix(arr1, M, N);
                // print_matrix(arr2, M, N);
                return false;
            }
        }
    }
    return true;
}

void multiplyCpu(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N)
{
    __TIME_BEGIN
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            for (int k = 0; k < K; k++)
            {
                arrC[i * N + j] += arrA[i * K + k] * arrB[k * N + j];
            }
        }
    }
    __TIME_END
    printf("0. CPU calculation time = %f ms\n", elapsedTime);
    // print_matrix(arrA, M, K);
    // print_matrix(arrB, K, N);
    // print_matrix(arrC, M, N);
}

void multiplyGpuOmp(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N)
{
    result_reset();
#pragma omp target
    {
        ;
    }
#pragma omp target enter data map(to \
                                  : arrA [0:M * K], arrB [0:K * N], arrC [0:M * N])
    __TIME_BEGIN
#pragma omp target data use_device_ptr(arrA, arrB, arrC)
    {
#pragma omp target teams distribute parallel for collapse(2)
// #pragma omp target teams loop order(concurrent) collapse(2)
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                for (int k = 0; k < K; k++)
                {
                    arrC[i * N + j] += arrA[i * K + k] * arrB[k * N + j];
                }
            }
        }
    }
    __TIME_END

#pragma omp target update from(arrC [0:M * N])
#pragma omp target exit data map(delete \
                                 : arrA [0:M * K], arrB [0:K * N], arrC [0:M * N])
    if (!compare_matrix(arrC, arrayC_href, M, N))
    {
        std::cout << "6. Error at multiplyCpuOmp" << std::endl;
    }
    else
    {
        printf("6. Pass, GPU calculation time (OpenMP) = %f ms\n", elapsedTime);
    }
}
