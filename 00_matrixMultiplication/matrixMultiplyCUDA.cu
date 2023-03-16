//==============================================================
// Copyright © 2023 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// haowei.zhang@intel.com

// HPCSDK: nvc++ -acc -mp=gpu -gpu=cc70 -Minfo=all matrixMultiplyCUDA.cu
// CUDA Toolkit: nvcc -arch=sm_70 matrixMultiplyCUDA.cu

#include <iostream>
#include <chrono>
#include <cmath>
#include <cstring>
#include <omp.h>

#define __TIME_BEGIN cudaEventRecord(start);
#define __TIME_END              \
    cudaEventRecord(stop);      \
    cudaEventSynchronize(stop); \
    cudaEventElapsedTime(&elapsedTime, start, stop);

#define TILE_WIDTH 16

typedef float fp;
typedef float4 fp4;

static int M = 2048;
static int K = 1024;
static int N = 512;

fp *arrayA_h, *arrayB_h, *arrayC_h, *arrayC_href;
fp *arrayA_d, *arrayB_d, *arrayC_d;
cudaEvent_t start, stop;

float elapsedTime;

void resources_init();
void result_reset();
void resources_free();
void print_matrix(const fp *arr, int M, int N);
bool compare_matrix(const fp *arr1, const fp *arr2, int M, int N);
void multiplyCpu(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N);
void multiplyGpu(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N);
void multiplyGpuVec(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N);
void multiplyGpuSh(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N);
void multiplyGpuSh2(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N);
void multiplyGpuShBc(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N);
void multiplyGpuShBcPd(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N);
void multiplyGpuAcc(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N);
void multiplyGpuOmp(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N);

int main()
{
    resources_init();

    multiplyCpu(arrayA_h, arrayB_h, arrayC_href, M, K, N);

    multiplyGpu(arrayA_d, arrayB_d, arrayC_d, M, K, N);

    multiplyGpuVec(arrayA_d, arrayB_d, arrayC_d, M, K, N);

    multiplyGpuSh(arrayA_d, arrayB_d, arrayC_d, M, K, N);

    // multiplyGpuSh2(arrayA_d, arrayB_d, arrayC_d, M, K, N);

    multiplyGpuShBc(arrayA_d, arrayB_d, arrayC_d, M, K, N);

    multiplyGpuShBcPd(arrayA_d, arrayB_d, arrayC_d, M, K, N);

    multiplyGpuAcc(arrayA_h, arrayB_h, arrayC_h, M, K, N);

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

    cudaMalloc((void **)&arrayA_d, M * K * sizeof(fp));
    cudaMalloc((void **)&arrayB_d, K * N * sizeof(fp));
    cudaMalloc((void **)&arrayC_d, M * N * sizeof(fp));

    for (int i = 0; i < M * K; i++)
        arrayA_h[i] = rand() / (fp)RAND_MAX * 1.0;

    for (int i = 0; i < K * N; i++)
        arrayB_h[i] = rand() / (fp)RAND_MAX * 1.0;

    memset(arrayC_h, 0, M * N * sizeof(fp));
    memset(arrayC_href, 0, M * N * sizeof(fp));

    cudaMemcpy(arrayA_d, arrayA_h, M * K * sizeof(fp), cudaMemcpyHostToDevice);
    cudaMemcpy(arrayB_d, arrayB_h, K * N * sizeof(fp), cudaMemcpyHostToDevice);
    cudaMemset(arrayC_d, 0, M * N * sizeof(fp));
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    printf("There are %d OpenMP devices\n", omp_get_num_devices());
    omp_set_default_device(0);
}

void result_reset()
{
    memset(arrayC_h, 0, M * N * sizeof(fp));
    cudaMemset(arrayC_d, 0, M * N * sizeof(fp));
}

void resources_free()
{
    delete[] arrayA_h;
    delete[] arrayB_h;
    delete[] arrayC_h;
    delete[] arrayC_href;
    cudaFree(arrayA_d);
    cudaFree(arrayB_d);
    cudaFree(arrayC_d);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
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

__global__ void _matrixMul(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N, int order)
{
    // absolute row and col
    unsigned int row;
    unsigned int col;
    if (order == 1)
    {
        row = blockDim.x * blockIdx.x + threadIdx.x;
        col = blockDim.y * blockIdx.y + threadIdx.y;
    }
    else if (order == 2)
    {
        row = blockDim.z * blockIdx.z + threadIdx.z;
        col = blockDim.y * blockIdx.y + threadIdx.y;
    }
    else if (order == 3)
    {
        row = blockDim.z * blockIdx.z + threadIdx.z;
        col = blockDim.x * blockIdx.x + threadIdx.x;
    }
    else if (order == 4)
    {
        row = blockDim.y * blockIdx.y + threadIdx.y;
        col = blockDim.x * blockIdx.x + threadIdx.x;
    }
    else if (order == 5)
    {
        row = blockDim.y * blockIdx.y + threadIdx.y;
        col = blockDim.z * blockIdx.z + threadIdx.z;
    }
    else if (order == 6)
    {
        row = blockDim.x * blockIdx.x + threadIdx.x;
        col = blockDim.z * blockIdx.z + threadIdx.z;
    }
    else if (order == 7)
    {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        row = idx / N;
        col = idx % N;
    }
    if (row < M && col < N)
    {
        for (int k = 0; k < K; k++)
        {
            arrC[row * N + col] += arrA[row * K + k] * arrB[k * N + col];
        }
    }
}

void multiplyGpu(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N)
{
    dim3 dimBlock;
    dim3 dimGrid;
    // order 1
    result_reset();
    dimBlock = dim3(TILE_WIDTH, TILE_WIDTH, 1);
    dimGrid = dim3((M + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH, 1);

    __TIME_BEGIN
    _matrixMul<<<dimGrid, dimBlock>>>(arrA, arrB, arrC, M, K, N, 1);
    __TIME_END

    cudaMemcpy(arrayC_h, arrC, M * N * sizeof(fp), cudaMemcpyDeviceToHost);
    if (!compare_matrix(arrayC_h, arrayC_href, M, N))
    {
        std::cout << "1. Error at multiplyGpu order 1" << std::endl;
    }
    else
    {
        printf("1. Pass, GPU calculation time (without shared memory order 1) = %f ms\n", elapsedTime);
    }

    // order 2
    result_reset();
    dimBlock = dim3(1, TILE_WIDTH, TILE_WIDTH);
    dimGrid = dim3(1, (N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    __TIME_BEGIN
    _matrixMul<<<dimGrid, dimBlock>>>(arrA, arrB, arrC, M, K, N, 2);
    __TIME_END

    cudaMemcpy(arrayC_h, arrC, M * N * sizeof(fp), cudaMemcpyDeviceToHost);
    if (!compare_matrix(arrayC_h, arrayC_href, M, N))
    {
        std::cout << "1. Error at multiplyGpu order 2" << std::endl;
    }
    else
    {
        printf("1. Pass, GPU calculation time (without shared memory order 2) = %f ms\n", elapsedTime);
    }

    // order 3
    result_reset();
    dimBlock = dim3(TILE_WIDTH, 1, TILE_WIDTH);
    dimGrid = dim3((N + TILE_WIDTH - 1) / TILE_WIDTH, 1, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    __TIME_BEGIN
    _matrixMul<<<dimGrid, dimBlock>>>(arrA, arrB, arrC, M, K, N, 3);
    __TIME_END

    cudaMemcpy(arrayC_h, arrC, M * N * sizeof(fp), cudaMemcpyDeviceToHost);
    if (!compare_matrix(arrayC_h, arrayC_href, M, N))
    {
        std::cout << "1. Error at multiplyGpu order 3" << std::endl;
    }
    else
    {
        printf("1. Pass, GPU calculation time (without shared memory order 3) = %f ms\n", elapsedTime);
    }

    // order 4
    result_reset();
    dimBlock = dim3(TILE_WIDTH, TILE_WIDTH, 1);
    dimGrid = dim3((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH, 1);

    __TIME_BEGIN
    _matrixMul<<<dimGrid, dimBlock>>>(arrA, arrB, arrC, M, K, N, 4);
    __TIME_END

    cudaMemcpy(arrayC_h, arrC, M * N * sizeof(fp), cudaMemcpyDeviceToHost);
    if (!compare_matrix(arrayC_h, arrayC_href, M, N))
    {
        std::cout << "1. Error at multiplyGpu order 4" << std::endl;
    }
    else
    {
        printf("1. Pass, GPU calculation time (without shared memory order 4) = %f ms\n", elapsedTime);
    }

    // order 5
    result_reset();
    dimBlock = dim3(1, TILE_WIDTH, TILE_WIDTH);
    dimGrid = dim3(1, (M + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    __TIME_BEGIN
    _matrixMul<<<dimGrid, dimBlock>>>(arrA, arrB, arrC, M, K, N, 5);
    __TIME_END

    cudaMemcpy(arrayC_h, arrC, M * N * sizeof(fp), cudaMemcpyDeviceToHost);
    if (!compare_matrix(arrayC_h, arrayC_href, M, N))
    {
        std::cout << "1. Error at multiplyGpu order 5" << std::endl;
    }
    else
    {
        printf("1. Pass, GPU calculation time (without shared memory order 5) = %f ms\n", elapsedTime);
    }

    // order 6
    result_reset();
    dimBlock = dim3(TILE_WIDTH, 1, TILE_WIDTH);
    dimGrid = dim3((M + TILE_WIDTH - 1) / TILE_WIDTH, 1, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    __TIME_BEGIN
    _matrixMul<<<dimGrid, dimBlock>>>(arrA, arrB, arrC, M, K, N, 6);
    __TIME_END

    cudaMemcpy(arrayC_h, arrC, M * N * sizeof(fp), cudaMemcpyDeviceToHost);
    if (!compare_matrix(arrayC_h, arrayC_href, M, N))
    {
        std::cout << "1. Error at multiplyGpu order 6" << std::endl;
    }
    else
    {
        printf("1. Pass, GPU calculation time (without shared memory order 6) = %f ms\n", elapsedTime);
    }

    // order 7, use id grid & block
    result_reset();
    dimBlock = dim3(TILE_WIDTH * TILE_WIDTH, 1, 1);
    dimGrid = dim3((M * N + dimBlock.x - 1) / dimBlock.x, 1, 1);

    __TIME_BEGIN
    _matrixMul<<<dimGrid, dimBlock>>>(arrA, arrB, arrC, M, K, N, 7);
    __TIME_END

    cudaMemcpy(arrayC_h, arrC, M * N * sizeof(fp), cudaMemcpyDeviceToHost);
    if (!compare_matrix(arrayC_h, arrayC_href, M, N))
    {
        std::cout << "1. Error at multiplyGpu order 7" << std::endl;
    }
    else
    {
        printf("1. Pass, GPU calculation time (without shared memory order 7) = %f ms\n", elapsedTime);
    }
}

__device__ void _setVecVal(fp4 & a, fp val){
    a.x = val;
    a.y = val;
    a.z = val;
    a.w = val;
}

__global__ void _matrixVecLoadH(const fp *src, fp4 *dst, int M, int K4, int K)
{
    // absolute row and col
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int row = idx / K4;
    unsigned int col = idx % K4;
    if (row >= M || col >= K4)
        return;
    unsigned int offset = row * K4 + col;
    unsigned int offsetSrc = row * K + col * 4;
    _setVecVal(dst[offset], 0.0f);

    if (col * 4 + 3 < K)
    {
        dst[offset].x = src[offsetSrc];
        dst[offset].y = src[offsetSrc + 1];
        dst[offset].z = src[offsetSrc + 2];
        dst[offset].w = src[offsetSrc + 3];
    }
    else if (col * 4 + 2 < K)
    {
        dst[offset].x = src[offsetSrc];
        dst[offset].y = src[offsetSrc + 1];
        dst[offset].z = src[offsetSrc + 2];
    }
    else if (col * 4 + 1 < K)
    {
        dst[offset].x = src[offsetSrc];
        dst[offset].y = src[offsetSrc + 1];
    }
    else
    {
        dst[offset].x = src[offsetSrc];
    }
}

__global__ void _matrixVecLoadV(const fp *src, fp4 *dst, int K4, int N, int K)
{
    // absolute row and col
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int row = idx / N;
    unsigned int col = idx % N;
    if (row >= K4 || col >= N)
        return;
    unsigned int offset = row * N + col;
    unsigned int offsetSrc = row * 4 * N + col;
    _setVecVal(dst[offset], 0.0f);

    if (row * 4 + 3 < K)
    {
        dst[offset].x = src[offsetSrc];
        dst[offset].y = src[offsetSrc + N];
        dst[offset].z = src[offsetSrc + 2 * N];
        dst[offset].w = src[offsetSrc + 3 * N];
    }
    else if (row * 4 + 2 < K)
    {
        dst[offset].x = src[offsetSrc];
        dst[offset].y = src[offsetSrc + N];
        dst[offset].z = src[offsetSrc + 2 * N];
    }
    else if (row * 4 + 1 < K)
    {
        dst[offset].x = src[offsetSrc];
        dst[offset].y = src[offsetSrc + N];
    }
    else
    {
        dst[offset].x = src[offsetSrc];
    }
}

__device__ fp dot_prod(const fp4 &a, const fp4 &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__global__ void _matrixMulVec(const fp4 *arrA, const fp4 *arrB, fp *arrC, int M, int K4, int N)
{
    // absolute row and col
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int row = idx / N;
    unsigned int col = idx % N;

    if (row < M && col < N)
    {
        fp sum = 0;
        for (int k = 0; k < K4; k++)
        {
            sum += dot_prod(arrA[row * K4 + k], arrB[k * N + col]);
        }
        arrC[row * N + col] = sum;
    }
}

__global__ void _matrixMulVecSh(const fp4 *arrA, const fp4 *arrB, fp *arrC, int M, int K4, int N)
{
    // absolute row and col
    unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int col = blockDim.y * blockIdx.y + threadIdx.y;

    __shared__ fp4 arrAs[TILE_WIDTH * TILE_WIDTH];
    __shared__ fp4 arrBs[TILE_WIDTH * TILE_WIDTH];

    fp elementC = 0.0f;
    for (int i = 0; i < K4 / TILE_WIDTH; i++)
    {
        if (i * TILE_WIDTH + threadIdx.y < K4)
            arrAs[threadIdx.y * TILE_WIDTH + threadIdx.x] = arrA[row * K4 + i * TILE_WIDTH + threadIdx.y];
        else
            _setVecVal(arrAs[threadIdx.y * TILE_WIDTH + threadIdx.x], 0.0f);
        if (i * TILE_WIDTH + threadIdx.x < K4)
            arrBs[threadIdx.y * TILE_WIDTH + threadIdx.x] = arrB[(i * TILE_WIDTH + threadIdx.x) * N + col];
        else
            _setVecVal(arrBs[threadIdx.y * TILE_WIDTH + threadIdx.x], 0.0f);
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; k++)
            elementC += dot_prod(arrAs[k * TILE_WIDTH + threadIdx.x], arrBs[threadIdx.y * TILE_WIDTH + k]);
        __syncthreads();
    }

    if (row < M && col < N)
        arrC[row * N + col] = elementC;
}

#define USE_SHARED_MEMORY_VEC
void multiplyGpuVec(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N)
{
    result_reset();
    fp4 *arrayA_d4, *arrayB_d4;
    int K4 = (K + 3) / 4;
    cudaMalloc((void **)&arrayA_d4, M * K4 * sizeof(fp4));
    cudaMalloc((void **)&arrayB_d4, K4 * N * sizeof(fp4));

    dim3 dimBlock;
    dim3 dimGrid;

    __TIME_BEGIN
    // load matrix values into vector type
    dimBlock = dim3(TILE_WIDTH * TILE_WIDTH, 1, 1);
    dimGrid = dim3((M * K4 + dimBlock.x - 1) / dimBlock.x, 1, 1);
    _matrixVecLoadH<<<dimGrid, dimBlock>>>(arrA, arrayA_d4, M, K4, K);

    dimBlock = dim3(TILE_WIDTH * TILE_WIDTH, 1, 1);
    dimGrid = dim3((K4 * N + dimBlock.x - 1) / dimBlock.x, 1, 1);
    _matrixVecLoadV<<<dimGrid, dimBlock>>>(arrB, arrayB_d4, K4, N, K);

#ifdef USE_SHARED_MEMORY_VEC
    dimBlock = dim3(TILE_WIDTH, TILE_WIDTH, 1);
    dimGrid = dim3((M + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH, 1);
    _matrixMulVecSh<<<dimGrid, dimBlock>>>(arrayA_d4, arrayB_d4, arrC, M, K4, N);
#else
    dimBlock = dim3(TILE_WIDTH * TILE_WIDTH, 1, 1);
    dimGrid = dim3((M * N + dimBlock.x - 1) / dimBlock.x, 1, 1);
    _matrixMulVec<<<dimGrid, dimBlock>>>(arrayA_d4, arrayB_d4, arrC, M, K4, N);
#endif
    __TIME_END

    cudaMemcpy(arrayC_h, arrC, M * N * sizeof(fp), cudaMemcpyDeviceToHost);
    if (!compare_matrix(arrayC_h, arrayC_href, M, N))
    {
        std::cout << "1. Error at multiplyGpuVec" << std::endl;
    }
    else
    {
        printf("1. Pass, GPU calculation time (with float4 type) = %f ms\n", elapsedTime);
    }
    cudaFree(arrayA_d4);
    cudaFree(arrayB_d4);
}

__global__ void _matrixMulSh(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N)
{
    // absolute row and col
    unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int col = blockDim.y * blockIdx.y + threadIdx.y;

    __shared__ fp arrAs[TILE_WIDTH * TILE_WIDTH];
    __shared__ fp arrBs[TILE_WIDTH * TILE_WIDTH];
    fp elementC = 0;

    for (int i = 0; i < (K + TILE_WIDTH - 1) / TILE_WIDTH; i++)
    {
        if (i * TILE_WIDTH + threadIdx.y < K)
            arrAs[threadIdx.y * TILE_WIDTH + threadIdx.x] = arrA[row * K + i * TILE_WIDTH + threadIdx.y];
        else
            arrAs[threadIdx.y * TILE_WIDTH + threadIdx.x] = 0;
        if (i * TILE_WIDTH + threadIdx.x < K)
            arrBs[threadIdx.y * TILE_WIDTH + threadIdx.x] = arrB[(i * TILE_WIDTH + threadIdx.x) * N + col];
        else
            arrBs[threadIdx.y * TILE_WIDTH + threadIdx.x] = 0;
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; k++)
            elementC += arrAs[k * TILE_WIDTH + threadIdx.x] * arrBs[threadIdx.y * TILE_WIDTH + k];
        __syncthreads();
    }
    if (row < M && col < N)
        arrC[row * N + col] = elementC;
}

void multiplyGpuSh(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N)
{
    result_reset();
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid((M + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH, 1);

    __TIME_BEGIN
    _matrixMulSh<<<dimGrid, dimBlock>>>(arrA, arrB, arrC, M, K, N);
    __TIME_END

    cudaMemcpy(arrayC_h, arrC, M * N * sizeof(fp), cudaMemcpyDeviceToHost);
    if (!compare_matrix(arrayC_h, arrayC_href, M, N))
    {
        std::cout << "2. Error at multiplyGpuSh" << std::endl;
    }
    else
    {
        printf("2. Pass, GPU calculation time (with shared memory) = %f ms\n", elapsedTime);
    }
}

__global__ void _matrixMulSh2(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N)
{
    // absolute row and col
    unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int col = blockDim.y * blockIdx.y + threadIdx.y;
    extern __shared__ fp arrAs[];
    fp *arrBs = arrAs + blockDim.x * blockDim.y;
    fp elementC = 0;

    for (int i = 0; i < (K + TILE_WIDTH - 1) / TILE_WIDTH; i++)
    {
        if (i * TILE_WIDTH + threadIdx.y < K)
            arrAs[threadIdx.y * TILE_WIDTH + threadIdx.x] = arrA[row * K + i * TILE_WIDTH + threadIdx.y];
        else
            arrAs[threadIdx.y * TILE_WIDTH + threadIdx.x] = 0;
        if (i * TILE_WIDTH + threadIdx.x < K)
            arrBs[threadIdx.y * TILE_WIDTH + threadIdx.x] = arrB[(i * TILE_WIDTH + threadIdx.x) * N + col];
        else
            arrBs[threadIdx.y * TILE_WIDTH + threadIdx.x] = 0;
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; k++)
            elementC += arrAs[k * TILE_WIDTH + threadIdx.x] * arrBs[threadIdx.y * TILE_WIDTH + k];
        __syncthreads();
    }
    if (row < M && col < N)
        arrC[row * N + col] = elementC;
}

void multiplyGpuSh2(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N)
{
    result_reset();
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid((M + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH, 1);

    __TIME_BEGIN
    _matrixMulSh2<<<dimGrid, dimBlock, TILE_WIDTH * TILE_WIDTH * sizeof(fp) * 2>>>(arrA, arrB, arrC, M, K, N);
    __TIME_END

    cudaMemcpy(arrayC_h, arrC, M * N * sizeof(fp), cudaMemcpyDeviceToHost);
    if (!compare_matrix(arrayC_h, arrayC_href, M, N))
    {
        std::cout << "2.1 Error at multiplyGpuSh2" << std::endl;
    }
    else
    {
        printf("2.1 Pass, GPU calculation time (with shared memory) = %f ms\n", elapsedTime);
    }
}

__global__ void _matrixMulShBc(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N)
{
    // absolute row and col
    unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int col = blockDim.y * blockIdx.y + threadIdx.y;
    __shared__ fp arrAs[TILE_WIDTH * TILE_WIDTH];
    __shared__ fp arrBs[TILE_WIDTH * TILE_WIDTH];
    fp elementC = 0;

    for (int i = 0; i < (K + TILE_WIDTH - 1) / TILE_WIDTH; i++)
    {
        if (i * TILE_WIDTH + threadIdx.y < K)
            arrAs[threadIdx.x * TILE_WIDTH + threadIdx.y] = arrA[row * K + i * TILE_WIDTH + threadIdx.y];
        else
            arrAs[threadIdx.x * TILE_WIDTH + threadIdx.y] = 0;
        if (i * TILE_WIDTH + threadIdx.x < K)
            arrBs[threadIdx.x * TILE_WIDTH + threadIdx.y] = arrB[(i * TILE_WIDTH + threadIdx.x) * N + col];
        else
            arrBs[threadIdx.x * TILE_WIDTH + threadIdx.y] = 0;
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; k++)
            elementC += arrAs[threadIdx.x * TILE_WIDTH + k] * arrBs[k * TILE_WIDTH + threadIdx.y];
        __syncthreads();
    }
    if (row < M && col < N)
        arrC[row * N + col] = elementC;
}

void multiplyGpuShBc(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N)
{
    result_reset();
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid((M + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH, 1);

    __TIME_BEGIN
    _matrixMulShBc<<<dimGrid, dimBlock>>>(arrA, arrB, arrC, M, K, N);
    __TIME_END

    cudaMemcpy(arrayC_h, arrC, M * N * sizeof(fp), cudaMemcpyDeviceToHost);
    if (!compare_matrix(arrayC_h, arrayC_href, M, N))
    {
        std::cout << "3. Error at multiplyGpuShBc" << std::endl;
    }
    else
    {
        printf("3. Pass, GPU calculation time (with shared memory, bank conflict) = %f ms\n", elapsedTime);
    }
}

__global__ void _matrixMulShBcPd(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N)
{
    // absolute row and col
    unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int col = blockDim.y * blockIdx.y + threadIdx.y;
    __shared__ fp arrAs[TILE_WIDTH * (TILE_WIDTH + 1)];
    __shared__ fp arrBs[TILE_WIDTH * (TILE_WIDTH + 1)];
    fp elementC = 0;

    for (int i = 0; i < (K + TILE_WIDTH - 1) / TILE_WIDTH; i++)
    {
        if (i * TILE_WIDTH + threadIdx.y < K)
            arrAs[threadIdx.x * (TILE_WIDTH + 1) + threadIdx.y] = arrA[row * K + i * TILE_WIDTH + threadIdx.y];
        else
            arrAs[threadIdx.x * (TILE_WIDTH + 1) + threadIdx.y] = 0;
        if (i * TILE_WIDTH + threadIdx.x < K)
            arrBs[threadIdx.x * (TILE_WIDTH + 1) + threadIdx.y] = arrB[(i * TILE_WIDTH + threadIdx.x) * N + col];
        else
            arrBs[threadIdx.x * (TILE_WIDTH + 1) + threadIdx.y] = 0;
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; k++)
            elementC += arrAs[threadIdx.x * (TILE_WIDTH + 1) + k] * arrBs[k * (TILE_WIDTH + 1) + threadIdx.y];
        __syncthreads();
    }
    if (row < M && col < N)
        arrC[row * N + col] = elementC;
}

void multiplyGpuShBcPd(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N)
{
    result_reset();
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid((M + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH, 1);

    __TIME_BEGIN
    _matrixMulShBcPd<<<dimGrid, dimBlock>>>(arrA, arrB, arrC, M, K, N);
    __TIME_END

    cudaMemcpy(arrayC_h, arrC, M * N * sizeof(fp), cudaMemcpyDeviceToHost);
    if (!compare_matrix(arrayC_h, arrayC_href, M, N))
    {
        std::cout << "4. Error at multiplyGpuShBcPd" << std::endl;
    }
    else
    {
        printf("4. Pass, GPU calculation time (with shared memory, bank conflict fixed with padding) = %f ms\n", elapsedTime);
    }
}

void multiplyGpuAcc(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N)
{
    result_reset();
#pragma acc kernels
    {
        ;
    }
#pragma acc enter data copyin(arrA [0:M * K], arrB [0:K * N], arrC [0:M * N])
    __TIME_BEGIN
#pragma acc kernels default(present)
#pragma acc loop independent collapse(2)
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

#pragma acc update host(arrC [0:M * N])
#pragma acc exit data delete (arrA [0:M * K], arrB [0:K * N], arrC [0:M * N])
    if (!compare_matrix(arrC, arrayC_href, M, N))
    {
        std::cout << "5. Error at multiplyCpuAcc" << std::endl;
    }
    else
    {
        printf("5. Pass, GPU calculation time (OpenACC) = %f ms\n", elapsedTime);
    }
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
