//==============================================================
// Copyright © 2023 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// haowei.zhang@intel.com

// HPCSDK: nvc++ -acc -mp=gpu -gpu=cc70 -Minfo=all -I/usr/local/cuda/targets/x86_64-linux/include -L/usr/local/cuda/lib64 -lcublas matrixMultiplyCUDA_GEMM.cu // nvc++ not support for the WMMA due to incompatitable with -arch=sm_70
// CUDA Toolkit: nvcc -arch=sm_70 -lcublas matrixMultiplyCUDA_GEMM.cu

#include <iostream>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cublas_v2.h>
#include <mma.h>
// #include <omp.h>

#define __TIME_BEGIN cudaEventRecord(start);
#define __TIME_END              \
    cudaEventRecord(stop);      \
    cudaEventSynchronize(stop); \
    cudaEventElapsedTime(&elapsedTime, start, stop);

// #define cudaErrCheck(stat)                         \
//     {                                              \
//         cudaErrCheck_((stat), __FILE__, __LINE__); \
//     }
// void cudaErrCheck_(cudaError_t stat, const char *file, int line)
// {
//     if (stat != cudaSuccess)
//     {
//         fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
//     }
// }

// #define cublasErrCheck(stat)                         \
//     {                                                \
//         cublasErrCheck_((stat), __FILE__, __LINE__); \
//     }
// void cublasErrCheck_(cublasStatus_t stat, const char *file, int line)
// {
//     if (stat != CUBLAS_STATUS_SUCCESS)
//     {
//         fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
//     }
// }

// #define curandErrCheck(stat)                         \
//     {                                                \
//         curandErrCheck_((stat), __FILE__, __LINE__); \
//     }
// void curandErrCheck_(curandStatus_t stat, const char *file, int line)
// {
//     if (stat != CURAND_STATUS_SUCCESS)
//     {
//         fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
//     }
// }
// The only dimensions currently supported by WMMA, TILE_WIDTH,N,K = 16
#define TILE_WIDTH 16
#define N_REPEAT 10000

typedef float fp;

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
bool compare_matrix(const fp *arr1, const fp *arr2, int M, int N, bool transpose = false);
void multiplyCpu(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N);
void multiplyGpu(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N);
void multiplyGpuSh(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N);
void multiplyGpuShBc(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N);
void multiplyGpuShBcPd(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N);
void multiplyGpuAcc(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N);
void multiplyGpuOmp(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N);
void multiplyGpuGemm(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N);
void multiplyGpuWmma(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N);

int main()
{
    resources_init();

    multiplyCpu(arrayA_h, arrayB_h, arrayC_href, M, K, N);

    multiplyGpu(arrayA_d, arrayB_d, arrayC_d, M, K, N);

    multiplyGpuSh(arrayA_d, arrayB_d, arrayC_d, M, K, N);

    multiplyGpuShBc(arrayA_d, arrayB_d, arrayC_d, M, K, N);

    multiplyGpuShBcPd(arrayA_d, arrayB_d, arrayC_d, M, K, N);

    multiplyGpuAcc(arrayA_h, arrayB_h, arrayC_h, M, K, N);

    multiplyGpuOmp(arrayA_h, arrayB_h, arrayC_h, M, K, N);

    multiplyGpuGemm(arrayA_d, arrayB_d, arrayC_d, M, K, N);

    multiplyGpuWmma(arrayA_d, arrayB_d, arrayC_d, M, K, N);

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
    // printf("There are %d OpenMP devices\n", omp_get_num_devices());
    // omp_set_default_device(0);
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
        std::cout << "row " << i << ": ";
        for (int j = 0; j < N; j++)
        {
            std::cout << arr[i * N + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

bool compare_matrix(const fp *arr1, const fp *arr2, int M, int N, bool transpose)
{
    fp err = 0;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (!transpose)
                err = fabs(arr1[i * N + j] - arr2[i * N + j]);
            else
                err = fabs(arr1[j * M + i] - arr2[i * N + j]);
            if (err > fabs(arr2[i * N + j] * 1.E-4))
            {
                printf("   Mismatch at i = %d, j = %d, err = %f, arr2 = %f\n", i, j, err, arr2[i * N + j]);
                // if (!transpose)
                //     print_matrix(arr1, M, N);
                // else
                //     print_matrix(arr1, N, M);
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

__global__ void _matrixMul(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N)
{
    // absolute row and col
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int row = idx / N;
    unsigned int col = idx % N;
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
    result_reset();
    dim3 dimBlock(TILE_WIDTH * TILE_WIDTH, 1, 1);
    dim3 dimGrid((M * N + dimBlock.x - 1) / dimBlock.x, 1, 1);

    __TIME_BEGIN
    _matrixMul<<<dimGrid, dimBlock>>>(arrA, arrB, arrC, M, K, N);
    __TIME_END

    cudaMemcpy(arrayC_h, arrC, M * N * sizeof(fp), cudaMemcpyDeviceToHost);
    if (!compare_matrix(arrayC_h, arrayC_href, M, N))
    {
        std::cout << "1. Error at multiplyGpu" << std::endl;
    }
    else
    {
        printf("1. Pass, GPU calculation time (without shared memory) = %f ms\n", elapsedTime);
    }
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

// ref: https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/
//      https://blog.csdn.net/u011197534/article/details/78378536
void multiplyGpuGemm(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N)
{
    result_reset();
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);
    // Use tensor cores
    cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH);
    bool transpose = false;
    __TIME_BEGIN
    // Now using cuBLAS
    // transpose = true;
    // cublasGemmEx(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T,
    //                             M, N, K,
    //                             &alpha,
    //                             arrA, CUDA_R_32F, K,
    //                             arrB, CUDA_R_32F, N,
    //                             &beta,
    //                             arrC, CUDA_R_32F, M,
    //                             CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP);
    transpose = false;
    for (int i = 0; i < N_REPEAT; i++)
    {
        cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                     N, M, K,
                     &alpha,
                     arrB, CUDA_R_32F, N,
                     arrA, CUDA_R_32F, K,
                     &beta,
                     arrC, CUDA_R_32F, N,
                     CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP);
    }
    __TIME_END
    cudaMemcpy(arrayC_h, arrC, M * N * sizeof(fp), cudaMemcpyDeviceToHost);
    if (!compare_matrix(arrayC_h, arrayC_href, M, N, transpose))
    {
        std::cout << "7. Error at multiplyGpuGemm" << std::endl;
    }
    else
    {
        printf("7. Pass, GPU calculation time (Gemm Tensor Cores) = %f ms\n", elapsedTime / N_REPEAT);
    }
    cublasDestroy(cublasHandle);
}

__global__ void _fp2half(half *arrTgt, const fp *arrSrc, int M, int N)
{
    unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int col = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int idx = row * N + col;
    if (row < M && col < N)
    {
        arrTgt[idx] = arrSrc[idx];
    }
}

__global__ void _matrixMulWmma(const half *arrA, const half *arrB, fp *arrC, int M, int K, int N, float alpha, float beta)
{
    // Leading dimensions. Packed with no transpositions. cols
    int lda = M;
    int ldb = K;
    int ldc = M;

    // Tile using a 2D grid
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // Declare the fragments
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, TILE_WIDTH, TILE_WIDTH, TILE_WIDTH, half, nvcuda::wmma::col_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, TILE_WIDTH, TILE_WIDTH, TILE_WIDTH, half, nvcuda::wmma::col_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, TILE_WIDTH, TILE_WIDTH, TILE_WIDTH, fp> acc_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, TILE_WIDTH, TILE_WIDTH, TILE_WIDTH, fp> c_frag;

    nvcuda::wmma::fill_fragment(acc_frag, 0.0f);

    // Loop over k
    for (int i = 0; i < K; i += TILE_WIDTH)
    {
        int aRow = warpM * TILE_WIDTH;
        int aCol = i;

        int bRow = i;
        int bCol = warpN * TILE_WIDTH;

        // Bounds checking
        if (aRow < M && aCol < K && bRow < K && bCol < N)
        {
            // Load the inputs
            nvcuda::wmma::load_matrix_sync(a_frag, arrA + aRow + aCol * lda, lda);
            nvcuda::wmma::load_matrix_sync(b_frag, arrB + bRow + bCol * ldb, ldb);

            // Perform the matrix multiplication
            nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
    int cRow = warpM * TILE_WIDTH;
    int cCol = warpN * TILE_WIDTH;

    if (cRow < M && cCol < N)
    {
        nvcuda::wmma::load_matrix_sync(c_frag, arrC + cRow + cCol * ldc, ldc, nvcuda::wmma::mem_col_major);

        for (int i = 0; i < c_frag.num_elements; i++)
        {
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }

        // Store the output
        nvcuda::wmma::store_matrix_sync(arrC + cRow + cCol * ldc, c_frag, ldc, nvcuda::wmma::mem_col_major);
    }
}

void multiplyGpuWmma(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N)
{
    half *arrAfp16;
    half *arrBfp16;
    cudaMalloc((void **)&arrAfp16, M * K * sizeof(half));
    cudaMalloc((void **)&arrBfp16, K * N * sizeof(half));

    dim3 dimBlockA(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGridA((M + TILE_WIDTH - 1) / TILE_WIDTH, (K + TILE_WIDTH - 1) / TILE_WIDTH, 1);
    _fp2half<<<dimGridA, dimBlockA>>>(arrAfp16, arrA, M, K);

    dim3 dimBlockB(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGridB((K + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH, 1);
    _fp2half<<<dimGridB, dimBlockB>>>(arrBfp16, arrB, K, N);

    result_reset();
    float alpha = 1.0f;
    float beta = 0.0f;
    bool transpose = false;

    // blockDim.x must be a multple of warpSize
    // 128x4 means we have 16 warps and a block computes a 64x64 output tile
    const int warpSz = 32;
    dim3 dimBlock(4 * warpSz, 4, 1);
    dim3 dimGrid((N + (TILE_WIDTH * dimBlock.x / warpSz - 1)) / (TILE_WIDTH * dimBlock.x / warpSz), (M + TILE_WIDTH * dimBlock.y - 1) / (TILE_WIDTH * dimBlock.y), 1);

    __TIME_BEGIN
    for (int i = 0; i < N_REPEAT; i++)
    {
        _matrixMulWmma<<<dimGrid, dimBlock>>>(arrBfp16, arrAfp16, arrC, N, K, M, alpha, beta);
    }
    __TIME_END

    cudaMemcpy(arrayC_h, arrC, M * N * sizeof(fp), cudaMemcpyDeviceToHost);
    if (!compare_matrix(arrayC_h, arrayC_href, M, N, transpose))
    {
        std::cout << "8. Error at multiplyGpuWmma" << std::endl;
    }
    else
    {
        printf("8. Pass, GPU calculation time (Wmma Tensor Cores) = %f ms\n", elapsedTime / N_REPEAT);
    }
    cudaFree(arrAfp16);
    cudaFree(arrBfp16);
}