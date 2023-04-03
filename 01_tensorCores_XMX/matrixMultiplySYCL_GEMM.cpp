//==============================================================
// Copyright © 2023 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// haowei.zhang@intel.com

// transfer from CUDA with DPCT: dpct --in-root=. --cuda-include-path=/usr/local/cuda-11.7/include matrixMultiplyCUDA.cu
// run on CPU, Intel & NVIDIA GPU
// icpx -fsycl -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core matrixMultiplySYCL_GEMM.cpp
// icpx -fsycl -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -fiopenmp -fopenmp-targets=spir64 matrixMultiplySYCL_GEMM.cpp
// clang++ -fsycl -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core matrixMultiplySYCL_GEMM.cpp
// clang++ -fsycl -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -fiopenmp -fopenmp-targets=spir64 matrixMultiplySYCL_GEMM.cpp
// clang++ -fsycl -I${MKLROOT}/include -L${MKLROOT}/lib -lonemkl matrixMultiplySYCL_GEMM.cpp
// clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_70 -I${MKLROOT}/include -L${MKLROOT}/lib -lonemkl matrixMultiplySYCL_GEMM.cpp
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <chrono>
#include <cmath>
#include <cstring>
#include <oneapi/mkl.hpp>
// #include <dpct/blas_utils.hpp>
// #include <dpct/lib_common_utils.hpp>

/*
DPCT1012:4: Detected kernel execution time measurement pattern and generated an
initial code for time measurements in SYCL. You can change the way time is
measured depending on your goals.
*/
#define __TIME_BEGIN                              \
    start_ct1 = std::chrono::steady_clock::now(); \
    *start = q_ct1.ext_oneapi_submit_barrier();
/*
DPCT1012:5: Detected kernel execution time measurement pattern and generated an
initial code for time measurements in SYCL. You can change the way time is
measured depending on your goals.
*/
#define __TIME_END                               \
    stop_ct1 = std::chrono::steady_clock::now(); \
    *stop = q_ct1.ext_oneapi_submit_barrier();   \
    stop->wait_and_throw();                      \
    elapsedTime =                                \
        std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();

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

#define TILE_WIDTH 16
#define SUB_GRP_SZ 32
#define N_REPEAT 10000

typedef float fp;

static int M = 2048;
static int K = 1024;
static int N = 512;

fp *arrayA_h, *arrayB_h, *arrayC_h, *arrayC_href;
fp *arrayA_d, *arrayB_d, *arrayC_d;
dpct::event_ptr start, stop;
std::chrono::time_point<std::chrono::steady_clock> start_ct1;
std::chrono::time_point<std::chrono::steady_clock> stop_ct1;

dpct::device_ext &dev_ct1 = dpct::get_current_device();
sycl::queue &q_ct1 = dev_ct1.default_queue();

float elapsedTime;

void resources_init();
void result_reset();
void resources_free();
void print_matrix(const fp *arr, int M, int N);
bool compare_matrix(const fp *arr1, const fp *arr2, int M, int N, bool transpose = false);
void multiplyCpu(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N);
void warmupGpu(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N);
void multiplyGpu(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N);
void multiplyGpuSh(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N);
void multiplyGpuShBc(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N);
void multiplyGpuShBcPd(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N);
void multiplyGpuAcc(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N);
void multiplyGpuOmp(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N);
void multiplyGpuGemm(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N);

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

    resources_free();
    return 0;
}

void resources_init()
{
    // https://www.intel.com/content/www/us/en/developer/articles/technical/device-discovery-with-sycl.html#gs.pmah8j
    // for (auto platform : sycl::platform::get_platforms())
    // {
    //     std::cout << "Platform: "
    //               << platform.get_info<sycl::info::platform::name>()
    //               << std::endl;

    //     for (auto device : platform.get_devices())
    //     {
    //         std::cout << "\tDevice: "
    //                   << device.get_info<sycl::info::device::name>()
    //                   << std::endl;
    //     }
    // }

    // auto platforms = sycl::platform::get_platforms();
    // q_ct1 = sycl::queue(platforms[1].get_devices()[0]);

    std::cout << "Selected device: " << q_ct1.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << "Device vendor: " << q_ct1.get_device().get_info<sycl::info::device::vendor>() << "\n";
    std::cout << "Max group/block size = " << q_ct1.get_device().get_info<sycl::info::device::max_work_group_size>() << "\n";
    std::cout << "Max Compute Units = " << q_ct1.get_device().get_info<sycl::info::device::max_compute_units>() << "\n";
    std::cout << "Shared Local Memory size = " << q_ct1.get_device().get_info<sycl::info::device::local_mem_size>() << " Bytes\n";
    std::cout << "Sub-group Sizes: ";
    for (const auto &s : q_ct1.get_device().get_info<sycl::info::device::sub_group_sizes>())
        std::cout << s << " ";
    std::cout << std::endl;
    arrayA_h = new fp[M * K]();
    arrayB_h = new fp[K * N]();
    arrayC_h = new fp[M * N]();
    arrayC_href = new fp[M * N]();

    arrayA_d = sycl::malloc_device<fp>(M * K, q_ct1);
    arrayB_d = sycl::malloc_device<fp>(K * N, q_ct1);
    arrayC_d = sycl::malloc_device<fp>(M * N, q_ct1);

    for (int i = 0; i < M * K; i++)
        arrayA_h[i] = rand() / (fp)RAND_MAX * 1.0;

    for (int i = 0; i < K * N; i++)
        arrayB_h[i] = rand() / (fp)RAND_MAX * 1.0;

    memset(arrayC_h, 0, M * N * sizeof(fp));
    memset(arrayC_href, 0, M * N * sizeof(fp));

    q_ct1.memcpy(arrayA_d, arrayA_h, M * K * sizeof(fp));
    q_ct1.memcpy(arrayB_d, arrayB_h, K * N * sizeof(fp)).wait();
    q_ct1.memset(arrayC_d, 0, M * N * sizeof(fp)).wait();
    start = new sycl::event();
    stop = new sycl::event();
}

void result_reset()
{
    memset(arrayC_h, 0, M * N * sizeof(fp));
    q_ct1.memset(arrayC_d, 0, M * N * sizeof(fp)).wait();
}

void resources_free()
{
    delete[] arrayA_h;
    delete[] arrayB_h;
    delete[] arrayC_h;
    delete[] arrayC_href;
    sycl::free(arrayA_d, q_ct1);
    sycl::free(arrayB_d, q_ct1);
    sycl::free(arrayC_d, q_ct1);
    dpct::destroy_event(start);
    dpct::destroy_event(stop);
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

void _matrixMul(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N,
                sycl::nd_item<3> item_ct1)
{
    // absolute row and col
    int idx = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
              item_ct1.get_local_id(2);
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

void warmupGpu(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N)
{
    result_reset();
    sycl::range<3> dimBlock(1, 1, TILE_WIDTH * TILE_WIDTH);
    sycl::range<3> dimGrid(1, 1, (N * M + dimBlock[2] - 1) / dimBlock[2]);

    __TIME_BEGIN
    *stop =
        q_ct1.parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                           [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(SUB_GRP_SZ)]]
                           {
                               _matrixMul(arrA, arrB, arrC, M, K, N, item_ct1);
                           });
    stop->wait();

    float alpha = 1.0f;
    float beta = 0.0f;

    // dpct::gemm(q_ct1, oneapi::mkl::transpose::nontrans,
    //            oneapi::mkl::transpose::nontrans, N, M, K, &alpha, arrB,
    //            dpct::library_data_t::real_float, N, arrA,
    //            dpct::library_data_t::real_float, K, &beta, arrC,
    //            dpct::library_data_t::real_float, N,
    //            dpct::library_data_t::real_float);
    oneapi::mkl::blas::column_major::gemm(q_ct1, oneapi::mkl::transpose::nontrans,
                            oneapi::mkl::transpose::nontrans,
                            N, M, K, alpha, arrB, N, arrA, K, beta, arrC, N);
    q_ct1.wait();
    result_reset();

    __TIME_END

    // printf("0. warmupGpu time = %f ms\n", elapsedTime);
}

void multiplyGpu(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N)
{
    warmupGpu(arrA, arrB, arrC, M, K, N);
    result_reset();
    sycl::range<3> dimBlock(1, 1, TILE_WIDTH * TILE_WIDTH);
    sycl::range<3> dimGrid(1, 1, (N * M + dimBlock[2] - 1) / dimBlock[2]);

    __TIME_BEGIN
    /*
    DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    *stop =
        q_ct1.parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                           [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(SUB_GRP_SZ)]]
                           {
                               _matrixMul(arrA, arrB, arrC, M, K, N, item_ct1);
                           });
    stop->wait();
    __TIME_END

    q_ct1.memcpy(arrayC_h, arrC, M * N * sizeof(fp)).wait();
    if (!compare_matrix(arrayC_h, arrayC_href, M, N))
    {
        std::cout << "1. Error at multiplyGpu" << std::endl;
    }
    else
    {
        printf("1. Pass, GPU calculation time (without shared memory) = %f ms\n", elapsedTime);
    }
}

void _matrixMulSh(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N,
                  sycl::nd_item<3> item_ct1, fp *arrAs, fp *arrBs)
{
    // absolute row and col
    unsigned int row = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                       item_ct1.get_local_id(2);
    unsigned int col = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                       item_ct1.get_local_id(1);

    fp elementC = 0;

    for (int i = 0; i < (K + TILE_WIDTH - 1) / TILE_WIDTH; i++)
    {
        if (i * TILE_WIDTH + item_ct1.get_local_id(1) < K)
            arrAs[item_ct1.get_local_id(1) * TILE_WIDTH +
                  item_ct1.get_local_id(2)] =
                arrA[row * K + i * TILE_WIDTH + item_ct1.get_local_id(1)];
        else
            arrAs[item_ct1.get_local_id(1) * TILE_WIDTH +
                  item_ct1.get_local_id(2)] = 0;
        if (i * TILE_WIDTH + item_ct1.get_local_id(2) < K)
            arrBs[item_ct1.get_local_id(1) * TILE_WIDTH +
                  item_ct1.get_local_id(2)] =
                arrB[(i * TILE_WIDTH + item_ct1.get_local_id(2)) * N + col];
        else
            arrBs[item_ct1.get_local_id(1) * TILE_WIDTH +
                  item_ct1.get_local_id(2)] = 0;
        item_ct1.barrier(sycl::access::fence_space::local_space);
        for (int k = 0; k < TILE_WIDTH; k++)
            elementC += arrAs[k * TILE_WIDTH + item_ct1.get_local_id(2)] *
                        arrBs[item_ct1.get_local_id(1) * TILE_WIDTH + k];
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }
    if (row < M && col < N)
        arrC[row * N + col] = elementC;
}

void multiplyGpuSh(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N)
{
    result_reset();
    sycl::range<3> dimBlock(1, TILE_WIDTH, TILE_WIDTH);
    sycl::range<3> dimGrid(1, (N + TILE_WIDTH - 1) / TILE_WIDTH,
                           (M + TILE_WIDTH - 1) / TILE_WIDTH);

    __TIME_BEGIN
    /*
    DPCT1049:1: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    *stop = q_ct1.submit([&](sycl::handler &cgh)
                         {
        sycl::local_accessor<fp, 1> arrAs_acc_ct1(
            sycl::range<1>(256 /*TILE_WIDTH * TILE_WIDTH*/), cgh);
        sycl::local_accessor<fp, 1> arrBs_acc_ct1(
            sycl::range<1>(256 /*TILE_WIDTH * TILE_WIDTH*/), cgh);

        cgh.parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                         [=](sycl::nd_item<3> item_ct1) {
                             _matrixMulSh(arrA, arrB, arrC, M, K, N, item_ct1,
                                          arrAs_acc_ct1.get_pointer(),
                                          arrBs_acc_ct1.get_pointer());
                         }); });
    stop->wait();
    __TIME_END

    q_ct1.memcpy(arrayC_h, arrC, M * N * sizeof(fp)).wait();
    if (!compare_matrix(arrayC_h, arrayC_href, M, N))
    {
        std::cout << "2. Error at multiplyGpuSh" << std::endl;
    }
    else
    {
        printf("2. Pass, GPU calculation time (with shared memory) = %f ms\n", elapsedTime);
    }
}

void _matrixMulShBc(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N,
                    sycl::nd_item<3> item_ct1, fp *arrAs, fp *arrBs)
{
    // absolute row and col
    unsigned int row = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                       item_ct1.get_local_id(2);
    unsigned int col = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                       item_ct1.get_local_id(1);

    fp elementC = 0;

    for (int i = 0; i < (K + TILE_WIDTH - 1) / TILE_WIDTH; i++)
    {
        if (i * TILE_WIDTH + item_ct1.get_local_id(1) < K)
            arrAs[item_ct1.get_local_id(2) * TILE_WIDTH +
                  item_ct1.get_local_id(1)] =
                arrA[row * K + i * TILE_WIDTH + item_ct1.get_local_id(1)];
        else
            arrAs[item_ct1.get_local_id(2) * TILE_WIDTH +
                  item_ct1.get_local_id(1)] = 0;
        if (i * TILE_WIDTH + item_ct1.get_local_id(2) < K)
            arrBs[item_ct1.get_local_id(2) * TILE_WIDTH +
                  item_ct1.get_local_id(1)] =
                arrB[(i * TILE_WIDTH + item_ct1.get_local_id(2)) * N + col];
        else
            arrBs[item_ct1.get_local_id(2) * TILE_WIDTH +
                  item_ct1.get_local_id(1)] = 0;
        item_ct1.barrier(sycl::access::fence_space::local_space);
        for (int k = 0; k < TILE_WIDTH; k++)
            elementC += arrAs[item_ct1.get_local_id(2) * TILE_WIDTH + k] *
                        arrBs[k * TILE_WIDTH + item_ct1.get_local_id(1)];
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }
    if (row < M && col < N)
        arrC[row * N + col] = elementC;
}

void multiplyGpuShBc(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N)
{
    result_reset();
    sycl::range<3> dimBlock(1, TILE_WIDTH, TILE_WIDTH);
    sycl::range<3> dimGrid(1, (N + TILE_WIDTH - 1) / TILE_WIDTH,
                           (M + TILE_WIDTH - 1) / TILE_WIDTH);

    __TIME_BEGIN
    /*
    DPCT1049:2: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    *stop = q_ct1.submit([&](sycl::handler &cgh)
                         {
        sycl::local_accessor<fp, 1> arrAs_acc_ct1(
            sycl::range<1>(256 /*TILE_WIDTH * TILE_WIDTH*/), cgh);
        sycl::local_accessor<fp, 1> arrBs_acc_ct1(
            sycl::range<1>(256 /*TILE_WIDTH * TILE_WIDTH*/), cgh);

        cgh.parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                         [=](sycl::nd_item<3> item_ct1) {
                             _matrixMulShBc(arrA, arrB, arrC, M, K, N, item_ct1,
                                            arrAs_acc_ct1.get_pointer(),
                                            arrBs_acc_ct1.get_pointer());
                         }); });
    stop->wait();
    __TIME_END

    q_ct1.memcpy(arrayC_h, arrC, M * N * sizeof(fp)).wait();
    if (!compare_matrix(arrayC_h, arrayC_href, M, N))
    {
        std::cout << "3. Error at multiplyGpuShBc" << std::endl;
    }
    else
    {
        printf("3. Pass, GPU calculation time (with shared memory, bank conflict) = %f ms\n", elapsedTime);
    }
}

void _matrixMulShBcPd(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N,
                      sycl::nd_item<3> item_ct1, fp *arrAs, fp *arrBs)
{
    // absolute row and col
    unsigned int row = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                       item_ct1.get_local_id(2);
    unsigned int col = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                       item_ct1.get_local_id(1);

    fp elementC = 0;

    for (int i = 0; i < (K + TILE_WIDTH - 1) / TILE_WIDTH; i++)
    {
        if (i * TILE_WIDTH + item_ct1.get_local_id(1) < K)
            arrAs[item_ct1.get_local_id(2) * (TILE_WIDTH + 1) +
                  item_ct1.get_local_id(1)] =
                arrA[row * K + i * TILE_WIDTH + item_ct1.get_local_id(1)];
        else
            arrAs[item_ct1.get_local_id(2) * (TILE_WIDTH + 1) +
                  item_ct1.get_local_id(1)] = 0;
        if (i * TILE_WIDTH + item_ct1.get_local_id(2) < K)
            arrBs[item_ct1.get_local_id(2) * (TILE_WIDTH + 1) +
                  item_ct1.get_local_id(1)] =
                arrB[(i * TILE_WIDTH + item_ct1.get_local_id(2)) * N + col];
        else
            arrBs[item_ct1.get_local_id(2) * (TILE_WIDTH + 1) +
                  item_ct1.get_local_id(1)] = 0;
        item_ct1.barrier(sycl::access::fence_space::local_space);
        for (int k = 0; k < TILE_WIDTH; k++)
            elementC += arrAs[item_ct1.get_local_id(2) * (TILE_WIDTH + 1) + k] *
                        arrBs[k * (TILE_WIDTH + 1) + item_ct1.get_local_id(1)];
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }
    if (row < M && col < N)
        arrC[row * N + col] = elementC;
}

void multiplyGpuShBcPd(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N)
{
    result_reset();
    sycl::range<3> dimBlock(1, TILE_WIDTH, TILE_WIDTH);
    sycl::range<3> dimGrid(1, (N + TILE_WIDTH - 1) / TILE_WIDTH,
                           (M + TILE_WIDTH - 1) / TILE_WIDTH);

    __TIME_BEGIN
    /*
    DPCT1049:3: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    *stop = q_ct1.submit([&](sycl::handler &cgh)
                         {
        sycl::local_accessor<fp, 1> arrAs_acc_ct1(
            sycl::range<1>(272 /*TILE_WIDTH * (TILE_WIDTH + 1)*/), cgh);
        sycl::local_accessor<fp, 1> arrBs_acc_ct1(
            sycl::range<1>(272 /*TILE_WIDTH * (TILE_WIDTH + 1)*/), cgh);

        cgh.parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                         [=](sycl::nd_item<3> item_ct1) {
                             _matrixMulShBcPd(arrA, arrB, arrC, M, K, N,
                                              item_ct1,
                                              arrAs_acc_ct1.get_pointer(),
                                              arrBs_acc_ct1.get_pointer());
                         }); });
    stop->wait();
    __TIME_END

    q_ct1.memcpy(arrayC_h, arrC, M * N * sizeof(fp)).wait();
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
    // Use tensor cores
    /*
    DPCT1026:6: The call to cublasSetMathMode was removed because this call is
    redundant in SYCL.
    */
    bool transpose = false;
    __TIME_BEGIN
    // Now using cuBLAS
    // transpose = true;
    // dpct::gemm(q_ct1, oneapi::mkl::transpose::trans,
    //            oneapi::mkl::transpose::trans, M, N, K, &alpha, arrA,
    //            dpct::library_data_t::real_float, K, arrB,
    //            dpct::library_data_t::real_float, N, &beta, arrC,
    //            dpct::library_data_t::real_float, M,
    //            dpct::library_data_t::real_float);
    transpose = false;
    for (int i = 0; i < N_REPEAT; i++)
    {
        // dpct::gemm(q_ct1, oneapi::mkl::transpose::nontrans,
        //            oneapi::mkl::transpose::nontrans, N, M, K, &alpha, arrB,
        //            dpct::library_data_t::real_float, N, arrA,
        //            dpct::library_data_t::real_float, K, &beta, arrC,
        //            dpct::library_data_t::real_float, N,
        //            dpct::library_data_t::real_float);
        oneapi::mkl::blas::column_major::gemm(q_ct1, oneapi::mkl::transpose::nontrans,
                                oneapi::mkl::transpose::nontrans,
                                N, M, K, alpha, arrB, N, arrA, K, beta, arrC, N);
        // oneapi::mkl::blas::row_major::gemm(q_ct1, oneapi::mkl::transpose::nontrans, 
        //                         oneapi::mkl::transpose::nontrans, M, N, K, 
        //                         alpha, arrA, K, arrB, N, beta, arrC, N);
    }
    q_ct1.wait();
    __TIME_END
    q_ct1.memcpy(arrayC_h, arrC, M * N * sizeof(fp)).wait();
    if (!compare_matrix(arrayC_h, arrayC_href, M, N, transpose))
    {
        std::cout << "7. Error at multiplyGpuGemm" << std::endl;
    }
    else
    {
        printf("7. Pass, GPU calculation time (Gemm Tensor Cores) = %f ms\n", elapsedTime / N_REPEAT);
    }
}