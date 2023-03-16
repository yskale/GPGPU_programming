//==============================================================
// Copyright © 2023 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
// haowei.zhang@intel.com

// transfer from CUDA with DPCT: dpct --in-root=. --cuda-include-path=/usr/local/cuda-11.7/include matrixMultiplyCUDA.cu
// clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_70 matrixMultiplySYCL.cpp
// clang++ -fsycl -fsycl-targets=spir64_x86_64 matrixMultiplySYCL.cpp
// clang++ -fsycl -fsycl-targets=spir64 matrixMultiplySYCL.cpp
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <chrono>
#include <cmath>
#include <cstring>
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

#define TILE_WIDTH 16
#define SUB_GRP_SZ 32

typedef float fp;
typedef sycl::float4 fp4;

// const int M = 16;
// const int K = 16;
// const int N = 16;
const int M = 2048;
const int K = 1024;
const int N = 512;
constexpr int rangeMK = M * K;
constexpr int rangeKN = K * N;
constexpr int rangeMN = M * N;

fp *arrayA_h, *arrayB_h, *arrayC_h, *arrayC_href;
fp *arrayA_d, *arrayB_d, *arrayC_d;
dpct::event_ptr start, stop;
std::chrono::time_point<std::chrono::steady_clock> start_ct1;
std::chrono::time_point<std::chrono::steady_clock> stop_ct1;

dpct::device_ext &dev_ct1 = dpct::get_current_device();
sycl::queue &q_ct1 = dev_ct1.default_queue();
// sycl::usm_allocator<fp, sycl::usm::alloc::shared> alloc(q_ct1);

float elapsedTime;

void resources_init();
void result_reset();
void resources_free();
void print_matrix(const fp *arr, int M, int N);
bool compare_matrix(const fp *arr1, const fp *arr2, int M, int N);
void warmupGpu(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N);
void multiplyCpu(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N);
void multiplyGpuAccessor(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N);
void multiplyGpuGrp(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N);
void multiplyGpuGrpSh(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N);
void multiplyGpuBcast(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N);
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

    multiplyGpuAccessor(arrayA_h, arrayB_h, arrayC_h, M, K, N);

    multiplyGpuGrp(arrayA_d, arrayB_d, arrayC_d, M, K, N);

    multiplyGpuGrpSh(arrayA_d, arrayB_d, arrayC_d, M, K, N);

    multiplyGpuBcast(arrayA_d, arrayB_d, arrayC_d, M, K, N);

    multiplyGpuSh(arrayA_d, arrayB_d, arrayC_d, M, K, N);

    // multiplyGpuSh2(arrayA_d, arrayB_d, arrayC_d, M, K, N);

    multiplyGpuShBc(arrayA_d, arrayB_d, arrayC_d, M, K, N);

    multiplyGpuShBcPd(arrayA_d, arrayB_d, arrayC_d, M, K, N);

    // multiplyGpuAcc(arrayA_h, arrayB_h, arrayC_h, M, K, N);

    // multiplyGpuOmp(arrayA_h, arrayB_h, arrayC_h, M, K, N);

    resources_free();
    return 0;
}

#define LIST_ALL_DEVICES
void resources_init()
{
    // https://www.intel.com/content/www/us/en/developer/articles/technical/device-discovery-with-sycl.html#gs.pmah8j
#ifdef LIST_ALL_DEVICES
    int numPlatform = 0;
    for (auto const &platform : sycl::platform::get_platforms())
    {
        std::cout << "Platform[" << numPlatform++ << "]: "
                  << platform.get_info<sycl::info::platform::name>()
                  << std::endl;
        std::cout << "\tVendor: " << platform.get_info<sycl::info::platform::vendor>()
                  << std::endl;
        std::cout << "\tVersion: " << platform.get_info<sycl::info::platform::version>()
                  << std::endl;
        std::cout << "\tProfile: " << platform.get_info<sycl::info::platform::profile>()
                  << std::endl;
        int numDevice = 0;
        for (auto device : platform.get_devices())
        {
            std::cout << "\tDevice[" << numDevice++ << "]: " << device.get_info<sycl::info::device::name>()
                      << "\n\t\tType -> is_cpu: " << (device.is_cpu() ? "true" : "false")
                      << "\n\t\tType -> is_gpu: " << (device.is_gpu() ? "true" : "false")
                      << "\n\t\tType -> is_accelerator: " << (device.is_accelerator() ? "true" : "false")
                      << "\n\t\tVendor: " << device.get_info<sycl::info::device::vendor>()
                      << "\n\t\tDriver: " << device.get_info<sycl::info::device::driver_version>()
                      << "\n\t\tUsm_device_allocations: " << device.get_info<sycl::info::device::usm_device_allocations>()
                      << "\n\t\tUsm_host_allocations: " << device.get_info<sycl::info::device::usm_host_allocations>()
                      << "\n\t\tUsm_shared_allocations: " << device.get_info<sycl::info::device::usm_shared_allocations>()
                      << "\n\t\tUsm_restricted_shared_allocations: " << device.get_info<sycl::info::device::usm_restricted_shared_allocations>()
                      << "\n\t\tUsm_system_allocations: " << device.get_info<sycl::info::device::usm_system_allocations>()
                      << "\n\t\tMem_base_addr_align: " << device.get_info<sycl::info::device::mem_base_addr_align>()
                      << "\n\t\tGlobal_mem_size: " << device.get_info<sycl::info::device::global_mem_size>()
                      << "\n\t\tLocal_mem_size: " << device.get_info<sycl::info::device::local_mem_size>()
                      << "\n\t\tPartition_max_sub_devices: " << device.get_info<sycl::info::device::partition_max_sub_devices>()
                      << "\n\t\tMax_compute_units: " << device.get_info<sycl::info::device::max_compute_units>()
                      << "\n\t\tMax_work_item_dimensions: " << device.get_info<sycl::info::device::max_work_item_dimensions>()
                      << "\n\t\tMax_work_group/block_size: " << device.get_info<sycl::info::device::max_work_group_size>()
                      << "\n\t\tGlobal_mem_cache_line_size: " << device.get_info<sycl::info::device::global_mem_cache_line_size>()
                      << "\n\t\tGlobal_mem_cache_size: " << device.get_info<sycl::info::device::global_mem_cache_size>()
                      << "\n\t\tLocal_mem_type_enum {none(0), local(1), global(2)}: " << static_cast<int>(device.get_info<sycl::info::device::local_mem_type>())
                      << "\n\t\tSub_group_sizes: ";
            for (const auto &s : device.get_info<sycl::info::device::sub_group_sizes>())
                std::cout << s << " ";

            std::cout << "\n\t\tAtomic_memory_order_capabilities_enum {relaxed(0), acquire(1), "
                      << "\n\t\t\t__consume_unsupported(2), release(3), acq_rel(4), seq_cst(5)}: ";
            for (const auto &s : device.get_info<sycl::info::device::atomic_memory_order_capabilities>())
                std::cout << static_cast<int>(s) << " ";

            std::cout << "\n\t\tAtomic_memory_scope_capabilities_enum {work_item(0), sub_group(1), "
                      << "\n\t\t\twork_group(2), device(3), system(4)}: ";
            for (const auto &s : device.get_info<sycl::info::device::atomic_memory_scope_capabilities>())
                std::cout << static_cast<int>(s) << " ";

            std::cout << std::endl;
        }
    }
#endif

    // auto platforms = sycl::platform::get_platforms();
    // q_ct1 = sycl::queue(platforms[2].get_devices()[0]);

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
    // arrayC_d = sycl::malloc_host<fp>(M * N, q_ct1);
    // arrayC_d = sycl::malloc_shared<fp>(M * N, q_ct1);
    // arrayC_d = sycl::malloc_device<fp>(M * N, q_ct1.get_device(), q_ct1.get_context());
    // arrayC_d = sycl::aligned_alloc_device<fp>(32, M * N, q_ct1); // 4-, 8-, 16-, or 32-byte aligned
    // arrayC_d = sycl::malloc<fp>(M * N, q_ct1, sycl::usm::alloc::device); // device, shared, host
    // arrayC_d = sycl::aligned_alloc<fp>(32, M * N, q_ct1, sycl::usm::alloc::device); // device, shared, host
    // arrayC_d = static_cast<fp*>(sycl::malloc(M * N * sizeof(fp), q_ct1, sycl::usm::alloc::device)); // device, shared, host
    // arrayC_d = static_cast<fp*>(sycl::malloc_device(M * N * sizeof(fp), q_ct1));
    // fp *arrayC_d = alloc.allocate(M * N);

    for (int i = 0; i < M * K; i++)
        arrayA_h[i] = rand() / (fp)RAND_MAX * 1.0;

    for (int i = 0; i < K * N; i++)
        arrayB_h[i] = rand() / (fp)RAND_MAX * 1.0;

    memset(arrayC_h, 0, M * N * sizeof(fp));
    memset(arrayC_href, 0, M * N * sizeof(fp));

    q_ct1.memcpy(arrayA_d, arrayA_h, M * K * sizeof(fp));
    q_ct1.memcpy(arrayB_d, arrayB_h, K * N * sizeof(fp)).wait();
    q_ct1.memset(arrayC_d, 0, M * N * sizeof(fp)).wait();
    // q_ct1.fill<float>(arrayC_d, 0.0f, M * N).wait();
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
    // alloc.deallocate(arrayC_d, M * N);
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

void _matrixMul(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N, int order,
                sycl::nd_item<3> item_ct1)
{
    // absolute row and col
    unsigned int row;
    unsigned int col;
    if (order == 1)
    {
        row = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
              item_ct1.get_local_id(2);
        col = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
              item_ct1.get_local_id(1);
    }
    else if (order == 2)
    {
        row = item_ct1.get_local_range(0) * item_ct1.get_group(0) +
              item_ct1.get_local_id(0);
        col = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
              item_ct1.get_local_id(1);
    }
    else if (order == 3)
    {
        row = item_ct1.get_local_range(0) * item_ct1.get_group(0) +
              item_ct1.get_local_id(0);
        col = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
              item_ct1.get_local_id(2);
    }
    else if (order == 4)
    {
        row = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
              item_ct1.get_local_id(1);
        col = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
              item_ct1.get_local_id(2);
    }
    else if (order == 5)
    {
        row = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
              item_ct1.get_local_id(1);
        col = item_ct1.get_local_range(0) * item_ct1.get_group(0) +
              item_ct1.get_local_id(0);
    }
    else if (order == 6)
    {
        row = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
              item_ct1.get_local_id(2);
        col = item_ct1.get_local_range(0) * item_ct1.get_group(0) +
              item_ct1.get_local_id(0);
    }
    else if (order == 7)
    {
        int idx = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);
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

void warmupGpu(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N)
{
    result_reset();
    sycl::range<3> dimBlock(1, TILE_WIDTH, TILE_WIDTH);
    sycl::range<3> dimGrid(1, (N + TILE_WIDTH - 1) / TILE_WIDTH,
                           (M + TILE_WIDTH - 1) / TILE_WIDTH);

    __TIME_BEGIN
    *stop =
        q_ct1.parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                           [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(SUB_GRP_SZ)]]
                           {
                               _matrixMul(arrA, arrB, arrC, M, K, N, 1, item_ct1);
                           });
    stop->wait();
    __TIME_END

    // printf("0. warmupGpu time = %f ms\n", elapsedTime);
}

class matrixMulFunc
{
private:
    const fp *arrA;
    const fp *arrB;
    fp *arrC;
    int M, K, N;

public:
    matrixMulFunc(const fp *arr_A, const fp *arr_B, fp *arr_C, int m, int k, int n) : arrA(arr_A), arrB(arr_B), arrC(arr_C), M(m), K(k), N(n) {}
    void operator()(sycl::nd_item<3> item_ct1) const // the object of class matrixMulFunc will be const in parallel_for
    {
        // _matrixMul(arrA, arrB, arrC, M, K, N, 7, item_ct1);
        int idx = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);
        int row = idx / N;
        int col = idx % N;
        if (row < M && col < N)
        {
            for (int k = 0; k < K; k++)
            {
                arrC[row * N + col] += arrA[row * K + k] * arrB[k * N + col];
            }
        }
    }
};

void multiplyGpu(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N)
{
    warmupGpu(arrA, arrB, arrC, M, K, N);
    sycl::range<3> dimBlock(1, 1, 1);
    sycl::range<3> dimGrid(1, 1, 1);
    // order 1
    result_reset();
    dimBlock = sycl::range<3>(1, TILE_WIDTH, TILE_WIDTH);
    dimGrid = sycl::range<3>(1, (N + TILE_WIDTH - 1) / TILE_WIDTH,
                             (M + TILE_WIDTH - 1) / TILE_WIDTH);

    __TIME_BEGIN
    *stop = q_ct1.parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                               [=](sycl::nd_item<3> item_ct1)
                               {
                                   _matrixMul(arrA, arrB, arrC, M, K, N, 1,
                                              item_ct1);
                               });
    stop->wait();
    __TIME_END

    q_ct1.memcpy(arrayC_h, arrC, M * N * sizeof(fp)).wait();
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
    dimBlock = sycl::range<3>(TILE_WIDTH, TILE_WIDTH, 1);
    dimGrid = sycl::range<3>((M + TILE_WIDTH - 1) / TILE_WIDTH,
                             (N + TILE_WIDTH - 1) / TILE_WIDTH, 1);

    __TIME_BEGIN
    *stop = q_ct1.parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                               [=](sycl::nd_item<3> item_ct1)
                               {
                                   _matrixMul(arrA, arrB, arrC, M, K, N, 2,
                                              item_ct1);
                               });
    stop->wait();
    __TIME_END

    q_ct1.memcpy(arrayC_h, arrC, M * N * sizeof(fp)).wait();
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
    dimBlock = sycl::range<3>(TILE_WIDTH, 1, TILE_WIDTH);
    dimGrid = sycl::range<3>((M + TILE_WIDTH - 1) / TILE_WIDTH, 1,
                             (N + TILE_WIDTH - 1) / TILE_WIDTH);

    __TIME_BEGIN
    *stop = q_ct1.parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                               [=](sycl::nd_item<3> item_ct1)
                               {
                                   _matrixMul(arrA, arrB, arrC, M, K, N, 3,
                                              item_ct1);
                               });
    stop->wait();
    __TIME_END

    q_ct1.memcpy(arrayC_h, arrC, M * N * sizeof(fp)).wait();
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
    dimBlock = sycl::range<3>(1, TILE_WIDTH, TILE_WIDTH);
    dimGrid = sycl::range<3>(1, (M + TILE_WIDTH - 1) / TILE_WIDTH,
                             (N + TILE_WIDTH - 1) / TILE_WIDTH);

    __TIME_BEGIN
    *stop = q_ct1.parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                               [=](sycl::nd_item<3> item_ct1)
                               {
                                   _matrixMul(arrA, arrB, arrC, M, K, N, 4,
                                              item_ct1);
                               });
    stop->wait();
    __TIME_END

    q_ct1.memcpy(arrayC_h, arrC, M * N * sizeof(fp)).wait();
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
    dimBlock = sycl::range<3>(TILE_WIDTH, TILE_WIDTH, 1);
    dimGrid = sycl::range<3>((N + TILE_WIDTH - 1) / TILE_WIDTH,
                             (M + TILE_WIDTH - 1) / TILE_WIDTH, 1);

    __TIME_BEGIN
    *stop = q_ct1.parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                               [=](sycl::nd_item<3> item_ct1)
                               {
                                   _matrixMul(arrA, arrB, arrC, M, K, N, 5,
                                              item_ct1);
                               });
    stop->wait();
    __TIME_END

    q_ct1.memcpy(arrayC_h, arrC, M * N * sizeof(fp)).wait();
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
    dimBlock = sycl::range<3>(TILE_WIDTH, 1, TILE_WIDTH);
    dimGrid = sycl::range<3>((N + TILE_WIDTH - 1) / TILE_WIDTH, 1,
                             (M + TILE_WIDTH - 1) / TILE_WIDTH);

    __TIME_BEGIN
    *stop = q_ct1.parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                               [=](sycl::nd_item<3> item_ct1)
                               {
                                   _matrixMul(arrA, arrB, arrC, M, K, N, 6,
                                              item_ct1);
                               });
    stop->wait();
    __TIME_END

    q_ct1.memcpy(arrayC_h, arrC, M * N * sizeof(fp)).wait();
    if (!compare_matrix(arrayC_h, arrayC_href, M, N))
    {
        std::cout << "1. Error at multiplyGpu order 6" << std::endl;
    }
    else
    {
        printf("1. Pass, GPU calculation time (without shared memory order 6) = %f ms\n", elapsedTime);
    }

    // order 7
    result_reset();
    dimBlock = sycl::range<3>(1, 1, TILE_WIDTH * TILE_WIDTH);
    dimGrid = sycl::range<3>(1, 1, (N * M + dimBlock[2] - 1) / dimBlock[2]);

    __TIME_BEGIN
    *stop = q_ct1.parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                               [=](sycl::nd_item<3> item_ct1)
                               {
                                   _matrixMul(arrA, arrB, arrC, M, K, N, 7,
                                              item_ct1);
                               });
    stop->wait();
    __TIME_END

    q_ct1.memcpy(arrayC_h, arrC, M * N * sizeof(fp)).wait();
    if (!compare_matrix(arrayC_h, arrayC_href, M, N))
    {
        std::cout << "1. Error at multiplyGpu order 7" << std::endl;
    }
    else
    {
        printf("1. Pass, GPU calculation time (without shared memory order 7) = %f ms\n", elapsedTime);
    }

    // use functor, order 7
    result_reset();
    dimBlock = sycl::range<3>(1, 1, TILE_WIDTH * TILE_WIDTH);
    dimGrid = sycl::range<3>(1, 1, (N * M + dimBlock[2] - 1) / dimBlock[2]);

    __TIME_BEGIN
    *stop = q_ct1.parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                               matrixMulFunc(arrA, arrB, arrC, M, K, N));
    stop->wait();
    __TIME_END

    q_ct1.memcpy(arrayC_h, arrC, M * N * sizeof(fp)).wait();
    if (!compare_matrix(arrayC_h, arrayC_href, M, N))
    {
        std::cout << "1. Error at multiplyGpu order 7 (functor)" << std::endl;
    }
    else
    {
        printf("1. Pass, GPU calculation time (without shared memory order 7, functor) = %f ms\n", elapsedTime);
    }

    // // use functor, order 8, single_task
    // result_reset();
    // dimBlock = sycl::range<3>(1, 1, TILE_WIDTH * TILE_WIDTH);
    // dimGrid = sycl::range<3>(1, 1, (N * M + dimBlock[2] - 1) / dimBlock[2]);

    // __TIME_BEGIN
    // *stop = q_ct1.single_task([=]()
    //                           {
    //     for (int i = 0; i < M; i++)
    //     {
    //         for (int j = 0; j < N; j++)
    //         {
    //             for (int k = 0; k < K; k++)
    //             {
    //                 arrC[i * N + j] += arrA[i * K + k] * arrB[k * N + j];
    //             }
    //         }
    //     } });
    // stop->wait();
    // __TIME_END

    // q_ct1.memcpy(arrayC_h, arrC, M * N * sizeof(fp)).wait();
    // if (!compare_matrix(arrayC_h, arrayC_href, M, N))
    // {
    //     std::cout << "1. Error at multiplyGpu order 8 (single_task)" << std::endl;
    // }
    // else
    // {
    //     printf("1. Pass, GPU calculation time (without shared memory order 8, single_task) = %f ms\n", elapsedTime);
    // }
}

void _setVecVal(fp4 &a, fp val)
{
    a.x() = val;
    a.y() = val;
    a.z() = val;
    a.w() = val;
}

void _matrixVecLoadH(const fp *src, fp4 *dst, int M, int K4, int K,
                     sycl::nd_item<3> item_ct1)
{
    // absolute row and col
    unsigned int idx = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                       item_ct1.get_local_id(2);
    unsigned int row = idx / K4;
    unsigned int col = idx % K4;
    if (row >= M || col >= K4)
        return;
    unsigned int offset = row * K4 + col;
    unsigned int offsetSrc = row * K + col * 4;
    _setVecVal(dst[offset], 0.0f);

    if (col * 4 + 3 < K)
    {
        dst[offset].x() = src[offsetSrc];
        dst[offset].y() = src[offsetSrc + 1];
        dst[offset].z() = src[offsetSrc + 2];
        dst[offset].w() = src[offsetSrc + 3];
    }
    else if (col * 4 + 2 < K)
    {
        dst[offset].x() = src[offsetSrc];
        dst[offset].y() = src[offsetSrc + 1];
        dst[offset].z() = src[offsetSrc + 2];
    }
    else if (col * 4 + 1 < K)
    {
        dst[offset].x() = src[offsetSrc];
        dst[offset].y() = src[offsetSrc + 1];
    }
    else
    {
        dst[offset].x() = src[offsetSrc];
    }
}

void _matrixVecLoadV(const fp *src, fp4 *dst, int K4, int N, int K,
                     sycl::nd_item<3> item_ct1)
{
    // absolute row and col
    unsigned int idx = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                       item_ct1.get_local_id(2);
    unsigned int row = idx / N;
    unsigned int col = idx % N;
    if (row >= K4 || col >= N)
        return;
    unsigned int offset = row * N + col;
    unsigned int offsetSrc = row * 4 * N + col;
    _setVecVal(dst[offset], 0.0f);

    if (row * 4 + 3 < K)
    {
        dst[offset].x() = src[offsetSrc];
        dst[offset].y() = src[offsetSrc + N];
        dst[offset].z() = src[offsetSrc + 2 * N];
        dst[offset].w() = src[offsetSrc + 3 * N];
    }
    else if (row * 4 + 2 < K)
    {
        dst[offset].x() = src[offsetSrc];
        dst[offset].y() = src[offsetSrc + N];
        dst[offset].z() = src[offsetSrc + 2 * N];
    }
    else if (row * 4 + 1 < K)
    {
        dst[offset].x() = src[offsetSrc];
        dst[offset].y() = src[offsetSrc + N];
    }
    else
    {
        dst[offset].x() = src[offsetSrc];
    }
}

fp dot_prod(const fp4 &a, const fp4 &b)
{
    return a.x() * b.x() + a.y() * b.y() + a.z() * b.z() + a.w() * b.w();
}

void _matrixMulVec(const fp4 *arrA, const fp4 *arrB, fp *arrC, int M, int K4, int N,
                   sycl::nd_item<3> item_ct1)
{
    // absolute row and col
    int idx = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
              item_ct1.get_local_id(2);
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

void _matrixMulVecSh(const fp4 *arrA, const fp4 *arrB, fp *arrC, int M, int K4, int N,
                     sycl::nd_item<3> item_ct1, fp4 *arrAs, fp4 *arrBs)
{
    // absolute row and col
    unsigned int row = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                       item_ct1.get_local_id(2);
    unsigned int col = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                       item_ct1.get_local_id(1);

    fp elementC = 0.0f;
    for (int i = 0; i < K4 / TILE_WIDTH; i++)
    {
        if (i * TILE_WIDTH + item_ct1.get_local_id(1) < K4)
            arrAs[item_ct1.get_local_id(1) * TILE_WIDTH +
                  item_ct1.get_local_id(2)] =
                arrA[row * K4 + i * TILE_WIDTH + item_ct1.get_local_id(1)];
        else
            _setVecVal(arrAs[item_ct1.get_local_id(1) * TILE_WIDTH +
                             item_ct1.get_local_id(2)],
                       0.0f);
        if (i * TILE_WIDTH + item_ct1.get_local_id(2) < K4)
            arrBs[item_ct1.get_local_id(1) * TILE_WIDTH +
                  item_ct1.get_local_id(2)] =
                arrB[(i * TILE_WIDTH + item_ct1.get_local_id(2)) * N + col];
        else
            _setVecVal(arrBs[item_ct1.get_local_id(1) * TILE_WIDTH +
                             item_ct1.get_local_id(2)],
                       0.0f);
        // item_ct1.barrier();
        item_ct1.barrier(sycl::access::fence_space::local_space);
        for (int k = 0; k < TILE_WIDTH; k++)
            elementC +=
                dot_prod(arrAs[k * TILE_WIDTH + item_ct1.get_local_id(2)],
                         arrBs[item_ct1.get_local_id(1) * TILE_WIDTH + k]);
        // item_ct1.barrier();
        item_ct1.barrier(sycl::access::fence_space::local_space);
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
    arrayA_d4 = sycl::malloc_device<fp4>(M * K4, q_ct1);
    arrayB_d4 = sycl::malloc_device<fp4>(K4 * N, q_ct1);

    sycl::range<3> dimBlock(1, 1, 1);
    sycl::range<3> dimGrid(1, 1, 1);

    __TIME_BEGIN
    // load matrix values into vector type
    dimBlock = sycl::range<3>(1, 1, TILE_WIDTH * TILE_WIDTH);
    dimGrid = sycl::range<3>(1, 1, (M * K4 + dimBlock[2] - 1) / dimBlock[2]);
    *stop = q_ct1.parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                               [=](sycl::nd_item<3> item_ct1)
                               {
                                   _matrixVecLoadH(arrA, arrayA_d4, M, K4, K,
                                                   item_ct1);
                               });
    stop->wait();

    dimBlock = sycl::range<3>(1, 1, TILE_WIDTH * TILE_WIDTH);
    dimGrid = sycl::range<3>(1, 1, (K4 * N + dimBlock[2] - 1) / dimBlock[2]);
    *stop = q_ct1.parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                               [=](sycl::nd_item<3> item_ct1)
                               {
                                   _matrixVecLoadV(arrB, arrayB_d4, K4, N, K,
                                                   item_ct1);
                               });
    stop->wait();

#ifdef USE_SHARED_MEMORY_VEC
    dimBlock = sycl::range<3>(1, TILE_WIDTH, TILE_WIDTH);
    dimGrid = sycl::range<3>(1, (N + TILE_WIDTH - 1) / TILE_WIDTH,
                             (M + TILE_WIDTH - 1) / TILE_WIDTH);
    *stop = q_ct1.submit([&](sycl::handler &cgh)
                         {
        sycl::local_accessor<fp4, 1> arrAs_acc_ct1(
            sycl::range<1>(TILE_WIDTH * TILE_WIDTH), cgh);
        sycl::local_accessor<fp4, 1> arrBs_acc_ct1(
            sycl::range<1>(TILE_WIDTH * TILE_WIDTH), cgh);

        cgh.parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                         [=](sycl::nd_item<3> item_ct1) {
                             _matrixMulVecSh(arrayA_d4, arrayB_d4, arrC, M, K4,
                                             N, item_ct1,
                                             arrAs_acc_ct1.get_pointer(),
                                             arrBs_acc_ct1.get_pointer());
                         }); });
#else
    dimBlock = sycl::range<3>(1, 1, TILE_WIDTH * TILE_WIDTH);
    dimGrid = sycl::range<3>(1, 1, (M * N + dimBlock[2] - 1) / dimBlock[2]);
    *stop = q_ct1.parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                               [=](sycl::nd_item<3> item_ct1)
                               {
                                   _matrixMulVec(arrayA_d4, arrayB_d4, arrC, M,
                                                 K4, N, item_ct1);
                               });
#endif
    stop->wait();
    __TIME_END

    q_ct1.memcpy(arrayC_h, arrC, M * N * sizeof(fp)).wait();
    if (!compare_matrix(arrayC_h, arrayC_href, M, N))
    {
        std::cout << "1. Error at multiplyGpuVec" << std::endl;
    }
    else
    {
        printf("1. Pass, GPU calculation time (with float4 type) = %f ms\n", elapsedTime);
    }
    sycl::free(arrayA_d4, q_ct1);
    sycl::free(arrayB_d4, q_ct1);
}

void _matrixMulAccessor(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N, sycl::nd_item<3> item_ct1)
{
    // absolute row and col
    int idx = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
              item_ct1.get_local_id(2);
    unsigned int row = idx / N;
    unsigned int col = idx % N;

    if (row < M && col < N)
    {
        fp sum = 0;
        for (int k = 0; k < K; k++)
        {
            sum += arrA[row * K + k] * arrB[k * N + col];
        }
        arrC[row * N + col] = sum;
    }
}

void multiplyGpuAccessor(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N)
{
    result_reset();
    sycl::range<3> dimBlock(1, 1, TILE_WIDTH * TILE_WIDTH);
    sycl::range<3> dimGrid(1, 1, (N * M + dimBlock[2] - 1) / dimBlock[2]);
    // {
    sycl::buffer<fp> arrAbuf{arrA, sycl::range{rangeMK}};
    sycl::buffer<fp> arrBbuf{arrB, sycl::range{rangeKN}};
    sycl::buffer<fp> arrCbuf{arrC, sycl::range{rangeMN}};
    // sycl::accessor arrAacc{arrAbuf, sycl::read_only};
    // sycl::accessor arrBacc{arrBbuf, sycl::read_only};
    // sycl::accessor arrCacc{arrCbuf, sycl::write_only};
    __TIME_BEGIN
    *stop = q_ct1.submit([&](sycl::handler &cgh)
                         {
                            // cgh.require(arrAacc);
                            // cgh.require(arrBacc);
                            // cgh.require(arrCacc);
        sycl::accessor arrAacc{arrAbuf, cgh, sycl::read_only};
        sycl::accessor arrBacc{arrBbuf, cgh, sycl::read_only};
        sycl::accessor arrCacc{arrCbuf, cgh, sycl::write_only, sycl::no_init};
        cgh.parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                         [=](sycl::nd_item<3> item_ct1)
                         {
                             _matrixMulAccessor(arrAacc.get_pointer(), arrBacc.get_pointer(), arrCacc.get_pointer(), M, K, N,
                                                item_ct1);
                         }); });
    stop->wait();
    __TIME_END
    // }
    sycl::host_accessor h_arrCacc{arrCbuf};
    if (!compare_matrix(arrayC_h, arrayC_href, M, N))
    {
        std::cout << "1. Error at multiplyGpuAccessor" << std::endl;
    }
    else
    {
        printf("1. Pass, GPU calculation time (without shared memory, with buffer + accessor) = %f ms\n", elapsedTime);
    }
}

#define GRP_METHOD1

void _matrixMulGrp(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N,
                   sycl::group<3> grp, sycl::h_item<3> item_ct1)
{
    // absolute row and col
#ifdef GRP_METHOD1
    // method 1
    unsigned int row = grp.get_group_id(0) * grp.get_local_range(0) +
                       item_ct1.get_logical_local_id(0);
    unsigned int col = grp.get_group_id(1) * grp.get_local_range(1) +
                       item_ct1.get_logical_local_id(1);
#else
    // method 2
    unsigned int row = grp.get_group_id(0) * item_ct1.get_local_range(0) +
                       item_ct1.get_logical_local_id(0);
    unsigned int col = grp.get_group_id(1) * item_ct1.get_local_range(1) +
                       item_ct1.get_logical_local_id(1);
#endif

    if (row < M && col < N)
    {
        for (int k = 0; k < K; k++)
        {
            arrC[row * N + col] += arrA[row * K + k] * arrB[k * N + col];
        }
    }
}

void multiplyGpuGrp(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N)
{
    result_reset();
    sycl::range<3> dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    sycl::range<3> dimGrid((M + TILE_WIDTH - 1) / TILE_WIDTH,
                           (N + TILE_WIDTH - 1) / TILE_WIDTH, 1);

    __TIME_BEGIN
#ifdef GRP_METHOD1
    // method 1
    *stop = q_ct1.submit([&](sycl::handler &cgh)
                         { cgh.parallel_for_work_group(dimGrid, dimBlock, [=](sycl::group<3> grp)
                                                       { grp.parallel_for_work_item([&](sycl::h_item<3> item_ct1)
                                                                                    { _matrixMulGrp(arrA, arrB, arrC, M, K, N, grp, item_ct1); }); }); });
#else
    // method 2
    *stop = q_ct1.submit([&](sycl::handler &cgh)
                         { cgh.parallel_for_work_group(dimGrid, dimBlock, [=](sycl::group<3> grp)
                                                       { grp.parallel_for_work_item(dimBlock, [&](sycl::h_item<3> item_ct1)
                                                                                    { _matrixMulGrp(arrA, arrB, arrC, M, K, N, grp, item_ct1); }); }); });
    // below case is very slow without the optional parameter dimBlock in parallel_for_work_group, though the result is correct
    // *stop = q_ct1.submit([&](sycl::handler &cgh)
    //                      { cgh.parallel_for_work_group(dimGrid, [=](sycl::group<3> grp)
    //                                                    { grp.parallel_for_work_item(dimBlock, [&](sycl::h_item<3> item_ct1)
    //                                                                                 { _matrixMulGrp(arrA, arrB, arrC, M, K, N, grp, item_ct1); }); }); });
#endif

    stop->wait();
    __TIME_END

    q_ct1.memcpy(arrayC_h, arrC, M * N * sizeof(fp)).wait();
    if (!compare_matrix(arrayC_h, arrayC_href, M, N))
    {
        std::cout << "1. Error at multiplyGpuGrp" << std::endl;
    }
    else
    {
        printf("1. Pass, GPU calculation time (without shared memory, with Grp) = %f ms\n", elapsedTime);
    }
}

void multiplyGpuGrpSh(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N)
{
    result_reset();
    sycl::range<3> dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    sycl::range<3> dimGrid((M + TILE_WIDTH - 1) / TILE_WIDTH,
                           (N + TILE_WIDTH - 1) / TILE_WIDTH, 1);

    __TIME_BEGIN

    // method 1
    *stop = q_ct1.submit([&](sycl::handler &cgh)
                         { 
        // sycl::local_accessor<fp, 1> arrAs_acc_ct1(
        //     sycl::range<1>(TILE_WIDTH * TILE_WIDTH), cgh);
        // sycl::local_accessor<fp, 1> arrBs_acc_ct1(
        //     sycl::range<1>(TILE_WIDTH * TILE_WIDTH), cgh);

        cgh.parallel_for_work_group(dimGrid, dimBlock, [=](sycl::group<3> grp)
        {    
            // fp *arrAs = arrAs_acc_ct1.get_pointer();
            // fp *arrBs = arrBs_acc_ct1.get_pointer();
            fp arrAs[TILE_WIDTH * TILE_WIDTH];
            fp arrBs[TILE_WIDTH * TILE_WIDTH];

            for (int i = 0; i < (K + TILE_WIDTH - 1) / TILE_WIDTH; i++)
            {
                // implicit barrier
                grp.parallel_for_work_item([&](sycl::h_item<3> item_ct1) 
                {                 
                    unsigned int row = grp.get_group_id(0) * grp.get_local_range(0) +
                                    item_ct1.get_logical_local_id(0);
                    unsigned int col = grp.get_group_id(1) * grp.get_local_range(1) +
                                    item_ct1.get_logical_local_id(1);
                    if (i * TILE_WIDTH + item_ct1.get_logical_local_id(1) < K)
                        arrAs[item_ct1.get_logical_local_id(1) * TILE_WIDTH +
                            item_ct1.get_logical_local_id(0)] =
                            arrA[row * K + i * TILE_WIDTH + item_ct1.get_logical_local_id(1)];
                    else
                        arrAs[item_ct1.get_logical_local_id(1) * TILE_WIDTH +
                            item_ct1.get_logical_local_id(0)] = 0;
                    if (i * TILE_WIDTH + item_ct1.get_logical_local_id(0) < K)
                        arrBs[item_ct1.get_logical_local_id(1) * TILE_WIDTH +
                            item_ct1.get_logical_local_id(0)] =
                            arrB[(i * TILE_WIDTH + item_ct1.get_logical_local_id(0)) * N + col];
                    else
                        arrBs[item_ct1.get_logical_local_id(1) * TILE_WIDTH +
                            item_ct1.get_logical_local_id(0)] = 0; 
                });
                // implicit barrier

                grp.parallel_for_work_item([&](sycl::h_item<3> item_ct1) 
                { 
                    unsigned int row = grp.get_group_id(0) * grp.get_local_range(0) +
                                    item_ct1.get_logical_local_id(0);
                    unsigned int col = grp.get_group_id(1) * grp.get_local_range(1) +
                                    item_ct1.get_logical_local_id(1);
                    if (row < M && col < N)
                    {
                        for (int k = 0; k < TILE_WIDTH; k++)
                            arrC[row * N + col] += arrAs[k * TILE_WIDTH + item_ct1.get_logical_local_id(0)] *
                                                arrBs[item_ct1.get_logical_local_id(1) * TILE_WIDTH + k];
                    }
                });
            }                
        }); });

    stop->wait();
    __TIME_END

    q_ct1.memcpy(arrayC_h, arrC, M * N * sizeof(fp)).wait();
    if (!compare_matrix(arrayC_h, arrayC_href, M, N))
    {
        std::cout << "1. Error at multiplyGpuGrpSh" << std::endl;
    }
    else
    {
        printf("1. Pass, GPU calculation time (with shared memory, with GrpSh) = %f ms\n", elapsedTime);
    }
}

void _matrixMulBcast(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N,
                     sycl::nd_item<2> item_ct1)
{
    // absolute row and col
    unsigned int row = item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                       item_ct1.get_local_id(0);
    unsigned int col = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                       item_ct1.get_local_id(1);

    fp elementC = 0;
    auto sg = item_ct1.get_sub_group();
    for (int i = 0; i < K; i += TILE_WIDTH)
    {
        fp tileA = arrA[row * K + i + item_ct1.get_local_id(1)];
        for (int k = 0; k < TILE_WIDTH; k++)
            elementC += sycl::group_broadcast(sg, tileA, k) * arrB[(i + k) * N + col];
        // elementC += arrA[row * K + i + k] * arrB[(i + k) * N + col]; // the speed is the same as above one in CUDA_BACKEND, perhaps the broadcast mechanism is automatically triggered.
    }
    if (row < M && col < N)
        arrC[row * N + col] = elementC;
}

void multiplyGpuBcast(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N)
{
    result_reset();
    sycl::range<2> dimBlock(1, TILE_WIDTH);
    sycl::range<2> dimGrid(M, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    __TIME_BEGIN
    *stop = q_ct1.submit([&](sycl::handler &cgh)
                         { cgh.parallel_for(sycl::nd_range<2>(dimGrid * dimBlock, dimBlock),
                                            [=](sycl::nd_item<2> item_ct1) [[intel::reqd_sub_group_size(SUB_GRP_SZ)]]
                                            // SUB_GRP_SZ >= TILE_WIDTH, so as to cover the whole stencil
                                            {
                                                _matrixMulBcast(arrA, arrB, arrC, M, K, N, item_ct1);
                                            }); });
    stop->wait();
    __TIME_END

    q_ct1.memcpy(arrayC_h, arrC, M * N * sizeof(fp)).wait();
    if (!compare_matrix(arrayC_h, arrayC_href, M, N))
    {
        std::cout << "1. Error at multiplyGpuBcast" << std::endl;
    }
    else
    {
        printf("1. Pass, GPU calculation time (without shared memory, with bcast) = %f ms\n", elapsedTime);
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
        // item_ct1.barrier();
        for (int k = 0; k < TILE_WIDTH; k++)
            elementC += arrAs[k * TILE_WIDTH + item_ct1.get_local_id(2)] *
                        arrBs[item_ct1.get_local_id(1) * TILE_WIDTH + k];
        item_ct1.barrier(sycl::access::fence_space::local_space);
        // item_ct1.barrier();
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
    *stop = q_ct1.submit([&](sycl::handler &cgh)
                         {
        sycl::local_accessor<fp, 1> arrAs_acc_ct1(
            sycl::range<1>(TILE_WIDTH * TILE_WIDTH), cgh);
        // sycl::accessor<fp, 1, sycl::access::mode::read_write, sycl::access::target::local> arrAs_acc_ct1(
            // sycl::range<1>(TILE_WIDTH * TILE_WIDTH), cgh);
        sycl::local_accessor<fp, 1> arrBs_acc_ct1(
            sycl::range<1>(TILE_WIDTH * TILE_WIDTH), cgh);

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

void _matrixMulSh2(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N,
                   sycl::nd_item<3> item_ct1, uint8_t *dpct_local)
{
    // absolute row and col
    unsigned int row = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                       item_ct1.get_local_id(2);
    unsigned int col = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                       item_ct1.get_local_id(1);
    auto arrAs = (fp *)dpct_local;
    fp *arrBs =
        arrAs + item_ct1.get_local_range(2) * item_ct1.get_local_range(1);
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

void multiplyGpuSh2(const fp *arrA, const fp *arrB, fp *arrC, int M, int K, int N)
{
    result_reset();
    sycl::range<3> dimBlock(1, TILE_WIDTH, TILE_WIDTH);
    sycl::range<3> dimGrid(1, (N + TILE_WIDTH - 1) / TILE_WIDTH,
                           (M + TILE_WIDTH - 1) / TILE_WIDTH);

    __TIME_BEGIN
    *stop = q_ct1.submit([&](sycl::handler &cgh)
                         {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(TILE_WIDTH * TILE_WIDTH * sizeof(fp) * 2), cgh);

        cgh.parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
                         [=](sycl::nd_item<3> item_ct1) {
                             _matrixMulSh2(arrA, arrB, arrC, M, K, N, item_ct1,
                                           dpct_local_acc_ct1.get_pointer());
                         }); });
    stop->wait();
    __TIME_END

    q_ct1.memcpy(arrayC_h, arrC, M * N * sizeof(fp)).wait();
    if (!compare_matrix(arrayC_h, arrayC_href, M, N))
    {
        std::cout << "2.1 Error at multiplyGpuSh2" << std::endl;
    }
    else
    {
        printf("2.1 Pass, GPU calculation time (with shared memory) = %f ms\n", elapsedTime);
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
    *stop = q_ct1.submit([&](sycl::handler &cgh)
                         {
        sycl::local_accessor<fp, 1> arrAs_acc_ct1(
            sycl::range<1>(TILE_WIDTH * TILE_WIDTH), cgh);
        sycl::local_accessor<fp, 1> arrBs_acc_ct1(
            sycl::range<1>(TILE_WIDTH * TILE_WIDTH), cgh);

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
    *stop = q_ct1.submit([&](sycl::handler &cgh)
                         {
        sycl::local_accessor<fp, 1> arrAs_acc_ct1(
            sycl::range<1>(TILE_WIDTH * (TILE_WIDTH + 1)), cgh);
        sycl::local_accessor<fp, 1> arrBs_acc_ct1(
            sycl::range<1>(TILE_WIDTH * (TILE_WIDTH + 1)), cgh);

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
