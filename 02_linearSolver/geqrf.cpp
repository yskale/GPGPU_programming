// icpx -fsycl -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core geqrf.cpp
// solve Ax=b, with QR factorization
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <oneapi/mkl.hpp>
#include <dpct/blas_utils.hpp>

#include <chrono>

#define __TIME_BEGIN                              \
    start_ct1 = std::chrono::steady_clock::now(); \
    *start = q_ct1.ext_oneapi_submit_barrier();

#define __TIME_END                               \
    stop_ct1 = std::chrono::steady_clock::now(); \
    *stop = q_ct1.ext_oneapi_submit_barrier();   \
    stop->wait_and_throw();                      \
    elapsedTime =                                \
        std::chrono::duration<float, std::milli>(stop_ct1 - start_ct1).count();

// #define SHOW_MATRIX
// #define DOUBLE_FP_CASE
#ifdef DOUBLE_FP_CASE
typedef double fp;
#else
typedef float fp;
#endif

const fp sparselevel = 0.3;
const int N = 1000;
constexpr int lda = N;
constexpr int ldb = N;
constexpr int matSize = N * N;
const fp alpha = 1;
int lwork = 0;        /* size of workspace */
fp *work_d = nullptr; /* device workspace for geqrf */

fp *matA_h, *vecb_h, *resx_h;
fp *matA_d, *vecb_d, *tau_d;

dpct::event_ptr start, stop;
std::chrono::time_point<std::chrono::steady_clock> start_ct1;
std::chrono::time_point<std::chrono::steady_clock> stop_ct1;
float elapsedTime;

dpct::device_ext &dev_ct1 = dpct::get_current_device();
sycl::queue &q_ct1 = dev_ct1.default_queue();

void print_matrix(const fp *arr, int M, int N)
{
    for (int i = 0; i < M; i++)
    {
        // std::cout << "row " << i << ": ";
        std::cout << std::fixed;
        for (int j = 0; j < N; j++)
        {
            std::cout << arr[j * M + i] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

void resources_init()
{
    std::cout << "Selected device: " << q_ct1.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << "Device vendor: " << q_ct1.get_device().get_info<sycl::info::device::vendor>() << "\n";
    std::cout << "Max group/block size = " << q_ct1.get_device().get_info<sycl::info::device::max_work_group_size>() << "\n";
    std::cout << "Max Compute Units = " << q_ct1.get_device().get_info<sycl::info::device::max_compute_units>() << "\n";
    std::cout << "Shared Local Memory size = " << q_ct1.get_device().get_info<sycl::info::device::local_mem_size>() << " Bytes\n";
    std::cout << "Sub-group Sizes: ";
    for (const auto &s : q_ct1.get_device().get_info<sycl::info::device::sub_group_sizes>())
        std::cout << s << " ";
    std::cout << std::endl;

    matA_h = new fp[matSize]();
    vecb_h = new fp[N]();
    resx_h = new fp[N]();

    matA_d = sycl::malloc_device<fp>(matSize, q_ct1);
    vecb_d = sycl::malloc_device<fp>(N, q_ct1);
    tau_d = sycl::malloc_device<fp>(N, q_ct1);

    for (int i = 0; i < matSize; i++)
    {
        matA_h[i] = rand() / (fp)RAND_MAX * 1.0;
        if (rand() / (fp)RAND_MAX * 1.0 < sparselevel) // make the matrix become sparse
            matA_h[i] = 0.0;
    }

    for (int i = 0; i < N; i++)
        vecb_h[i] = rand() / (fp)RAND_MAX * 1.0;

    memset(resx_h, 0, N * sizeof(fp));

    q_ct1.memcpy(matA_d, matA_h, matSize * sizeof(fp));
    q_ct1.memcpy(vecb_d, vecb_h, N * sizeof(fp)).wait();

    int lwork_geqrf = 0;
    int lwork_ormqr = 0;

    lwork_geqrf = oneapi::mkl::lapack::geqrf_scratchpad_size<fp>(
        q_ct1, N, N, lda);
    lwork_ormqr = oneapi::mkl::lapack::ormqr_scratchpad_size<fp>(
        q_ct1, oneapi::mkl::side::left, oneapi::mkl::transpose::trans, N,
        1, N, lda, ldb);

    lwork = std::max(lwork_geqrf, lwork_ormqr);
    work_d = sycl::malloc_device<fp>(lwork, q_ct1);

    start = new sycl::event();
    stop = new sycl::event();

#ifdef SHOW_MATRIX
    std::cout << "A = \n";
    print_matrix(matA_h, N, N);
    std::cout << "b = \n";
    print_matrix(vecb_h, N, 1);
#endif
}

void result_reset()
{
    memset(resx_h, 0, N * sizeof(fp));
    q_ct1.memcpy(matA_d, matA_h, matSize * sizeof(fp));
    q_ct1.memcpy(vecb_d, vecb_h, N * sizeof(fp)).wait();
}

void resources_free()
{
    delete[] matA_h;
    delete[] vecb_h;
    delete[] resx_h;

    sycl::free(matA_d, q_ct1);
    sycl::free(vecb_d, q_ct1);
    sycl::free(work_d, q_ct1);
    sycl::free(tau_d, q_ct1);

    dpct::destroy_event(start);
    dpct::destroy_event(stop);

    dev_ct1.reset();
}

void check_result(fp *matA, fp *resX, fp *vecb, int n)
{
    fp errorNorm = 0.0;
    for (int i = 0; i < n; i++)
    {
        fp sum = 0.0;
        for (int j = 0; j < n; j++)
        {
            sum += matA[j * n + i] * resX[j];
        }
        errorNorm += std::pow<fp, int>((sum - vecb[i]), 2);
    }
    errorNorm = std::pow<fp, fp>(errorNorm, 0.5f);
    std::cout << "error 2-norm = " << errorNorm << std::endl;
}

int main()
{
    resources_init();
    for (int i = 0; i < 10; i++)
    {
        result_reset();
        __TIME_BEGIN
        // compute QR factorization
        oneapi::mkl::lapack::geqrf(q_ct1, N, N, matA_d, lda,
                                   tau_d, work_d, lwork);
        // compute Q^T*B
        oneapi::mkl::lapack::ormqr(
            q_ct1, oneapi::mkl::side::left, oneapi::mkl::transpose::trans,
            N, 1, N, matA_d, lda, tau_d, vecb_d, ldb,
            work_d, lwork);
        // compute x = R \ Q^T*B
        oneapi::mkl::blas::column_major::trsm(
            q_ct1, oneapi::mkl::side::left, oneapi::mkl::uplo::upper,
            oneapi::mkl::transpose::nontrans, oneapi::mkl::diag::nonunit, N, 1,
            alpha, matA_d, lda, vecb_d, ldb);
        q_ct1.wait();
        __TIME_END
        std::cout << "No. " << i << " run, GPU calculation time = " << elapsedTime << "ms\n";
    }

    q_ct1.memcpy(resx_h, vecb_d, sizeof(fp) * N).wait();

#ifdef SHOW_MATRIX
    std::cout << "x = \n";
    print_matrix(resx_h, N, 1);
#endif

    check_result(matA_h, resx_h, vecb_h, N);

    resources_free();
    return 0;
}