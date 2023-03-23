// icpx -fsycl -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core syevd.cpp
// solve Ax=\lambda x, using syevd to compute the spectrum of a dense symmetric (Hermitian) system

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <oneapi/mkl.hpp>
#include <dpct/blas_utils.hpp>

#include <limits>
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
constexpr int matSize = N * N;

int lwork = 0;         /* size of workspace */
fp *work_d = nullptr;  /* device workspace for potrf */

fp *matA_h, *eigenVec_h, *eigenVal_h;
fp *matA_d, *eigenVal_d;

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
        std::cout << "row " << i << ": ";
        std::cout << std::fixed;
        for (int j = 0; j < N; j++)
        {
            std::cout << arr[j * N + i] << " ";
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
    eigenVec_h = new fp[matSize]();
    eigenVal_h = new fp[N]();
    // matA_h = new fp[matSize]{3.5, 0.5, 0.0, 0.5, 3.5, 0.0, 0.0, 0.0, 2.0};
    // eigenVec_h = new fp[matSize]{0.00, 0.00, 1.00, 0.71, -0.71, 0.00, 0.71, 0.71, 0.00};
    // eigenVal_h = new fp[N]{2.0, 3.0, 4.0};

    matA_d = sycl::malloc_device<fp>(matSize, q_ct1);
    eigenVal_d = sycl::malloc_device<fp>(N, q_ct1);

    // ======================================================================================
    // create a tmp random matrix
    fp *matTmp_h = new fp[matSize]();
    for (int i = 0; i < matSize; i++)
    {
        matTmp_h[i] = rand() / (fp)RAND_MAX * 1.0;
        if (rand() / (fp)RAND_MAX * 1.0 < sparselevel) // make the matrix become sparse
            matTmp_h[i] = 0.0;
    }

    // create positive definite Hermitian (symmetry) matrix, https://cplusplus.com/forum/general/257711/
    fp maxVal = std::numeric_limits<fp>::min();
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            matA_h[j * N + i] = 0;
            for (int k = 0; k < N; k++)
            {
                matA_h[j * N + i] += matTmp_h[k * N + i] * matTmp_h[k * N + j];
            }
            maxVal = std::max(maxVal, matA_h[j * N + i]);
        }
    }
    // normalization
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            matA_h[j * N + i] /= maxVal;
        }
    }
    delete[] matTmp_h;
    // ======================================================================================

    memset(eigenVec_h, 0, matSize * sizeof(fp));
    memset(eigenVal_h, 0, N * sizeof(fp));

    q_ct1.memcpy(matA_d, matA_h, matSize * sizeof(fp));
    q_ct1.memcpy(eigenVal_d, eigenVal_h, N * sizeof(fp)).wait();

    // DPCT1007:0: Migration of cusolverDnDsyevd_bufferSize is not supported.
    lwork = oneapi::mkl::lapack::syevd_scratchpad_size<fp>(q_ct1, oneapi::mkl::job::vec, oneapi::mkl::uplo::lower, N, lda);
    work_d = sycl::malloc_device<fp>(lwork, q_ct1);

    start = new sycl::event();
    stop = new sycl::event();

#ifdef SHOW_MATRIX
    std::cout << "A = \n";
    print_matrix(matA_h, N, N);
#endif
}

void result_reset()
{
    memset(eigenVal_h, 0, N * sizeof(fp));
    q_ct1.memcpy(matA_d, matA_h, matSize * sizeof(fp)).wait();
}

void resources_free()
{
    delete[] matA_h;
    delete[] eigenVec_h;
    delete[] eigenVal_h;

    sycl::free(matA_d, q_ct1);
    sycl::free(eigenVal_d, q_ct1);
    sycl::free(work_d, q_ct1);

    dpct::destroy_event(start);
    dpct::destroy_event(stop);

    dev_ct1.reset();
}

void check_result(fp *matA, fp *resX, fp *vecs, int n)
{
    fp errorNorm = 0.0;
    for (int i = 0; i < n; i++)
    {
        fp eigVal = resX[i];
        fp *eigVec = vecs + i * n;
        // calculate each row of Ax
        fp sum = 0.0;
        for (int j = 0; j < n; j++)
        {
            sum += matA[j * n + i] * eigVec[j];
        }
        // calculate norm of Ax - lambda x, should equal 0
        errorNorm += std::pow<fp, int>((sum - eigVal * eigVec[i]), 2);
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

        // compute spectrum
        // DPCT1007:2: Migration of cusolverDnDsyevd is not supported.
        oneapi::mkl::lapack::syevd(q_ct1, oneapi::mkl::job::vec, oneapi::mkl::uplo::lower, N, matA_d, lda, eigenVal_d, work_d, lwork);
        q_ct1.wait();
        __TIME_END
        std::cout << "No. " << i << " run, GPU calculation time = " << elapsedTime << "ms\n";
    }

    q_ct1.memcpy(eigenVec_h, matA_d, sizeof(fp) * matSize);
    q_ct1.memcpy(eigenVal_h, eigenVal_d, sizeof(fp) * N).wait();

#ifdef SHOW_MATRIX
    std::cout << "Eigenvaule = \n";
    print_matrix(eigenVal_h, N, 1);
    std::cout << "Eigenvector = \n";
    print_matrix(eigenVec_h, N, N);
#endif

    check_result(matA_h, eigenVal_h, eigenVec_h, N);

    resources_free();
    return 0;
}