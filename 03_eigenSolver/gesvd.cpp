// icpx -fsycl -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core gesvd.cpp
// solve A = U \Sigma V^T, Computes the singular value
// decomposition (SVD) of a general MxN rectangular matrix.
// Remark 1: gesvd only supports m>=n.
// Remark 2: the routine returns V^T, not V

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
const int M = 1000;    // rows
const int N = 800;     // cols, N <= M
static_assert(M >= N); // Not required for gesvd in oneMKL, but necessary in cuSolver. Pls modify this code if M < N is used.
constexpr int lda = M;
constexpr int matAsize = M * N;
constexpr int numSVD = std::min(M, N);

int lwork = 0;        /* size of workspace */
fp *work_d = nullptr; /* device workspace for potrf */

fp *matA_h /* MxN */, *matU_h /* MxM */, *matVT_h /* NxN, but shoud use MxN, not know why */, *sVal_h /* min(M, N) */;
fp *matA_d /* MxN */, *matU_d /* MxM */, *matVT_d /* NxN, but shoud use MxN, not know why */, *sVal_d /* min(M, N) */;

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

    matA_h = new fp[matAsize]();
    matU_h = new fp[M * M]();
    matVT_h = new fp[M * N](); // VT is extended to M * N, not know why, otherwise, the result is not correct
    sVal_h = new fp[numSVD]();
    // matA_h = new fp[matAsize]{1.0, 4.0, 2.0, 2.0, 5.0, 1.0};
    // matA_h = new fp[matAsize]{0.840188, 0.783099, 0.911647, 0.335223, 0.277775, 0.477397};
    // sVal_h = new fp[numSVD]{7.065283497082729, 1.040081297712078};

    matA_d = sycl::malloc_device<fp>(matAsize, q_ct1);
    matU_d = sycl::malloc_device<fp>(M * M, q_ct1);
    matVT_d = sycl::malloc_device<fp>(M * N, q_ct1);
    sVal_d = sycl::malloc_device<fp>(numSVD, q_ct1);

    // create a random matrix
    for (int i = 0; i < matAsize; i++)
    {
        matA_h[i] = rand() / (fp)RAND_MAX * 1.0;
        if (rand() / (fp)RAND_MAX * 1.0 < sparselevel) // make the matrix become sparse
            matA_h[i] = 0.0;
    }

    memset(matU_h, 0, M * M * sizeof(fp));
    memset(matVT_h, 0, M * N * sizeof(fp));
    memset(sVal_h, 0, numSVD * sizeof(fp));

    q_ct1.memcpy(matA_d, matA_h, matAsize * sizeof(fp));
    q_ct1.memcpy(matU_d, matU_h, M * M * sizeof(fp));
    q_ct1.memcpy(matVT_d, matVT_h, M * N * sizeof(fp));
    q_ct1.memcpy(sVal_d, sVal_h, numSVD * sizeof(fp)).wait();

    // {
    //     oneapi::mkl::jobsvd job_ct_mkl_jobu;
    //     oneapi::mkl::jobsvd job_ct_mkl_jobvt;
    //     lwork = oneapi::mkl::lapack::gesvd_scratchpad_size<float>(
    //         q_ct1, job_ct_mkl_jobu, job_ct_mkl_jobvt, M, N, lda, M, N);
    // }
    lwork = oneapi::mkl::lapack::gesvd_scratchpad_size<fp>(q_ct1, oneapi::mkl::jobsvd::A, oneapi::mkl::jobsvd::A, M, N, lda, M, M);
    work_d = sycl::malloc_device<fp>(lwork, q_ct1);

    start = new sycl::event();
    stop = new sycl::event();

#ifdef SHOW_MATRIX
    std::cout << "A = \n";
    print_matrix(matA_h, M, N);
#endif
}

void result_reset()
{
    q_ct1.memcpy(matA_d, matA_h, matAsize * sizeof(fp)).wait();
}

void resources_free()
{
    delete[] matA_h;
    delete[] matU_h;
    delete[] matVT_h;
    delete[] sVal_h;

    sycl::free(matA_d, q_ct1);
    sycl::free(matU_d, q_ct1);
    sycl::free(matVT_d, q_ct1);
    sycl::free(sVal_d, q_ct1);
    sycl::free(work_d, q_ct1);

    dpct::destroy_event(start);
    dpct::destroy_event(stop);

    dev_ct1.reset();
}

void check_result(fp *_matA_d, fp *_matU_d, fp *_matVT_d, fp *_sVal_d, int m, int n)
{

    fp h_one = 1;
    fp h_minus_one = -1;
    fp *_W_d = nullptr; /* W = diag(s)*VT */
    _W_d = sycl::malloc_device<fp>(lda * n, q_ct1);
    // oneapi::mkl::blas::column_major::dgmm_batch(
    //     q_ct1, oneapi::mkl::side::left, n, n, _matVT_d, lda, 0, _sVal_d, 1,
    //     0, _W_d, lda, lda * n, 1);
    oneapi::mkl::blas::column_major::dgmm(q_ct1, oneapi::mkl::side::left, n, n, _matVT_d, lda, _sVal_d, 1, _W_d, lda);
    q_ct1.memcpy(_matA_d, matA_h, sizeof(fp) * matAsize).wait();
    oneapi::mkl::blas::column_major::gemm(
        q_ct1, oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans, m, n, n, h_minus_one, _matU_d, lda,
        _W_d, lda, h_one, _matA_d, lda);

    fp *dR_fro = sycl::malloc_shared<fp>(1, q_ct1);
    // fp dR_fro = 0.0;
    oneapi::mkl::blas::column_major::nrm2(q_ct1, lda * n, _matA_d, 1, dR_fro);
    q_ct1.wait();

    std::cout << "error 2-norm = " << *dR_fro << std::endl;
    sycl::free(_W_d, q_ct1);
}

int main()
{
    resources_init();
    for (int i = 0; i < 10; i++)
    {
        result_reset();
        __TIME_BEGIN
        // compute SVD, note here the ldv should equals to M instead of N, otherwise, the check_result is not correct, why ??????
        oneapi::mkl::lapack::gesvd(q_ct1, oneapi::mkl::jobsvd::A, oneapi::mkl::jobsvd::A, M, N,
                                   matA_d, lda, sVal_d, matU_d, lda, matVT_d, lda /*ldv is set to lda*/, work_d, lwork);
        q_ct1.wait();
        __TIME_END
        std::cout << "No. " << i << " run, GPU calculation time = " << elapsedTime << "ms\n";
    }

    q_ct1.memcpy(matU_h, matU_d, sizeof(fp) * M * M);
    q_ct1.memcpy(matVT_h, matVT_d, sizeof(fp) * M * N);
    q_ct1.memcpy(sVal_h, sVal_d, sizeof(fp) * numSVD).wait();

#ifdef SHOW_MATRIX
    std::cout << "S (singular values) = \n";
    print_matrix(sVal_h, numSVD, 1);
    std::cout << "U (left singular vectors) = \n";
    print_matrix(matU_h, M, M);
    std::cout << "VT (right singular vectors) = \n";
    print_matrix(matVT_h, M, N);
#endif

    check_result(matA_d, matU_d, matVT_d, sVal_d, M, N);

    resources_free();
    return 0;
}