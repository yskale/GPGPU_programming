// icpx -fsycl -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core getrf.cpp
// clang++ -fsycl -I${MKLROOT}/include -L${MKLROOT}/lib -lonemkl -lonemkl_blas_mklcpu -lonemkl_blas_mklgpu -lonemkl_lapack_mklcpu -lonemkl_lapack_mklgpu -O1 getrf.cpp
// clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_70 -I${MKLROOT}/include -L${MKLROOT}/lib -lonemkl -lonemkl_blas_cublas -lonemkl_lapack_cusolver getrf.cpp
// solve Ax=b, with LU or PLU factorization
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <oneapi/mkl.hpp>
// #include <dpct/blas_utils.hpp>

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

const bool pivot_on = true;
const fp sparselevel = 0.3;
const int N = 1000;
constexpr int lda = N;
constexpr int ldb = N;
constexpr int matSize = N * N;

int64_t lworkMat = 0;    /* size of workspace */
fp *workMat_d = nullptr; /* device workspace for getrf */
int64_t lworkVec = 0;    /* size of workspace */
fp *workVec_d = nullptr; /* device workspace for getrs */

fp *matA_h, *vecb_h, *resx_h;
fp *matA_d, *vecb_d, *LU_h;

int64_t *P_h, *P_d;

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
    LU_h = new fp[matSize]();

    matA_d = sycl::malloc_device<fp>(matSize, q_ct1);
    vecb_d = sycl::malloc_device<fp>(N, q_ct1);

    assert(pivot_on);
    if (pivot_on)
    {
        std::cout << "pivot is on : compute P*A = L*U \n";
        P_d = sycl::malloc_device<int64_t>(N, q_ct1);
        q_ct1.memset(P_d, 0, N * sizeof(fp)).wait();
        P_h = new int64_t[N]();
    }
    else
    {
        std::cout << "pivot is off: compute A = L*U (not numerically stable)\n";
        P_d = nullptr;
        P_h = nullptr;
    }

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

    lworkMat = oneapi::mkl::lapack::getrf_scratchpad_size<fp>(q_ct1, N, N, lda);
    workMat_d = sycl::malloc_device<fp>(lworkMat, q_ct1);

    lworkVec = oneapi::mkl::lapack::getrf_scratchpad_size<fp>(q_ct1, N, N, lda);
    workVec_d = sycl::malloc_device<fp>(lworkVec, q_ct1);

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
    delete[] LU_h;

    sycl::free(matA_d, q_ct1);
    sycl::free(vecb_d, q_ct1);
    if (pivot_on)
    {
        delete[] P_h;
        sycl::free(P_d, q_ct1);
    }
    sycl::free(workMat_d, q_ct1);
    sycl::free(workVec_d, q_ct1);

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
        /*
        DPCT1047:1: The meaning of P_d in the oneapi::mkl::lapack::getrf is
        different from the cusolverDnDgetrf. You may need to check the migrated
        code.
        */
        __TIME_BEGIN
        oneapi::mkl::lapack::getrf(q_ct1, N, N, matA_d, lda,
                                   P_d, workMat_d, lworkMat);
        oneapi::mkl::lapack::getrs(
            q_ct1, oneapi::mkl::transpose::nontrans, N, 1, matA_d,
            lda, P_d, vecb_d, ldb, workVec_d, lworkVec);
        q_ct1.wait();
        __TIME_END
        std::cout << "No. " << i << " run, GPU calculation time = " << elapsedTime << "ms\n";
    }
    __TIME_BEGIN
    // std::vector<void *> ws_vec_ct5{workVec_d};
    // dpct::async_dpct_free(ws_vec_ct5, {event_ct4}, q_ct1);
    q_ct1.memcpy(LU_h, matA_d, sizeof(fp) * matSize);
    q_ct1.memcpy(resx_h, vecb_d, sizeof(fp) * N).wait();

#ifdef SHOW_MATRIX
    if (pivot_on)
    {
        q_ct1.memcpy(P_h, P_d, sizeof(int) * N).wait();
        std::cout << "pivoting sequence\n";
        for (int j = 0; j < N; j++)
        {
            std::cout << "P_h(" << j << ") = " << P_h[j] << "\n";
        }
    }
    std::cout << "L and U = \n";
    print_matrix(LU_h, N, N);
    std::cout << "x = \n";
    print_matrix(resx_h, N, 1);
#endif

    check_result(matA_h, resx_h, vecb_h, N);

    resources_free();
    return 0;
}