// nvcc -arch=sm_70 -lcublas -lcusolver gesvd.cu
// solve A = U \Sigma V^T, Computes the singular value
// decomposition (SVD) of a general MxN rectangular matrix.
// Remark 1: gesvd only supports m>=n.
// Remark 2: the routine returns V^T, not V

#include <iostream>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <limits>

#define __TIME_BEGIN cudaEventRecord(start);
#define __TIME_END              \
    cudaEventRecord(stop);      \
    cudaEventSynchronize(stop); \
    cudaEventElapsedTime(&elapsedTime, start, stop);

// #define SHOW_MATRIX
// #define DOUBLE_FP_CASE
#ifdef DOUBLE_FP_CASE
typedef double fp;
#else
typedef float fp;
#endif

const fp sparselevel = 0.3;
const int M = 1000; // rows
const int N = 800;  // cols, N <= M
static_assert(M >= N);
constexpr int lda = M;
constexpr int matAsize = M * N;
constexpr int numSVD = std::min(M, N);
cusolverDnHandle_t cusolverH = NULL;
cublasHandle_t cublasH = NULL;

int lwork = 0;        /* size of workspace */
fp *work_d = nullptr; /* device workspace for potrf */
fp *rwork_d = nullptr;
int *info_d = nullptr; /* error info */
int *info_h = nullptr;
const int info_size = 1;

fp *matA_h /* MxN */, *matU_h /* MxM */, *matVT_h /* NxN, but shoud use MxN, not know why */, *sVal_h /* min(M, N) */;
fp *matA_d /* MxN */, *matU_d /* MxM */, *matVT_d /* NxN, but shoud use MxN, not know why */, *sVal_d /* min(M, N) */;

cudaEvent_t start, stop;
float elapsedTime;

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
    matA_h = new fp[matAsize]();
    matU_h = new fp[M * M]();
    matVT_h = new fp[M * N](); // VT is extended to M * N, not know why, otherwise, the result is not correct
    sVal_h = new fp[numSVD]();
    // matA_h = new fp[matAsize]{1.0, 4.0, 2.0, 2.0, 5.0, 1.0};
    // matA_h = new fp[matAsize]{0.840188, 0.783099, 0.911647, 0.335223, 0.277775, 0.477397};
    // sVal_h = new fp[numSVD]{7.065283497082729, 1.040081297712078};
    info_h = new int[info_size]();

    cudaMalloc((void **)&matA_d, matAsize * sizeof(fp));
    cudaMalloc((void **)&matU_d, M * M * sizeof(fp));
    cudaMalloc((void **)&matVT_d, M * N * sizeof(fp));
    cudaMalloc((void **)&sVal_d, numSVD * sizeof(fp));
    cudaMalloc((void **)&info_d, sizeof(int) * info_size);

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

    cudaMemcpy(matA_d, matA_h, matAsize * sizeof(fp), cudaMemcpyHostToDevice);
    cudaMemcpy(matU_d, matU_h, M * M * sizeof(fp), cudaMemcpyHostToDevice);
    cudaMemcpy(matVT_d, matVT_h, M * N * sizeof(fp), cudaMemcpyHostToDevice);
    cudaMemcpy(sVal_d, sVal_h, numSVD * sizeof(fp), cudaMemcpyHostToDevice);

    cusolverDnCreate(&cusolverH);
    cublasCreate(&cublasH);
    // cusolverDnSetStream(cusolverH, stream)

#ifdef DOUBLE_FP_CASE
    cusolverDnDgesvd_bufferSize(cusolverH, M, N, &lwork);
#else
    cusolverDnSgesvd_bufferSize(cusolverH, M, N, &lwork);
#endif
    cudaMalloc((void **)&work_d, sizeof(fp) * lwork);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

#ifdef SHOW_MATRIX
    std::cout << "A = \n";
    print_matrix(matA_h, M, N);
#endif
}

void result_reset()
{
    cudaMemcpy(matA_d, matA_h, matAsize * sizeof(fp), cudaMemcpyHostToDevice);
}

void resources_free()
{
    delete[] matA_h;
    delete[] matU_h;
    delete[] matVT_h;
    delete[] sVal_h;
    delete[] info_h;

    cudaFree(matA_d);
    cudaFree(matU_d);
    cudaFree(matVT_d);
    cudaFree(sVal_d);
    cudaFree(info_d);
    cudaFree(work_d);

    cusolverDnDestroy(cusolverH);
    cublasDestroy(cublasH);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaDeviceReset();
}

void check_result(fp *_matA_d, fp *_matU_d, fp *_matVT_d, fp *_sVal_d, int m, int n)
{

    const fp h_one = 1;
    const fp h_minus_one = -1;
    fp *_W_d = nullptr; /* W = diag(s)*VT */
    cudaMalloc((void **)&_W_d, sizeof(fp) * lda * n);

#ifdef DOUBLE_FP_CASE
    // calculate _W = diag(_sVal) x _matVT
    cublasDdgmm(cublasH, CUBLAS_SIDE_LEFT, n, n, _matVT_d, lda, _sVal_d, 1, _W_d, lda);
#else
    cublasSdgmm(cublasH, CUBLAS_SIDE_LEFT, n, n, _matVT_d, lda, _sVal_d, 1, _W_d, lda);
#endif
    cudaMemcpy(_matA_d, matA_h, sizeof(fp) * matAsize, cudaMemcpyHostToDevice);
#ifdef DOUBLE_FP_CASE
    // calculate - _matU x diag(_sVal) x _matVT + matA
    cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n, &h_minus_one, _matU_d, lda, _W_d, lda, &h_one, _matA_d, lda);
#else
    cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n, &h_minus_one, _matU_d, lda, _W_d, lda, &h_one, _matA_d, lda);
#endif

    fp dR_fro = 0.0;
#ifdef DOUBLE_FP_CASE
    cublasDnrm2(cublasH, lda * n, _matA_d, 1, &dR_fro);
#else
    cublasSnrm2(cublasH, lda * n, _matA_d, 1, &dR_fro);
#endif

    std::cout << "error 2-norm = " << dR_fro << std::endl;
    cudaFree(_W_d);
}

int main()
{
    resources_init();
    for (int i = 0; i < 10; i++)
    {
        result_reset();
        __TIME_BEGIN
#ifdef DOUBLE_FP_CASE
        // compute SVD, note here the ldv should equals to M instead of N, otherwise, the check_result is not correct, why ??????
        cusolverDnDgesvd(cusolverH, 'A', 'A', M, N, matA_d, lda, sVal_d, matU_d, lda, matVT_d, lda /*ldv is set to lda*/, work_d, lwork, rwork_d, &info_d[0]);
#else
        // compute SVD, note here the ldv should equals to M instead of N, otherwise, the check_result is not correct, why ??????
        cusolverDnSgesvd(cusolverH, 'A', 'A', M, N, matA_d, lda, sVal_d, matU_d, lda, matVT_d, lda /*ldv is set to lda*/, work_d, lwork, rwork_d, &info_d[0]);
#endif
        __TIME_END
        std::cout << "No. " << i << " run, GPU calculation time = " << elapsedTime << "ms\n";
    }

    cudaMemcpy(matU_h, matU_d, sizeof(fp) * M * M, cudaMemcpyDeviceToHost);
    cudaMemcpy(matVT_h, matVT_d, sizeof(fp) * M * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(sVal_h, sVal_d, sizeof(fp) * numSVD, cudaMemcpyDeviceToHost);
    cudaMemcpy(info_h, info_d, sizeof(int) * info_size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < info_size; i++)
    {
        if (0 > info_h[i])
        {
            std::cout << "i = " << i << ", " << -info_h[i] << "-th parameter is wrong \n";
            exit(i + 1);
        }
        else if (0 < info_h[i])
        {
            std::cout << "WARNING: info = " << info_h[i] << ": gesvd does not converge \n";
        }
    }

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