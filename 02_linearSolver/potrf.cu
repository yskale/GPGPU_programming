// nvcc -arch=sm_70 -lcublas -lcusolver potrf.cu
// solve Ax=b, with Cholesky factorization for positive definite Hermitian (symmetry) matrix
// A = L0*(L0*T), where *T = conjugate transpose

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
const int N = 1000;
constexpr int lda = N;
constexpr int ldb = N;
constexpr int matSize = N * N;
cusolverDnHandle_t cusolverH = NULL;

int lwork = 0;         /* size of workspace */
fp *work_d = nullptr;  /* device workspace for potrf */
int *info_d = nullptr; /* error info */
int *info_h = nullptr;
const int info_size = 2;

fp *matA_h, *vecb_h, *resx_h;
fp *matA_d, *vecb_d;
fp *L0; /* cholesky factor of A */

cudaEvent_t start, stop;
float elapsedTime;

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
    matA_h = new fp[matSize]();
    vecb_h = new fp[N]();
    // matA_h = new fp[matSize]{1.0, 2.0, 3.0, 2.0, 5.0, 5.0, 3.0, 5.0, 12.0};
    // vecb_h = new fp[N]{1.0, 1.0, 1.0};
    resx_h = new fp[N]();
    info_h = new int[info_size]();
    L0 = new fp[matSize]();

    cudaMalloc((void **)&matA_d, matSize * sizeof(fp));
    cudaMalloc((void **)&vecb_d, N * sizeof(fp));
    cudaMalloc((void **)&info_d, sizeof(int) * info_size);

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

    for (int i = 0; i < N; i++)
        vecb_h[i] = rand() / (fp)RAND_MAX * 1.0;

    memset(resx_h, 0, N * sizeof(fp));

    cudaMemcpy(matA_d, matA_h, matSize * sizeof(fp), cudaMemcpyHostToDevice);
    cudaMemcpy(vecb_d, vecb_h, N * sizeof(fp), cudaMemcpyHostToDevice);

    cusolverDnCreate(&cusolverH);
    // cusolverDnSetStream(cusolverH, stream)

#ifdef DOUBLE_FP_CASE
    cusolverDnDpotrf_bufferSize(cusolverH, CUBLAS_FILL_MODE_LOWER, N, matA_d, lda, &lwork);
#else
    cusolverDnSpotrf_bufferSize(cusolverH, CUBLAS_FILL_MODE_LOWER, N, matA_d, lda, &lwork);
#endif
    cudaMalloc((void **)&work_d, sizeof(fp) * lwork);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

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
    cudaMemcpy(matA_d, matA_h, matSize * sizeof(fp), cudaMemcpyHostToDevice);
    cudaMemcpy(vecb_d, vecb_h, N * sizeof(fp), cudaMemcpyHostToDevice);
}

void resources_free()
{
    delete[] matA_h;
    delete[] vecb_h;
    delete[] resx_h;
    delete[] info_h;
    delete[] L0;

    cudaFree(matA_d);
    cudaFree(vecb_d);
    cudaFree(info_d);
    cudaFree(work_d);
    cusolverDnDestroy(cusolverH);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaDeviceReset();
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
#ifdef DOUBLE_FP_CASE
        // Cholesky factorization
        cusolverDnDpotrf(cusolverH, CUBLAS_FILL_MODE_LOWER, N, matA_d, lda, work_d, lwork, &info_d[0]);
        // solve A*x = b
        cusolverDnDpotrs(cusolverH, CUBLAS_FILL_MODE_LOWER, N, 1, matA_d, lda, vecb_d, ldb, &info_d[1]);
#else
        // Cholesky factorization
        cusolverDnSpotrf(cusolverH, CUBLAS_FILL_MODE_LOWER, N, matA_d, lda, work_d, lwork, &info_d[0]);
        // solve A*x = b
        cusolverDnSpotrs(cusolverH, CUBLAS_FILL_MODE_LOWER, N, 1, matA_d, lda, vecb_d, ldb, &info_d[1]);
#endif
        __TIME_END
        std::cout << "No. " << i << " run, GPU calculation time = " << elapsedTime << "ms\n";
    }

    cudaMemcpy(L0, matA_d, sizeof(fp) * matSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(resx_h, vecb_d, sizeof(fp) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(info_h, info_d, sizeof(int) * info_size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < info_size; i++)
    {
        if (i == 0 && 2 == info_h[i])
            std::cout << "Error, Matrix A is not positive definite \n";
        if (0 > info_h[i])
        {
            std::cout << "i = " << i << ", " << -info_h[i] << "-th parameter is wrong \n";
            exit(i + 1);
        }
    }

#ifdef SHOW_MATRIX
    std::cout << "L0 = (upper triangle doesn't matter, which is same as A) \n";
    print_matrix(L0, N, N);
    std::cout << "x = \n";
    print_matrix(resx_h, N, 1);
#endif

    check_result(matA_h, resx_h, vecb_h, N);

    resources_free();
    return 0;
}