// nvcc -arch=sm_70 -lcublas -lcusolver syevd.cu
// solve Ax=\lambda x, using syevd to compute the spectrum of a dense symmetric (Hermitian) system

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
constexpr int matSize = N * N;
cusolverDnHandle_t cusolverH = NULL;

int lwork = 0;         /* size of workspace */
fp *work_d = nullptr;  /* device workspace for potrf */
int *info_d = nullptr; /* error info */
int *info_h = nullptr;
const int info_size = 1;

fp *matA_h, *eigenVec_h, *eigenVal_h;
fp *matA_d, *eigenVal_d;

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
    eigenVec_h = new fp[matSize]();
    eigenVal_h = new fp[N]();
    // matA_h = new fp[matSize]{3.5, 0.5, 0.0, 0.5, 3.5, 0.0, 0.0, 0.0, 2.0};
    // eigenVec_h = new fp[matSize]{0.00, 0.00, 1.00, 0.71, -0.71, 0.00, 0.71, 0.71, 0.00};
    // eigenVal_h = new fp[N]{2.0, 3.0, 4.0};
    info_h = new int[info_size]();

    cudaMalloc((void **)&matA_d, matSize * sizeof(fp));
    cudaMalloc((void **)&eigenVal_d, N * sizeof(fp));
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

    memset(eigenVec_h, 0, matSize * sizeof(fp));
    memset(eigenVal_h, 0, N * sizeof(fp));

    cudaMemcpy(matA_d, matA_h, matSize * sizeof(fp), cudaMemcpyHostToDevice);
    cudaMemcpy(eigenVal_d, eigenVal_h, N * sizeof(fp), cudaMemcpyHostToDevice);

    cusolverDnCreate(&cusolverH);
    // cusolverDnSetStream(cusolverH, stream)

#ifdef DOUBLE_FP_CASE
    cusolverDnDsyevd_bufferSize(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, N, matA_d, lda, eigenVal_d, &lwork);
#else
    cusolverDnSsyevd_bufferSize(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, N, matA_d, lda, eigenVal_d, &lwork);
#endif
    cudaMalloc((void **)&work_d, sizeof(fp) * lwork);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

#ifdef SHOW_MATRIX
    std::cout << "A = \n";
    print_matrix(matA_h, N, N);
#endif
}

void result_reset()
{
    memset(eigenVal_h, 0, N * sizeof(fp));
    cudaMemcpy(matA_d, matA_h, matSize * sizeof(fp), cudaMemcpyHostToDevice);
}

void resources_free()
{
    delete[] matA_h;
    delete[] eigenVec_h;
    delete[] eigenVal_h;
    delete[] info_h;

    cudaFree(matA_d);
    cudaFree(eigenVal_d);
    cudaFree(info_d);
    cudaFree(work_d);
    cusolverDnDestroy(cusolverH);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaDeviceReset();
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
#ifdef DOUBLE_FP_CASE
        // compute spectrum
        cusolverDnDsyevd(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, N, matA_d, lda, eigenVal_d, work_d, lwork, &info_d[0]);
#else
        // compute spectrum
        cusolverDnSsyevd(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, N, matA_d, lda, eigenVal_d, work_d, lwork, &info_d[0]);
#endif
        __TIME_END
        std::cout << "No. " << i << " run, GPU calculation time = " << elapsedTime << "ms\n";
    }

    cudaMemcpy(eigenVec_h, matA_d, sizeof(fp) * matSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(eigenVal_h, eigenVal_d, sizeof(fp) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(info_h, info_d, sizeof(int) * info_size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < info_size; i++)
    {
        if (0 > info_h[i])
        {
            std::cout << "i = " << i << ", " << -info_h[i] << "-th parameter is wrong \n";
            exit(i + 1);
        }
    }

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