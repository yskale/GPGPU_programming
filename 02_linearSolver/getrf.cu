// nvcc -arch=sm_70 -lcublas -lcusolver getrf.cu
// solve Ax=b, with LU or PLU factorization
#include <iostream>
#include <cuda_runtime.h>
#include <cusolverDn.h>

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

const bool pivot_on = true;
const fp sparselevel = 0.3;
const int N = 1000;
constexpr int lda = N;
constexpr int ldb = N;
constexpr int matSize = N * N;
cusolverDnHandle_t cusolverH = NULL;
int lwork = 0;         /* size of workspace */
fp *work_d = nullptr;  /* device workspace for getrf */
int *info_d = nullptr; /* error info */
int *info_h = nullptr;
const int info_size = 2;

fp *matA_h, *vecb_h, *resx_h;
fp *matA_d, *vecb_d, *LU_h;

int *P_h, *P_d;

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
    matA_h = new fp[matSize]();
    vecb_h = new fp[N]();
    resx_h = new fp[N]();
    LU_h = new fp[matSize]();
    info_h = new int[info_size]();

    cudaMalloc((void **)&matA_d, matSize * sizeof(fp));
    cudaMalloc((void **)&vecb_d, N * sizeof(fp));
    cudaMalloc((void **)&info_d, sizeof(int) * info_size);

    if (pivot_on)
    {
        std::cout << "pivot is on : compute P*A = L*U \n";
        cudaMalloc((void **)&P_d, N * sizeof(int));
        cudaMemset(P_d, 0, N * sizeof(fp));
        P_h = new int[N]();
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

    cudaMemcpy(matA_d, matA_h, matSize * sizeof(fp), cudaMemcpyHostToDevice);
    cudaMemcpy(vecb_d, vecb_h, N * sizeof(fp), cudaMemcpyHostToDevice);

    cusolverDnCreate(&cusolverH);
    // cusolverDnSetStream(cusolverH, stream)
#ifdef DOUBLE_FP_CASE
    cusolverDnDgetrf_bufferSize(cusolverH, N, N, matA_d, lda, &lwork);
#else
    cusolverDnSgetrf_bufferSize(cusolverH, N, N, matA_d, lda, &lwork);
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
    delete[] LU_h;
    delete[] info_h;

    cudaFree(matA_d);
    cudaFree(vecb_d);
    if (pivot_on)
    {
        delete[] P_h;
        cudaFree(P_d);
    }
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
        cusolverDnDgetrf(cusolverH, N, N, matA_d, lda, work_d, P_d, &info_d[0]);
        cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, N, 1, matA_d, lda, P_d, vecb_d, ldb, &info_d[1]);
#else
        cusolverDnSgetrf(cusolverH, N, N, matA_d, lda, work_d, P_d, &info_d[1]);
        cusolverDnSgetrs(cusolverH, CUBLAS_OP_N, N, 1, matA_d, lda, P_d, vecb_d, ldb, &info_d[1]);
#endif
        __TIME_END
        std::cout << "No. " << i << " run, GPU calculation time = " << elapsedTime << "ms\n";
    }
    
    cudaMemcpy(LU_h, matA_d, sizeof(fp) * matSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(resx_h, vecb_d, sizeof(fp) * N, cudaMemcpyDeviceToHost);
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
    if (pivot_on)
    {
        cudaMemcpy(P_h, P_d, sizeof(int) * N, cudaMemcpyDeviceToHost);
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