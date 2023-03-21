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

typedef double fp;
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
int info_h = 0;

fp *matA_h, *vecb_h, *resx_h, *resx_href;
fp *matA_d, *vecb_d, *LU_h;

int *P_h, *P_d;

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
    resx_h = new fp[N]();
    resx_href = new fp[N]();
    LU_h = new fp[matSize]();

    cudaMalloc((void **)&matA_d, matSize * sizeof(fp));
    cudaMalloc((void **)&vecb_d, N * sizeof(fp));
    cudaMalloc((void **)&info_d, sizeof(int));

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
    memset(resx_href, 0, N * sizeof(fp));

    cudaMemcpy(matA_d, matA_h, matSize * sizeof(fp), cudaMemcpyHostToDevice);
    cudaMemcpy(vecb_d, vecb_h, N * sizeof(fp), cudaMemcpyHostToDevice);

    cusolverDnCreate(&cusolverH);
    // cusolverDnSetStream(cusolverH, stream)

    cusolverDnDgetrf_bufferSize(cusolverH, N, N, matA_d, lda, &lwork); // cusolverDnSgetrf_bufferSize
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
}

void resources_free()
{
    delete[] matA_h;
    delete[] vecb_h;
    delete[] resx_h;
    delete[] resx_href;
    delete[] LU_h;

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
__TIME_BEGIN
    cusolverDnDgetrf(cusolverH, N, N, matA_d, lda, work_d, P_d, info_d);
    // cudaMemcpy(&info_h, info_d, sizeof(int), cudaMemcpyDeviceToHost);
    cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, N, 1, matA_d, lda, P_d, vecb_d, ldb, info_d);
__TIME_END
    cudaMemcpy(&info_h, info_d, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(LU_h, matA_d, sizeof(fp) * matSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(resx_h, vecb_d, sizeof(fp) * N, cudaMemcpyDeviceToHost);

    if (0 > info_h)
    {
        std::cout << -info_h << "-th parameter is wrong \n";
        exit(2);
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
    std::cout << "GPU calculation time = " << elapsedTime << "ms\n";

    resources_free();
    return 0;
}