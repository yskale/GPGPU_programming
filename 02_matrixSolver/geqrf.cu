// nvcc -arch=sm_70 -lcublas -lcusolver geqrf.cu
// solve Ax=b, with QR factorization
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

const fp sparselevel = 0.3;
const int N = 1000;
constexpr int lda = N;
constexpr int ldb = N;
constexpr int matSize = N * N;
const fp alpha = 1;
cusolverDnHandle_t cusolverH = NULL;
cublasHandle_t cublasH = NULL;
int lwork = 0;         /* size of workspace */
fp *work_d = nullptr;  /* device workspace for getrf */
int *info_d = nullptr; /* error info */
int *info_h = nullptr;
const int info_size = 2;

fp *matA_h, *vecb_h, *resx_h;
fp *matA_d, *vecb_d, *tau_d;

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
    info_h = new int[info_size]();

    cudaMalloc((void **)&matA_d, matSize * sizeof(fp));
    cudaMalloc((void **)&vecb_d, N * sizeof(fp));
    cudaMalloc((void **)&info_d, sizeof(int) * info_size);
    cudaMalloc((void **)&tau_d, sizeof(fp) * N);

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
    cublasCreate(&cublasH);
    // cusolverDnSetStream(cusolverH, stream)
    int lwork_geqrf = 0;
    int lwork_ormqr = 0;
#ifdef DOUBLE_FP_CASE
    cusolverDnDgeqrf_bufferSize(cusolverH, N, N, matA_d, lda, &lwork_geqrf);
    cusolverDnDormqr_bufferSize(cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, N, 1, N, matA_d, lda, tau_d, vecb_d, ldb, &lwork_ormqr);
#else
    cusolverDnSgeqrf_bufferSize(cusolverH, N, N, matA_d, lda, &lwork_geqrf);
    cusolverDnSormqr_bufferSize(cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, N, 1, N, matA_d, lda, tau_d, vecb_d, ldb, &lwork_ormqr);
#endif
    lwork = std::max(lwork_geqrf, lwork_ormqr);
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

    cudaFree(matA_d);
    cudaFree(vecb_d);
    cudaFree(info_d);
    cudaFree(work_d);
    cudaFree(tau_d);
    cusolverDnDestroy(cusolverH);
    cublasDestroy(cublasH);

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
        // compute QR factorization
        cusolverDnDgeqrf(cusolverH, N, N, matA_d, lda, tau_d, work_d, lwork, &info_d[0]);
        // compute Q^T*B
        cusolverDnDormqr(cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, N, 1, N,
                         matA_d, lda, tau_d, vecb_d, ldb, work_d, lwork, &info_d[1]);
        // compute x = R \ Q^T*B
        cublasDtrsm(cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                    CUBLAS_DIAG_NON_UNIT, N, 1, &alpha, matA_d, lda, vecb_d, ldb);
#else
        // compute QR factorization
        cusolverDnSgeqrf(cusolverH, N, N, matA_d, lda, tau_d, work_d, lwork, &info_d[0]);
        // compute Q^T*B
        cusolverDnSormqr(cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, N, 1, N,
                         matA_d, lda, tau_d, vecb_d, ldb, work_d, lwork, &info_d[1]);
        // compute x = R \ Q^T*B
        cublasStrsm(cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                    CUBLAS_DIAG_NON_UNIT, N, 1, &alpha, matA_d, lda, vecb_d, ldb);
#endif
        __TIME_END
        std::cout << "No. " << i << " run, GPU calculation time = " << elapsedTime << "ms\n";
    }
    
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
    std::cout << "x = \n";
    print_matrix(resx_h, N, 1);
#endif

    check_result(matA_h, resx_h, vecb_h, N);

    resources_free();
    return 0;
}