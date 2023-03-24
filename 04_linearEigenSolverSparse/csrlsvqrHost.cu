// nvcc -arch=sm_70 -lcublas -lcusolver -lcusparse csrlsvqrHost.cu
// solve Ax=b, with QR factorization, support both host & device execution, this is host version
// ref: https://stackoverflow.com/questions/30060067/cusolverspdcsrlsvlu-or-qr-method-using-cuda
#include "include/csr.hpp"

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
constexpr int matSize = N * N;
fp *matA_h, *vecb_h, *resx_h;
int singularity = 0;
csrMat<fp> csrA_h;
cusolverSpHandle_t cusolverH = NULL;
cusparseMatDescr_t descrA = NULL;
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

    for (int i = 0; i < matSize; i++)
    {
        matA_h[i] = rand() / (fp)RAND_MAX * 1.0;
        if (rand() / (fp)RAND_MAX * 1.0 < sparselevel) // make the matrix become sparse
            matA_h[i] = 0.0;
    }

    for (int i = 0; i < N; i++)
        vecb_h[i] = rand() / (fp)RAND_MAX * 1.0;

    int numNozeroA = 0;
    for (int i = 0; i < matSize; i++)
    {
        if (std::fabs(matA_h[i]) > 0.0)
            numNozeroA++;
    }

    csrA_h.init(numNozeroA, N, N, memType::host);
    csrA_h.dense2csrHost(matA_h);
#ifdef SHOW_MATRIX
    csrA_h.printCsrMatrixHost();
    csrA_h.printCsrFormHost();
#endif

    memset(resx_h, 0, N * sizeof(fp));

    cusolverSpCreate(&cusolverH);
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

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

    cusolverSpDestroy(cusolverH);
    cusparseDestroyMatDescr(descrA);

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
        cusolverSpDcsrlsvqrHost(cusolverH, csrA_h.nA, csrA_h.numA, descrA, csrA_h.csrValA, csrA_h.csrRowPtrA, csrA_h.csrColIndA, vecb_h, 0.0, 0, resx_h, &singularity);
#else
        cusolverSpScsrlsvqrHost(cusolverH, csrA_h.nA, csrA_h.numA, descrA, csrA_h.csrValA, csrA_h.csrRowPtrA, csrA_h.csrColIndA, vecb_h, 0.0, 0, resx_h, &singularity);
#endif
        __TIME_END
        std::cout << "No. " << i << " run, CPU calculation time = " << elapsedTime << "ms\n";
    }

#ifdef SHOW_MATRIX
    std::cout << "x = \n";
    print_matrix(resx_h, N, 1);
#endif
    if(singularity == -1)
        check_result(matA_h, resx_h, vecb_h, N);
    else
        std::cout << "A is uninvertible, singularity = " << singularity << std::endl;

    resources_free();
    return 0;
}