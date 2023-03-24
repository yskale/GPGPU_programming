// nvcc -arch=sm_70 -lcublas -lcusolver -lcusparse csreigvsiHost.cu
// solve Ax=\lambda x, with shift-inverse method, support both host & device execution, this is host version
// with the initial gussed input, this function caclulate only 1 eigenVal & eigenVec each time
#include "include/csr.hpp"
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
const int N = 10;
constexpr int matSize = N * N;
fp *matA_h, *eigenVec_h, eigenVal_h;
const fp eigenVal_h0 = 0.0; // guessed eigen value
fp *eigenVec_h0; // guessed eigen vec
int singularity = 0;
const int maxIterNum = 1000;

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
    eigenVec_h = new fp[N]();
    eigenVec_h0 = new fp[N]();

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
    // <positive definite Hermitian> is not necessary for this case, but random matrix may have coplex eigenvalues
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
        eigenVec_h0[i] = rand() / (fp)RAND_MAX * 1.0;

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

    memset(eigenVec_h, 0, N * sizeof(fp));

    cusolverSpCreate(&cusolverH);
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

#ifdef SHOW_MATRIX
    std::cout << "A = \n";
    print_matrix(matA_h, N, N);
#endif
}

void result_reset()
{
    memset(eigenVec_h, 0, N * sizeof(fp));
}

void resources_free()
{
    delete[] matA_h;
    delete[] eigenVec_h;
    delete[] eigenVec_h0;

    cusolverSpDestroy(cusolverH);
    cusparseDestroyMatDescr(descrA);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaDeviceReset();
}

void check_result(fp *matA, fp eigVal, fp *eigVec, int n)
{
    fp errorNorm = 0.0;
    for (int i = 0; i < n; i++)
    {
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
        cusolverSpDcsreigvsiHost(cusolverH, csrA_h.nA, csrA_h.numA, descrA, csrA_h.csrValA, csrA_h.csrRowPtrA, csrA_h.csrColIndA, eigenVal_h0, eigenVec_h0, maxIterNum, 0.0, &eigenVal_h, eigenVec_h);
#else
        cusolverSpScsreigvsiHost(cusolverH, csrA_h.nA, csrA_h.numA, descrA, csrA_h.csrValA, csrA_h.csrRowPtrA, csrA_h.csrColIndA, eigenVal_h0, eigenVec_h0, maxIterNum, 0.0, &eigenVal_h, eigenVec_h);
#endif
        __TIME_END
        std::cout << "No. " << i << " run, CPU calculation time = " << elapsedTime << "ms\n";
    }

    std::cout << "Find an eigenvaule = " << eigenVal_h << std::endl;
#ifdef SHOW_MATRIX
    std::cout << "Eigenvector = \n";
    print_matrix(eigenVec_h, N, 1);
#endif

    check_result(matA_h, eigenVal_h, eigenVec_h, N);

    resources_free();
    return 0;
}