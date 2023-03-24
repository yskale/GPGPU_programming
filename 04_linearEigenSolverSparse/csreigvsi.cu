// nvcc -arch=sm_70 -lcublas -lcusolver -lcusparse csreigvsi.cu
// solve Ax=\lambda x, with shift-inverse method, support both host & device execution, this is device version
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
fp *eigenVec_d, *eigenVec_d0, *eigenVal_d;
const fp eigenVal_d0 = 0.0; // guessed eigen value
// guessed eigen vec initially stored in eigenVec_h
int singularity = 0;
const int maxIterNum = 1000;

csrMat<fp> csrA_h;
csrMat<fp> csrA_d;
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

    cudaMalloc((void **)&eigenVec_d, N * sizeof(fp));
    cudaMalloc((void **)&eigenVec_d0, N * sizeof(fp));
    cudaMalloc((void **)&eigenVal_d, 1 * sizeof(fp));

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
        eigenVec_h[i] = rand() / (fp)RAND_MAX * 1.0;

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
    csrA_d.init(numNozeroA, N, N, memType::device);
    csrA_d.copyFromHost(csrA_h);

    cudaMemcpy(eigenVec_d0, eigenVec_h, N * sizeof(fp), cudaMemcpyHostToDevice);
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
    csrA_d.copyFromHost(csrA_h);
}

void resources_free()
{
    delete[] matA_h;
    delete[] eigenVec_h;

    cudaFree(eigenVec_d);
    cudaFree(eigenVec_d0);
    cudaFree(eigenVal_d);

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
        cusolverSpDcsreigvsi(cusolverH, csrA_d.nA, csrA_d.numA, descrA, csrA_d.csrValA, csrA_d.csrRowPtrA, csrA_d.csrColIndA, eigenVal_d0, eigenVec_d0, maxIterNum, 0.0, eigenVal_d, eigenVec_d);
#else
        cusolverSpScsreigvsi(cusolverH, csrA_d.nA, csrA_d.numA, descrA, csrA_d.csrValA, csrA_d.csrRowPtrA, csrA_d.csrColIndA, eigenVal_d0, eigenVec_d0, maxIterNum, 0.0, eigenVal_d, eigenVec_d);
#endif
        __TIME_END
        std::cout << "No. " << i << " run, GPU calculation time = " << elapsedTime << "ms\n";
    }
    cudaMemcpy(eigenVec_h, eigenVec_d, N * sizeof(fp), cudaMemcpyDeviceToHost);
    cudaMemcpy(&eigenVal_h, eigenVal_d, sizeof(fp), cudaMemcpyDeviceToHost);

    std::cout << "Find an eigenvaule = " << eigenVal_h << std::endl;
#ifdef SHOW_MATRIX
    std::cout << "Eigenvector = \n";
    print_matrix(eigenVec_h, N, 1);
#endif

    check_result(matA_h, eigenVal_h, eigenVec_h, N);

    resources_free();
    return 0;
}