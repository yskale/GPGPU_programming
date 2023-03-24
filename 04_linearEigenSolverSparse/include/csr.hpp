#include <iostream>
#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusparse.h>
#include <assert.h>

typedef enum
{
    host,
    device,
    shared
} memType;

template <typename fp>
class csrMat
{
public:
    int numA{};
    int mA{};
    int nA{};
    int *csrRowPtrA = nullptr;
    int *csrColIndA = nullptr;
    fp *csrValA = nullptr;
    memType tpA;

    csrMat(){};

    csrMat(int num, int m, int n, memType tp)
    {
        init(num, m, n, tp);
    }

    void init(int num, int m, int n, memType tp)
    {
        numA = num;
        mA = m;
        nA = n;
        tpA = tp;
        if (tpA == memType::host)
        {
            csrRowPtrA = new int[m + 1]();
            csrColIndA = new int[numA]();
            csrValA = new fp[numA]();
        }
        else if (tpA == memType::device)
        {
            cudaMalloc((void **)&csrRowPtrA, (m + 1) * sizeof(int));
            cudaMalloc((void **)&csrColIndA, numA * sizeof(int));
            cudaMalloc((void **)&csrValA, numA * sizeof(fp));
        }
    }

    void dense2csrHost(fp *A)
    {
        assert(tpA == memType::host);
        int num = 0;
        for (int i = 0; i < mA; i++)
        {
            csrRowPtrA[i] = num;
            for (int j = 0; j < nA; j++)
            {
                if (std::fabs(A[j * mA + i]) > 0.0)
                {
                    csrColIndA[num] = j;
                    csrValA[num++] = A[j * mA + i];
                }
            }
        }
        csrRowPtrA[mA] = num;
    }

    void copyFromHost(const csrMat &csrA_h)
    {
        assert(mA == csrA_h.mA);
        assert(numA == csrA_h.numA);
        assert(tpA == memType::device);
        cudaMemcpy(csrRowPtrA, csrA_h.csrRowPtrA, (mA + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(csrColIndA, csrA_h.csrColIndA, numA * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(csrValA, csrA_h.csrValA, numA * sizeof(fp), cudaMemcpyHostToDevice);
    }

    void printCsrFormHost()
    {
        std::cout << "CSR Form: \n";
        std::cout << "csrRowPtr: ";
        for (int i = 0; i <= mA; i++)
        {
            std::cout << csrRowPtrA[i] << " ";
        }
        std::cout << "\n";
        std::cout << "csrColInd: ";
        for (int i = 0; i < numA; i++)
        {
            std::cout << csrColIndA[i] << " ";
        }
        std::cout << "\n";
        std::cout << "csrVal:    ";
        for (int i = 0; i < numA; i++)
        {
            std::cout << csrValA[i] << " ";
        }
        std::cout << "\n";
    }

    void printCsrMatrixHost()
    {
        std::cout << "Matrix Form: \n";
        int num = 0;
        for (int i = 0; i < mA; i++)
        {
            // std::cout << "row " << i << ": ";
            std::cout << std::fixed;
            for (int j = 0; j < nA; j++)
                std::cout << (csrColIndA[num] == j ? csrValA[num++] : 0) << " ";
            std::cout << "\n";
        }
    }

    ~csrMat()
    {
        if (tpA == memType::host)
        {
            delete[] csrRowPtrA;
            delete[] csrColIndA;
            delete[] csrValA;
        }
        else if (tpA == memType::device)
        {
            cudaFree(csrRowPtrA);
            cudaFree(csrColIndA);
            cudaFree(csrValA);
        }
        // std::cout << "Call ~csrMat()" << std::endl;
    }
};