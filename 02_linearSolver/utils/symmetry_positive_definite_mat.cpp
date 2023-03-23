// https://cplusplus.com/forum/general/257711/
// Generate Hermitian positive definite matrix
#include <iostream>
#include <iomanip>
#include <vector>
#include <complex>
#include <random>
#include <ctime>
using namespace std;

using vec = vector<complex<double>>;
using matrix = vector<vec>;

mt19937 gen(time(0));
normal_distribution<double> dist(0.0, 1.0);

//======================================================================

void print(const matrix &M, int width)
{
    for (auto &row : M)
    {
        for (auto z : row)
            cout << setw(width) << z << "  ";
        cout << '\n';
    }
}

//======================================================================

matrix genNormal(int rows, int cols)
{
    const double sq = sqrt(2.0);
    matrix result(rows, vec(cols));

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
            result[i][j] = complex<double>(dist(gen), dist(gen)) / sq;
    }
    return result;
}

//======================================================================

matrix Hermitian(const matrix &A)
{
    int N = A.size();
    int nk = A[0].size();
    matrix result(N, vec(N));
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            result[i][j] = 0.0;
            for (int k = 0; k < nk; k++)
                result[i][j] += A[i][k] * conj(A[j][k]);
        }
    }
    return result;
}

//======================================================================

int main()
{
    const int K = 4, M = 16;

    matrix H = genNormal(K, M);
    matrix Z = Hermitian(H);
    print(Z, 20);
}