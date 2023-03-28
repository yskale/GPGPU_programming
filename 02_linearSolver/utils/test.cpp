static inline fp_rtype fp_real(fp_type x) { return x; }
static inline fp_rtype fp_imag(fp_type x) { return 0.0; }
static inline fp_type fp_cmplx(fp_rtype x) { return x; }
static inline fp_type fp_conj(fp_type x) { return x; }
static inline fp_rtype fp_abs2(fp_type x) { return x * x; }
static inline fp_rtype fp_abs(fp_type x) { return fabs(x); }
static inline fp_rtype fp_dabs(fp_type x) { return fabs(x); }
static inline fp_type fp_mul(fp_type x, fp_type y) { return x * y; }
static inline fp_type fp_scal(fp_rtype x, fp_type y) { return x * y; }
static inline fp_type fp_div(fp_type x, fp_type y) { return x / y; }
static inline fp_type fp_inv(fp_type x) { return 1.0 / x; }
static inline int fp_is_zero(fp_type x) { return (x == 0.0); }
static inline fp_rtype fp_eps()
{
    unsigned char eps_b[] = {0, 0, 0x80, 0x33};
    return *((fp_rtype *)eps_b);
}
static inline fp_rtype fp_sfmin()
{
    unsigned char sfmin_b[] = {0, 0, 0x80, 0};
    return *((fp_rtype *)sfmin_b);
}
static inline fp_rtype fp_prec()
{
    unsigned char prec_b[] = {0, 0, 0, 0x34};
    return *((fp_rtype *)prec_b);
}
static inline fp_rtype fp_small()
{
    unsigned char small_b[] = {0, 0, 0x80, 0x2a};
    return *((fp_rtype *)small_b);
}
static inline fp_type fp_bcast(fp_type x, uint y) { return sub_group_broadcast(x, y); }
static inline fp_type fp_shuffle(fp_type x)
{
    fp_type y;
    uint i = get_sub_group_local_id();
    y = x;
    return y;
}
static inline fp_type fp_unshuffle(fp_type x)
{
    fp_type y;
    uint i = get_sub_group_local_id();
    y = x;
    return y;
}
static inline fp_type fp_block_read(fp_type *x) { return fp_shuffle(as_type(intel_sub_group_block_read((__global uint *)x))); }
static inline void fp_block_write(__global fp_type *x, fp_type y)
{
    y = fp_unshuffle(y);
    intel_sub_group_block_write((__global uint *)x, as_uint(y));
}
static inline void nrm2(fp_rtype *res, long n, fp_type *x, long incx)
{
    long i;
    fp_rtype _res = 0.0;
    for (i = 0; i < n; i++)
    {
        _res += fp_abs2(x[i]);
    }
    *res = sqrt(_res);
}
__kernel void geqr2_cl(long m, long n, __global fp_type *a, long a_off, long lda, __global fp_type *tau, long tau_off, __global fp_type *work, long work_off, __global long *info, long info_off)
{
    a += a_off / sizeof(fp_type);
    tau += tau_off / sizeof(fp_type);
    work += work_off / sizeof(fp_type);
    info += info_off / sizeof(long);
    unsigned long id = get_local_id(0);
    unsigned long ws = get_local_size(0);
    long i, j, k, minmn = min(m, n);
    fp_rtype beta, xnorm;
    fp_type taui, alpha;
    for (k = 0; k < minmn; k++)
    {
        if (id == 0)
        {
            alpha = a[k + k * lda];
            nrm2(&xnorm, m - k - 1, &a[k + 1 + k * lda], 1);
            if (xnorm == 0.0 && fp_imag(alpha) == 0.0)
            {
                tau[k] = fp_cmplx(0.0);
            }
            else
            {
                beta = sqrt(xnorm * xnorm + fp_abs2(alpha));
                if (fp_real(alpha) > 0.0)
                    beta = -beta;
                tau[k] = (fp_cmplx(beta) - alpha) / beta;
                alpha = fp_inv(alpha - fp_cmplx(beta));
                for (i = k + 1; i < m; i++)
                    a[i + k * lda] = fp_mul(alpha, a[i + k * lda]);
                a[k + k * lda] = fp_cmplx(1.0);
            }
        }
        work_group_barrier(CLK_GLOBAL_MEM_FENCE);
        taui = tau[k];
        if (!fp_is_zero(taui))
        {
            long jstart = id + (k + 1) / ws * ws;
            if (id < (k + 1) % ws)
                jstart += ws;
            for (j = jstart; j < n; j += ws)
            {
                fp_type w = fp_cmplx(0.0);
                for (i = k; i < m; i++)
                    w += fp_mul(fp_conj(a[i + j * lda]), a[i + k * lda]);
                w = fp_mul(fp_conj(taui), fp_conj(w));
                for (i = k; i < m; i++)
                    a[i + j * lda] -= fp_mul(w, a[i + k * lda]);
            }
            work_group_barrier(CLK_GLOBAL_MEM_FENCE);
            if (id == 0)
                a[k + k * lda] = fp_cmplx(beta);
        }
    }
}