import torch
import triton
import struct

class _matmul(torch.autograd.Function):
    src = """
#define STM 8
#define STN 8

__global__ void matmul(TYPE * A __noalias __readonly __aligned(16),
                       TYPE * B __noalias __readonly __aligned(16),
                       TYPE * C __noalias __aligned(16),
                       float alpha,
                       int M,
                       int N,
                       int K __multipleof(16),
                       int lda __multipleof(LDA_POW2_DIV),
                       int ldb __multipleof(LDB_POW2_DIV),
                       int ldc __multipleof(LDC_POW2_DIV),
                       int* locks) {
      // prologue
      int pid = get_program_id(0);
      int pidz = get_program_id(2);
      int gridm = (M + TM - 1) / TM;
      int gridn = (N + TN - 1) / TN;

      // swizzle for better L2 performance
      int width = STM*gridn;
      int stm = pid / width;
      int RSTM  = min(gridm - stm*STM, STM);
      int stn =  (pid % width) / (RSTM*STN);
      int RSTN = min(gridn - stn*STN, STN);
      int laneid = pid % (RSTM * RSTN);
      int lanem = laneid / RSTN;
      int lanen = laneid % RSTN;
      int pidm = stm*STM + lanem;
      int pidn = stn*STN + lanen;
      int rm[TM] = pidm * TM + 0 ... TM;
      int rn[TN] = pidn * TN + 0 ... TN;

      // split-k for better parrelism
      K           = K / TZ;
      int rk[TK]  = 0 ... TK;
      // pointers to operands
      int offa[TM, TK] = (pidz*K + rk[newaxis, :]) * STRIDE_AK + rm[:, newaxis] * STRIDE_AM;
      int offb[TK, TN] = (pidz*K + rk[:, newaxis]) * STRIDE_BK + rn[newaxis, :] * STRIDE_BN;
      TYPE* pa[TM, TK] = A + offa;
      TYPE* pb[TK, TN] = B + offb;

      // prefetches operands
      bool checka[TM, TK] = rk[newaxis, :] < K;
      bool checkb[TK, TN] = rk[:, newaxis] < K;
      TYPE a[TM, TK] = checka ? *pa : 0;
      TYPE b[TK, TN] = checkb ? *pb : 0;
      pa += TK * STRIDE_AK;
      pb += TK * STRIDE_BK;

      // reduction loop
      float acc[TM, TN] = 0;
      for(int k = K; k > 0; k -= TK){
#ifdef K_MULTIPLE_OF_TK
        bool checkk[TK] = k > TK;
#else
        bool checkk[TK] = rk < k - TK;
#endif
        bool checka[TM, TK] = checkk[newaxis, :];
        bool checkb[TK, TN] = checkk[:, newaxis];
        acc += a @ b;
#ifdef K_MULTIPLE_OF_TK
        a = *?(checka)pa;
        b = *?(checkb)pb;
#else
        a = checka ? *pa : 0;
        b = checkb ? *pb : 0;
#endif
        pa += TK * STRIDE_AK;
        pb += TK * STRIDE_BK;
      }
      acc = acc * alpha;
      TYPE c[TM, TN] = acc;

      // epilogue
      int rcm[TM] = pidm * TM + 0 ... TM;
      int rcn[TN] = pidn * TN + 0 ... TN;
      int offc[TM, TN] = rcm[:, newaxis] * ldc + rcn[newaxis, :];
      TYPE* pc[TM, TN] = C + offc;
      bool checkc[TM, TN] = rcm[:, newaxis] < M && rcn[newaxis, :] < N;
#if (TZ==1)
      *?(checkc) pc = c;
#else
      // accumulate partial result using spin-locks
      int *plock  = locks + rid;
      int *pcount = plock + get_num_programs(0) * get_num_programs(1);
      for(int repeat = 1; repeat == 1; repeat = atomic_cas(plock, 0, 1));
      int count = *pcount;
      if(count == 0)
        *?(checkc) pc = c;
      else
        *?(checkc) pc = c + *?(checkc)pc;
      atomic_xchg(pcount, (count + 1) % TZ);
      atomic_xchg(plock, 0);
#endif
}
    """
    TM = 128
    TN = 128
    TK = 32
    TZ = 1
    num_warps = [4]
    kernel = dict()

    @staticmethod
    def largest_pow2_divisor(N):
        if N % 8 == 0: return 8
        if N % 4 == 0: return 4
        if N % 2 == 0: return 2
        return 1

        
    _locks = dict()
    @staticmethod
    def _call(a, b):
        dtype = a.dtype
        device = a.device
        # allocate output
        M, K = a.shape
        K, N = b.shape
        c = torch.empty((M, N), dtype=dtype, device=device)
        # handle non-contiguous inputs if necessary
        if a.stride(0) > 1 and a.stride(1) > 1: a = a.contiguous()
        if b.stride(0) > 1 and b.stride(1) > 1: b = b.contiguous()
        # kernel hash
        is_a_row = a.stride(1) == 1
        is_b_row = b.stride(1) == 1
        lda = a.stride(0) if is_a_row else a.stride(1)
        ldb = b.stride(0) if is_b_row else b.stride(1)
        ldc = c.stride(0)
        lda_pow2_div = _matmul.largest_pow2_divisor(lda)
        ldb_pow2_div = _matmul.largest_pow2_divisor(ldb)
        ldc_pow2_div = _matmul.largest_pow2_divisor(ldc)
        m_k_tk      = K % 32 == 0
        key = (device, dtype, is_a_row, is_b_row, lda_pow2_div, ldb_pow2_div, ldc_pow2_div, m_k_tk)
        if key not in _matmul.kernel:
            defines = {
                'TYPE' : dtype,
                'STRIDE_AM'   : 'lda' if is_a_row else '1', 
                'STRIDE_AK'   : '1'   if is_a_row else 'lda',
                'STRIDE_BK'   : 'ldb' if is_b_row else '1',
                'STRIDE_BN'   : '1'   if is_b_row else 'ldb',
                'LDA_POW2_DIV': lda_pow2_div,
                'LDB_POW2_DIV': ldb_pow2_div,
                'LDC_POW2_DIV': ldc_pow2_div,
                'TM'          : _matmul.TM,
                'TN'          : _matmul.TN,
                'TK'          : _matmul.TK,
                'TZ'          : _matmul.TZ
            }
            if m_k_tk:
                defines['K_MULTIPLE_OF_TK'] = '1'
            _matmul.kernel[key] = triton.kernel(_matmul.src, device, num_warps=_matmul.num_warps, defines=defines)
        kernel = _matmul.kernel[key]
        # # locks for split-k
        if device not in _matmul._locks:
          _matmul._locks[device] = torch.zeros(1024*1024, dtype=torch.int32, device=device)
        locks = _matmul._locks[device]
        # enqueue
        alpha = 1.
        args = [a.data_ptr(), b.data_ptr(), c.data_ptr(), alpha, M, N, K, lda, ldb, ldc, locks.data_ptr()]
        grid = lambda opt: [triton.cdiv(M, opt.TM) * triton.cdiv(N, opt.TN), 1, 1]
        kernel(*args, grid=grid)
        return c

    @staticmethod
    def forward(ctx, a, b):
        c = _matmul._call(a,b)
        return c

matmul = _matmul.apply
