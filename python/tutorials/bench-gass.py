import torch
import triton
import os

autotune_key = ["M", "N", "K"]

src = """
#define STM 8
#define STN 8

__global__ void matmul(TYPE *A __noalias __readonly,
                       TYPE *B __noalias __readonly,
                       TYPE *C __noalias,
                       int M,
                       int N,
                       int K __multipleof(16),
                       int lda,
                       int ldb,
                       int ldc) {
  // prologue
  int pidm = get_program_id(0);
  int pidn = get_program_id(1);

  // (no) swizzle for better L2 performance
  int rm[TM] = pidm * TM + 0 ... TM;
  int rn[TN] = pidn * TN + 0 ... TN;

  int rk[TK] = 0 ... TK;
  // pointers to operands
  int offa[TM, TK] = rk[newaxis, :] * STRIDE_AK + rm[:, newaxis] * STRIDE_AM;
  int offb[TK, TN] = rk[:, newaxis] * STRIDE_BK + rn [newaxis, :] * STRIDE_BN;
  TYPE *pa[TM, TK] = A + offa;
  TYPE *pb[TK, TN] = B + offb;

  // prefetches operands
  bool checka[TM, TK] = rk[newaxis, :] < K;
  bool checkb[TK, TN] = rk[:, newaxis] < K;
  TYPE a[TM, TK] = checka ? *pa : 0;
  TYPE b[TK, TN] = checkb ? *pb : 0;
  pa += TK * STRIDE_AK;
  pb += TK * STRIDE_BK;

  // reduction loop
  float acc[TM, TN] = 0;
  for (int k = K; k > 0; k -= TK) {
    bool checkk[TK] = k > TK;
    bool checka[TM, TK] = checkk[newaxis, :];
    bool checkb[TK, TN] = checkk[:, newaxis];
    acc += a @ b;

    a = *? (checka)pa;
    b = *? (checkb)pb;

    pa += TK * STRIDE_AK;
    pb += TK * STRIDE_BK;
  }
  TYPE c[TM, TN] = acc;

  // epilogue
  int rcm[TM] = pidm * TM + 0 ... TM;
  int rcn[TN] = pidn * TN + 0 ... TN;
  int offc[TM, TN] = rcm[:, newaxis] * 8192 + rcn [newaxis, :];
  TYPE *pc[TM, TN] = C + offc;
  bool checkc[TM, TN] = rcm[:, newaxis] < M && rcn [newaxis, :] < N;
  // *? (checkc)pc = c;
  *pc = c;
}
"""

# configs that work:
'''
[({"TM": "64", "TN": "64", "TK": "32"}, 4),
 ({"TM": "64", "TN": "32", "TK": "8"}, 4),
 ({"TM": "64", "TN": "64", "TK": "32"}, 8),
 ({"TM": "32", "TN": "32", "TK": "8"}, 2),
 ({"TM": "16", "TN": "16", "TK": "8"}, 2),]
'''


class _matmul(torch.autograd.Function):

    _DEFAULT_CONFIGS = [
        # ({"TM": "64", "TN": "64", "TK": "32"}, 4),
        triton.config(defines={"TM": "128", "TN": "128", "TK": "16"}, num_warps=4),
        # ({"TM": "128", "TN": "128", "TK": "32"}, 4),
        # ({'TM': '64', 'TN': '128', 'TK': '32'}, 4),
        # ({'TM': '128', 'TN': '64', 'TK': '32'}, 4),
        # ({'TM': '32', 'TN': '32', 'TK': '8'}, 2),
        # ({'TM': '32', 'TN': '128', 'TK': '64'}, 4),
        # ({'TM': '128', 'TN': '32', 'TK': '64'}, 4),
        # ({'TM': '64', 'TN': '32', 'TK': '64'}, 2),
        # ({'TM': '32', 'TN': '64', 'TK': '64'}, 2),
    ]
    _CONFIGS = _DEFAULT_CONFIGS

    @staticmethod
    def largest_pow2_divisor(N):
        if N % 8 == 0:
            return 8
        if N % 4 == 0:
            return 4
        if N % 2 == 0:
            return 2
        return 1

    _locks = dict()
    _kernels = dict()

    @staticmethod
    def _call(a, b):
        dtype = a.dtype
        device = a.device
        # allocate output
        M, K = a.shape
        K, N = b.shape
        c = torch.zeros((M, N), dtype=dtype, device=device)
        # kernel hash
        is_a_row = a.stride(1) == 1
        is_b_row = b.stride(1) == 1
        lda = a.stride(0) if is_a_row else a.stride(1)
        ldb = b.stride(0) if is_b_row else b.stride(1)
        ldc = c.stride(0)
        lda_pow2_div = _matmul.largest_pow2_divisor(lda)
        ldb_pow2_div = _matmul.largest_pow2_divisor(ldb)
        ldc_pow2_div = _matmul.largest_pow2_divisor(ldc)
        is_tk_div_k = K % 64 == 0
        key = (device, dtype, is_a_row, is_b_row, lda_pow2_div, ldb_pow2_div, ldc_pow2_div, is_tk_div_k)
        if key not in _matmul._kernels:
            defines = {
                "TYPE": dtype,
                "STRIDE_AM": f"{lda}" if is_a_row else "1",
                "STRIDE_AK": "1" if is_a_row else f"{lda}",
                "STRIDE_BK": f"{ldb}" if is_b_row else "1",
                "STRIDE_BN": "1" if is_b_row else f"{ldb}",
                "LDA_POW2_DIV": lda_pow2_div,
                "LDB_POW2_DIV": ldb_pow2_div,
                "LDC_POW2_DIV": ldc_pow2_div,
            }
            _matmul._kernels[key] = triton.kernel(
                src,
                device=device,
                defines=defines,
                autotune_configs=_matmul._CONFIGS,
                autotune_key=["M", "N"],
                direct_sass=False,
            )
        kernel = _matmul._kernels[key]
        # enqueue
        args = [a.data_ptr(), b.data_ptr(), c.data_ptr(), M, N, K, lda, ldb, ldc]
        grid = lambda opt: [
            triton.cdiv(M, opt.TM),
            triton.cdiv(N, opt.TN),
            1,
        ]
        obj = kernel(*args, grid=grid)
        print(obj.asm('ptx'))
        print(obj.asm('llir'))
        return c

    @staticmethod
    def forward(ctx, a, b):
        c = _matmul._call(a, b)
        return c


matmul = _matmul.apply

M, N, K = 8192, 8192, 8192

torch.manual_seed(0)

a = torch.randn((M, K), device="cuda", dtype=torch.float16)
b = torch.randn((K, N), device="cuda", dtype=torch.float16)

th_c = torch.matmul(a, b)
tt_c = matmul(a, b)

print('********* matmul **********')
print(f'The maximum difference between torch and triton is ' f'{torch.max(torch.abs(th_c - tt_c))}')
# assert triton.testing.allclose(th_c, tt_c)
#print(triton.testing.do_bench(lambda: torch.matmul(a, b)))
#print(triton.testing.do_bench(lambda: matmul(a, b)))
exit()
#print(triton.testing.do_bench(lambda: triton.ops.matmul(a, b)))
# print(1)