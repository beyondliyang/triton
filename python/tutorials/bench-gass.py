import torch
import triton

autotune_configs = [
    triton.config(defines={"MB": "128", "NB": "128", "KB": "16"}, num_warps=4),
    # triton.config(defines={'MB': '64', 'NB': '128', 'KB': '32'}, num_warps=4),
    # triton.config(defines={'MB': '128', 'NB': '64', 'KB': '32'}, num_warps=4),
    # triton.config(defines={'MB': '64', 'NB': '64', 'KB': '64'}, num_warps=4),
    # triton.config(defines={'MB': '32', 'NB': '128', 'KB': '64'}, num_warps=4),
    # triton.config(defines={'MB': '128', 'NB': '32', 'KB': '64'}, num_warps=4),
    # triton.config(defines={'MB': '64', 'NB': '32', 'KB': '64'}, num_warps=2),
    # triton.config(defines={'MB': '32', 'NB': '64', 'KB': '64'}, num_warps=2)
]

autotune_key = ["M", "N", "K"]

src = """
#define MAX_GROUP_SIZE 8

__global__ void dot(TYPE* A, TYPE* B, TYPE* C, 
                   int M, int N, int K, 
                   int lda, int ldb, int ldc) {
  int pid_m = get_program_id(0);
  int pid_n = get_program_id(1);
  int rm[MB] = pid_m * MB + 0 ... MB;
  int rn[NB] = pid_n * NB + 0 ... NB;
  int rk[KB] = 0 ... KB;
  TYPE *pa[MB, KB] = A + (rk [newaxis, :] * 1 + rm[:, newaxis] * lda);
  TYPE *pb[KB, NB] = B + (rk[:, newaxis] * ldb + rn [newaxis, :] * 1);
  float acc[MB, NB] = 0;
  for (int k = K; k > 0; k -= KB) {
    acc += (*pa) @ (*pb);
    //pa += KB * 1;
    //pb += KB * ldb;
  }
  rm = pid_m * MB + 0 ... MB;
  rn = pid_n * NB + 0 ... NB;
  TYPE *pc[MB, NB] = C + (rm[:, newaxis] * ldc + rn[newaxis, :]);
  *pc = acc;
}
"""


def make_kernel(device, dtype):
    key = (device, dtype)
    cache = make_kernel.cache
    if key not in cache:
        defines = {'TYPE': dtype}
        cache[key] = triton.kernel(
            src,
            device=device,
            defines=defines,
            autotune_configs=autotune_configs,
            autotune_key=autotune_key,
            direct_sass=True,
        )
    return cache[key]


make_kernel.cache = dict()


class _dot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        M, Ka = a.shape
        Kb, N = b.shape
        assert Ka == Kb, "incompatible dimensions"
        assert a.is_contiguous() and b.is_contiguous(), "inputs must be contiguous"
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
        kernel = make_kernel(a.device, a.dtype)
        grid = lambda opt: (
            triton.cdiv(M, opt.MB),
            triton.cdiv(N, opt.NB),
        )
        kernel(a.data_ptr(), b.data_ptr(), c.data_ptr(), \
               M, N, Ka, \
               a.stride(0), b.stride(0), c.stride(0), \
               grid=grid)
        return c


dot = _dot.apply

a = torch.rand((1024, 1024), device='cuda', dtype=torch.float16)
b = torch.rand((1024, 1024), device='cuda', dtype=torch.float16)
c_0 = dot(a, b)
c_1 = torch.matmul(a, b)
print(c_0)
print(c_1)
print(torch.allclose(c_0, c_1, rtol=1e-3, atol=1e-3))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],  # argument names to use as an x-axis for the plot
        x_vals=[8192],  # different possible values for `x_name`
        y_name='provider',  # argument name whose value corresponds to a different line in the plot
        y_vals=['torch', 'triton'],  # possible keys for `y_name`
        y_lines=["Torch", "Triton"],  # label name for the lines
        ylabel="TFLOPS",  # label name for the y-axis
        plot_name="matmul-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={}
    )
)
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b))
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: dot(a, b))
    if provider == 'cutlass':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton.testing.cutlass_matmul(a, b))
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    print(provider, perf(ms), ms)
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(show_plots=True)
