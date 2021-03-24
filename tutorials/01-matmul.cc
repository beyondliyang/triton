#include "triton/driver/backend.h"
#include "triton/driver/stream.h"
#include "triton/external/half.hpp"
#include "triton/runtime/function.h"
#include "triton/tools/bench.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <tuple>

namespace drv = triton::driver;
namespace rt = triton::runtime;

namespace src {

const char *dot =
    R"(
#define MAX_GROUP_SIZE 8

__global__ void dot(TYPE* A, TYPE* B, TYPE* C, 
                   int M, int N, int K, 
                   int lda, int ldb, int ldc) {
  int pid = get_program_id(0);
  int grid_m = (M + MB - 1) / MB;
  int grid_n = (N + NB - 1) / NB;
  int width = MAX_GROUP_SIZE * grid_n;
  int group_id = pid / width;
  int group_size = min(grid_m - group_id * MAX_GROUP_SIZE, MAX_GROUP_SIZE);
  int pid_m = group_id * MAX_GROUP_SIZE + (pid % group_size);
  int pid_n = (pid % width) / (group_size);
  int rm[MB] = pid_m * MB + 0 ... MB;
  int rn[NB] = pid_n * NB + 0 ... NB;
  int rk[KB] = 0 ... KB;
  TYPE *pa[MB, KB] = A + (rk [newaxis, :] * 1 + rm[:, newaxis] * lda);
  TYPE *pb[KB, NB] = B + (rk[:, newaxis] * ldb + rn [newaxis, :] * 1);
  float acc[MB, NB] = 0;
  for (int k = K; k > 0; k -= KB) {
    acc += (*pa) @ (*pb);
    pa += KB * 1;
    pb += KB * ldb;
  }
  rm = pid_m * MB + 0 ... MB;
  rn = pid_n * NB + 0 ... NB;
  TYPE *pc[MB, NB] = C + (rm[:, newaxis] * ldc + rn[newaxis, :]);
  *? (rm[:, newaxis] < M && rn [newaxis, :] < N) pc = acc;
}
)";
}

enum dtype_t {
  FLOAT,
  HALF,
  DOUBLE
};

template <class T>
struct to_string;

template <>
struct to_string<half_float::half> {
  static constexpr const char *value = "half";
};

template <>
struct to_string<float> {
  static constexpr const char *value = "float";
};

template <>
struct to_string<double> {
  static constexpr const char *value = "double";
};

template <class T>
float triton_dot(drv::context *context, drv::stream *stream,
                 bool AT, bool BT,
                 int32_t M, int32_t N, int32_t K) {
  std::string ty = to_string<T>::value;
  size_t dt_nbytes = sizeof(T);
  drv::device *device = context->device();
  int32_t lda = AT ? K : M;
  int32_t ldb = BT ? N : K;
  int32_t ldc = N;
  std::vector<std::string> sa = {"1", "lda"};
  std::vector<std::string> sb = {"1", "ldb"};
  // inputs
  auto dc = std::shared_ptr<drv::buffer>(drv::buffer::create(context, M * N * dt_nbytes));
  auto da = std::shared_ptr<drv::buffer>(drv::buffer::create(context, M * K * dt_nbytes));
  auto db = std::shared_ptr<drv::buffer>(drv::buffer::create(context, K * N * dt_nbytes));
  auto dlocks = std::shared_ptr<drv::buffer>(drv::buffer::create(context, 1024 * 1024 * 2 * 4));
  // initialize buffers
  std::vector<T> hc(M * N);
  std::vector<T> ha(M * K);
  std::vector<T> hb(K * N);
  for (size_t i = 0; i < ha.size(); i++)
    ha[i] = (float)rand() / RAND_MAX;
  for (size_t i = 0; i < hb.size(); i++)
    hb[i] = (float)rand() / RAND_MAX;
  stream->write(&*da, true, 0, ha);
  stream->write(&*db, true, 0, hb);
  // macros
  rt::options_t opt;
  opt.defines["STRIDE_AK"] = AT ? "1" : "lda";
  opt.defines["STRIDE_AM"] = AT ? "lda" : "1";
  opt.defines["STRIDE_BK"] = BT ? "ldb" : "1";
  opt.defines["STRIDE_BN"] = BT ? "1" : "ldb";
  opt.defines["TYPE"] = ty;
  opt.defines["MB"] = "128";
  opt.defines["NB"] = "128";
  opt.defines["KB"] = "32";
  opt.num_warps = 4;
  // arguments
  std::stringstream oss;
  rt::add_arg(oss, *da->cu());
  rt::add_arg(oss, *db->cu());
  rt::add_arg(oss, *dc->cu());
  rt::add_arg(oss, M);
  rt::add_arg(oss, N);
  rt::add_arg(oss, K);
  rt::add_arg(oss, lda);
  rt::add_arg(oss, ldb);
  rt::add_arg(oss, ldc);
  // function
  rt::function function(src::dot, opt, device);
  //  std::cout << function.get_kernels()[0].second->get_asm(rt::ASM_NV_PTX) << std::endl;
  // grid
  auto ceil = [](size_t x, size_t y) { return (x + y - 1) / y; };
  auto grid = [ceil, M, N](const rt::options_t &x) {
    return rt::kernel::grid_t{ceil(M, x.D<int>("MB")) *
                                  ceil(N, x.D<int>("NB")),
                              1};
  };

  // metrics
  auto tflops = [&](double nanosec) { return 2. * M * N * K / nanosec * 1e-3; };
  double triton_ns = triton::tools::bench([&]() { function(oss.str(), grid, stream); }, stream);
  return tflops(triton_ns);
}

float bench_dot(drv::context *context, drv::stream *stream,
                bool AT, bool BT,
                int32_t M, int32_t N, int32_t K,
                dtype_t dtype) {
  switch (dtype) {
  case HALF:
    return triton_dot<half_float::half>(context, stream, AT, BT, M, N, K);
  case FLOAT:
    return triton_dot<float>(context, stream, AT, BT, M, N, K);
  case DOUBLE:
    return triton_dot<double>(context, stream, AT, BT, M, N, K);
  default:
    return 0;
  }
}

int main() {
  // initialize default compute device
  auto context = triton::driver::backend::contexts::get_default();
  triton::driver::stream *stream = triton::driver::stream::create(context->backend());
  // shapes to benchmark
  typedef std::tuple<bool, bool, int, int, int> config_t;
  std::vector<config_t> configs = {
      {false, false, 8192, 8192, 8192}};
  // does the work
  bool AT, BT;
  int32_t M, N, K;
  dtype_t dtype = HALF;
  for (const auto &c : configs) {
    std::tie(AT, BT, M, N, K) = c;
    float tflops = bench_dot(context, stream, AT, BT, M, N, K, dtype);
    std::cout << "// " << AT << ", " << BT << ", " << M << ", " << N << ", " << K << ", " << tflops << std::endl;
  }
}