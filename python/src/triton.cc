﻿#include "triton/driver/stream.h"
#include "triton/runtime/function.h"
#include <pybind11/buffer_info.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <regex>
#include <string>

namespace py = pybind11;

using namespace triton;
namespace rt = triton::runtime;
namespace drv = triton::driver;

/*****************************************************************************/
/* Python bindings for triton::tools                                         */
/*****************************************************************************/

/*!
  @brief Function for extracting kernels out of a given source-string

  This can be important to enable pre-processor macros (or tunable parameters) that should only
  be defined within the scope of a single kernel function
*/
std::string extract_kernels(const std::string &str, const std::vector<std::string> &names) {
  if (names.empty())
    return str;
  // search for all regex matches of kernel_regex in str
  std::smatch matches;
  std::regex regex(" *__global__ +void +([_a-zA-Z][_a-zA-Z0-9]{0,30})");
  std::sregex_iterator it(str.begin(), str.end(), regex);
  std::sregex_iterator end;
  std::vector<std::tuple<std::string, int, int>> kernels;
  for (; it != end; ++it) {
    int pos = it->position();
    int len = it->length();
    std::string name = it->str(1);
    kernels.push_back(std::make_tuple(name, pos, len));
  }
  // check that all the kernels provided actually exist
  for (const std::string &name : names) {
    auto pred = [&name](const std::tuple<std::string, int, int> &t) { return std::get<0>(t) == name; };
    bool found = std::any_of(kernels.begin(), kernels.end(), pred);
    if (!found)
      throw std::runtime_error("Unable to find kernel `" + name + "` in provided source code:\n" + str);
  }
  // simple parsing logic to extract the declaration and body of each specified kernel
  std::string ret;
  for (const auto &k : kernels) {
    std::string name;
    int pos, len;
    std::tie(name, pos, len) = k;
    if (std::find(names.begin(), names.end(), name) == names.end())
      continue;
    std::string def = str.substr(pos, str.size() - pos);
    // skip over declaration
    // by finding matching ')' for first '('
    int count = 1;
    pos = def.find('(');
    while (!(def[pos++] == ')' && count == 0) && pos < def.size()) {
      count += def[pos] == '(';
      count -= def[pos] == ')';
    }
    // skip over definition
    // by finding matching '{' for first '}'
    count = 1;
    pos = def.find('{', pos);
    while (!(def[pos++] == '}' && count == 0) && pos < def.size()) {
      count += def[pos] == '{';
      count -= def[pos] == '}';
    }
    ret += def.substr(0, pos);
    ret += '\n';
  }
  return ret;
}

void init_triton_tools(py::module &&m) {
  m.def("extract_kernels", &extract_kernels);
}

/*****************************************************************************/
/* Python bindings for triton::driver                                        */
/*****************************************************************************/

void init_triton_driver(py::module &&m) {
  // base device
  py::class_<drv::device>(m, "device");
  // cuda device
  py::class_<drv::cu_device, driver::device>(m, "cu_device")
      .def(py::init([](int dev_id, bool take_ownership) {
        CUdevice handle;
        drv::dispatch::cuDeviceGet(&handle, dev_id);
        return new drv::cu_device(handle, take_ownership);
      }));
  // host device
  py::class_<drv::host_device, driver::device>(m, "host_device")
      .def(py::init<>());

  // base stream
  py::class_<drv::stream>(m, "stream");
  // host stream
  py::class_<drv::host_stream, drv::stream>(m, "host_stream")
      .def(py::init<>());
  // cuda stream
  py::class_<drv::cu_stream, drv::stream>(m, "cu_stream")
      // py doesn't support opaque pointer (e.g., CUstream) so
      // we assume it has been converted to uint64_t
      .def(py::init([](uint64_t handle, bool take_ownership) {
        return std::unique_ptr<driver::cu_stream>(new driver::cu_stream((CUstream)handle, take_ownership));
      }));
}

/*****************************************************************************/
/* Python bindings for triton::runtime                                       */
/*****************************************************************************/
void init_triton_runtime(py::module &&m) {
  // argument type
  py::enum_<rt::arg_type>(m, "arg_type")
      .value("int1", rt::INT1_T)
      .value("int8", rt::INT8_T)
      .value("int16", rt::INT16_T)
      .value("int32", rt::INT32_T)
      .value("int64", rt::INT64_T)
      .value("half", rt::HALF_T)
      .value("float", rt::FLOAT_T)
      .value("double", rt::DOUBLE_T)
      .value("buffer", rt::BUFFER_T);
  // compilation options
  py::class_<rt::options_t>(m, "options", py::dynamic_attr())
      .def(py::init<>())
      .def_readwrite("defines", &rt::options_t::defines)
      .def_readwrite("num_warps", &rt::options_t::num_warps)
      .def("__getattr__", [](rt::options_t *opt, const std::string &name) {
        return opt->D<int>(name);
      });
  //  kernel
  py::class_<rt::kernel>(m, "kernel")
      .def("__call__", &rt::kernel::operator())
      .def_readonly("opt", &rt::kernel::opt)
      .def("asm", &rt::kernel::get_asm);
  // tune conf
  py::class_<rt::config>(m, "config")
      .def(py::init<std::map<std::string, std::string>, int>(),
           py::arg("defines") = std::map<std::string, std::string>(),
           py::arg("num_warps"));

  // function
  py::class_<rt::function>(m, "function")
      .def(py::init<const std::string &, const rt::options_t &, driver::device *, const std::vector<rt::config> &, const std::vector<std::string> &>())
      .def("autotune", &rt::function::autotune, py::return_value_policy::reference_internal)
      .def("signature", &rt::function::get_signature);
}

void init_triton(py::module &m) {
  py::module subm = m.def_submodule("triton");
  init_triton_driver(std::move(subm.def_submodule("driver")));
  init_triton_runtime(std::move(subm.def_submodule("runtime")));
  init_triton_tools(std::move(subm.def_submodule("tools")));
}
