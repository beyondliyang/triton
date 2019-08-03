/* Copyright 2015-2017 Philippe Tillet
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files
* (the "Software"), to deal in the Software without restriction,
* including without limitation the rights to use, copy, modify, merge,
* publish, distribute, sublicense, and/or sell copies of the Software,
* and to permit persons to whom the Software is furnished to do so,
* subject to the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef TDL_INCLUDE_DNN_BASE_H
#define TDL_INCLUDE_DNN_BASE_H

#include "triton/driver/stream.h"
#include "triton/driver/kernel.h"
#include "triton/runtime/launch_info.h"

namespace triton{

namespace runtime{
  class jit;
}

namespace dnn{


enum autotuning_t{
  FULL_TUNING,
  PARTIAL_TUNING,
  NO_TUNING
};

class base;
struct launch_context_t{
  base *op;
  driver::kernel* kernel;
  triton::runtime::launch_information info;
};

typedef std::vector<unsigned> params_t;

class base {
  friend class recompile_hash;
  friend class recompile_equal;

protected:
  // leading dimensions
  static void set_ld(const std::vector<int32_t>& shapes,
                     std::vector<int32_t>& ld);
  // list of retuning parameters
  virtual std::vector<int64_t> retune_params() const = 0;

private:
  // initialize
  virtual void init_impl(driver::stream *, driver::cu_module *, triton::runtime::launch_information) = 0;
  // deinitialize
  virtual void deinit_impl() = 0;
  // enqueue
  virtual void enqueue_impl(driver::stream *stream, driver::kernel *kernel,
                    std::vector<driver::buffer*> args,
                    triton::runtime::launch_information info) = 0;
  // number of flops
  virtual size_t num_flops() const = 0;
  // default parameters
  virtual std::vector<params_t> search_space() const;
  virtual params_t heuristics() const;
  // obtain execution jit
  std::pair<base*, triton::runtime::jit*> get_profile_impl(driver::stream *stream, std::vector<driver::buffer *> args, autotuning_t autotune);

public:
  // constructor
  base(const std::string& name);
  // triton-c source
  virtual void triton_c_src(std::ostream &os) const = 0;
  // clone
  virtual base* clone() const = 0;
  // enqueue
  base* enqueue(driver::stream* stream, std::vector<driver::buffer*> args, autotuning_t autotune = PARTIAL_TUNING);
  // get profile
  launch_context_t get_launch_context(driver::stream *stream, std::vector<driver::buffer *> args, autotuning_t autotune = PARTIAL_TUNING);

private:
  std::string name_;
};


struct recompile_equal{
  bool operator()(base* x, base* y) const{
    return typeid(*x) == typeid(*y) &&
           x->retune_params() == y->retune_params();
  }
};

struct recompile_hash{
  unsigned operator()(base* x) const{
    return x->retune_params()[0];
  }
};


}
}

#endif
