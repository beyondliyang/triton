import triton._C.libtriton as libtriton
import os
import time
from struct import pack
import torch

codes = {
  libtriton.arg_type.int1:   'B',
  libtriton.arg_type.int8:   'B',
  libtriton.arg_type.int32:  'I',
  libtriton.arg_type.int64:  'Q',
  libtriton.arg_type.half:   'H',
  libtriton.arg_type.float:  'f',
  libtriton.arg_type.double: 'd',
  libtriton.arg_type.buffer: 'P'
}

def th_to_triton(obj):
  tys = {
    torch.int8: 'char',
    torch.int16: 'short',
    torch.int32: 'int',
    torch.int64: 'long',
    torch.float16: 'half',
    torch.float32: 'float',
    torch.float64: 'double'
  }
  if isinstance(obj, torch.dtype):
    return [tys[obj]]
  if isinstance(obj, list):
    return [th_to_triton(x)[0] for x in obj]
  return [str(obj)]

def cdiv(a, b):
    return libtriton.cdiv(a, b)

def largest_pow2_divisor(a):
    return libtriton.largest_pow2_divisor(a)

def cdiv_sum(a, b):
    return libtriton.cdiv_sum(a, b)
  
def synchronize(device):
    dev_id = device.index
    dev_id = -1 if dev_id is None else dev_id
    libtriton.synchronize(dev_id)

class kernel:

  def __init__(self, src, device, defines = dict(), num_warps = [2, 4, 8]):
    self.src = src
    self.opt = libtriton.options_space()
    self.opt.defines = [(k, th_to_triton(v)) for k, v in defines.items()]
    self.opt.num_warps = num_warps
    self.device = -1 if device.index is None else device.index
    self.op_id = libtriton.make_op_id()
    libtriton.register_fn(self.op_id, self.device, self.src, self.opt)
    # debug mode
    self.is_debug = 'TRITON_DEBUG' in os.environ
    # signature
    arg_types = libtriton.get_fn_signature(self.op_id)
    self.tys = ''.join([codes[x] for x in arg_types])
    self.buf_idx = [i for i, x in enumerate(self.tys) if x == 'P']

  def __call__(self, *args, grid):
    if self.is_debug:
      _args = args
      args = [x.clone() if isinstance(x, torch.Tensor) else x for x in _args]
      for i in range(len(args)):
        if isinstance(args[i], torch.Tensor):
          args[i] = libtriton.cuda_empty_like(args[i])
          args[i].copy_(_args[i])
    libtriton.cuda_set_device(self.device)
    params = pack(self.tys, *args)
    opt = libtriton.autotune(self.op_id, self.device, params, grid)
    grid = grid(opt)
    libtriton.launch_kernel(self.op_id, self.device, params, grid[0], grid[1], grid[2])
    if self.is_debug:
      for i in range(len(args)):
        if isinstance(args[i], torch.Tensor):
          _args[i].copy_(args[i].clone())
      args = _args