===========================
Writing a Custom Operation
===========================

--------------
Compute Kernel
--------------

Let us start with something simple, and see how Triton can be used to create a custom vector addition for PyTorch. The Triton compute kernel for this operation is the following:

.. code-block:: C

    // Triton
    // launch on a grid of `(N  + BLOCK - 1) // BLOCK` programs
    __global__ void add(float* z, float* x, float* y, int N){
        // program id
        int pid = get_program_id(0);
        // create arrays of pointers
        int offset[BLOCK] = pid * BLOCK + 0 ... BLOCK;
        float* pz[BLOCK] = z + offset;
        float* px[BLOCK] = x + offset;
        float* py[BLOCK] = y + offset;
        // bounds checking
        bool check[BLOCK] = offset < N;
        // write-back
        *?(check)pz = *?(check)px + *?(check)py;
    }

As you can see, arrays are first-class citizen in Triton. This has a number of important advantages that will be highlighted in the next tutorial. For now, let's keep it simple and see how to execute the above operation in PyTorch.

---------------
PyTorch Wrapper
---------------

As you will see, a wrapper for the above Triton function can be created in just a few lines of pure python code.

.. code-block:: python


    # source-code for Triton compute kernel
    _src = '''
    __global__ void add(float* z, float* x, float* y, int N){
        // program id
        int pid = get_program_id(0);
        // create arrays of pointers
        int offset[BLOCK] = pid * BLOCK + 0 ... BLOCK;
        float* pz[BLOCK] = z + offset;
        float* px[BLOCK] = x + offset;
        float* py[BLOCK] = y + offset;
        // bounds checking
        bool check[BLOCK] = offset < N;
        // write-back
        *?(check)pz = *?(check)px + *?(check)py;
    }
        '''
    # This function returns a callable `triton.kernel` object
    # created from the above source code.
    # For portability, we maintain a cache of kernels for different `torch.device`
    # We compile the kernel with -DBLOCK=1024
    _kernels = dict()
    def make_add_kernel(device):
        if device not in _kernels:
            defines = {'BLOCK': 1024}
            _kernels[device] = triton.kernel(_src, device = device, defines=defines)
        return _kernels[device]
    
    
    # This is a standard torch custom autograd Function
    # The only difference is that we can now use the above kernel
    # in the `forward` and `backward` functions.
    class _add(torch.autograd.Function):
        
        @staticmethod
        def forward(ctx, x, y):
            # constraints of the op
            assert x.dtype == torch.float32
            # *allocate output*
            z = torch.empty_like(x)
            # *create launch grid*:
            # this is a function which takes compilation parameters `opt`
            # as input and returns a grid for launching the kernel.
            # triton.cdiv is a shortcut for ceil division:
            # triton.cdiv(a, b) = (a + b - 1) // b
            N = z.shape[0]
            grid = lambda opt: (triton.cdiv(N, opt.BLOCK), )
            # *launch kernel*:
            # pointer to the data of torch tensors can be retrieved with
            # the `.data_ptr()` method
            kernel = make_add_kernel(z.device)
            kernel(z.data_ptr(), x.data_ptr(), y.data_ptr(), N, grid = grid)
            return z

    # get callable from Triton function
    add = _add.apply

    # test
    torch.manual_seed(0)
    x = torch.rand(98432, device='cuda')
    y = torch.rand(98432, device='cuda')
    za = x + y
    zb = add(x, y)
    print(za)
    print(zb)
    print(f'The maximum difference between torch and triton is '
        f'{torch.max(torch.abs(za - zb))}')


---------------
Benchmarking
---------------

