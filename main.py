import cupy
import os
from cupy.cuda import compiler
test_source = r'''
extern "C" __global__
void test_div(const float* x1, const float* x2, float* y, unsigned int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N)
        y[tid] = x1[tid] / (x2[tid] + 1.0);
}
'''
def generate_file(ext='cubin'):
    cc = cupy.cuda.get_nvcc_path()
    arch = '-gencode=arch=compute_{CC},code=sm_{CC}'.format(
    CC=compiler._get_arch())
    code = test_source
    cmd = cc.split()
    source = os.getcwd() + '/test_load_cubin.cu'
    file_path = os.getcwd() + '/test_load_cubin.cubin'
    flag = '-cubin'
    with open(source, 'w') as f:
        f.write(code)
    cmd += [arch, flag, source, '-o', file_path]
    cc = 'nvcc'
    print(cmd)
    compiler._run_cc(cmd, os.getcwd(), cc)

def _helper(kernel, dtype):
    N = 10
    x1 = cupy.arange(N**2, dtype=dtype).reshape(N, N)
    x2 = cupy.ones((N, N), dtype=dtype)
    y = cupy.zeros((N, N), dtype=dtype)
    kernel((N,), (N,), (x1, x2, y, N**2))
    return x1, x2, y


if __name__ == "__main__":
    # generate_file()
    mod = cupy.RawModule(path=os.getcwd() + '/test_load_cubin.cubin', backend="nvcc")
    ker = mod.get_function('test_div')
    x1, x2, y = _helper(ker, cupy.float32)
    assert cupy.allclose(y, x1 / (x2 + 1.0))
