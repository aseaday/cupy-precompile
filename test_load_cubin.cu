
extern "C" __global__
void test_div(const float* x1, const float* x2, float* y, unsigned int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < N)
        y[tid] = x1[tid] / (x2[tid] + 1.0);
}
