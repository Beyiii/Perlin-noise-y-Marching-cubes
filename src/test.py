import cupy as cp

source = r'''
extern "C" __global__
void add_one(float* x) {
    int i = threadIdx.x;
    x[i] += 1.0;
}
'''

module = cp.RawModule(code=source)
kernel = module.get_function("add_one")

x = cp.array([1, 2, 3], dtype=cp.float32)
kernel((1,), (3,), (x,))
print(x)  # debería mostrar [2. 3. 4.]
