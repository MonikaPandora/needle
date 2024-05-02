#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides



__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN SOLUTION
  if(gid < size){
    size_t t = gid;
    size_t sz = shape.size - 1;
    while(t){
      offset += (t % shape.data[sz]) * strides.data[sz];
      t /= shape.data[sz];
      sz--;
    }
    out[gid] = a[offset];
  }
  /// END SOLUTION
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the 
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give 
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to 
   * execute the underlying function.
   * 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}


__global__ void EwiseSetitemKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                                   CudaVec strides, size_t offset){
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if(gid < size){
    size_t t = gid;
    size_t sz = shape.size - 1;
    while(t){
      offset += (t % shape.data[sz]) * strides.data[sz];
      t /= shape.data[sz];
      sz--;
    }
    out[offset] = a[gid];
  }
}

void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   * 
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size, VecToCuda(shape),
                                              VecToCuda(strides), offset);
  /// END SOLUTION
}


__global__ void ScalarSetitemKernel(scalar_t a, scalar_t* out, size_t size, CudaVec shape,
                                   CudaVec strides, size_t offset){
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if(gid < size){
    size_t t = gid;
    size_t sz = shape.size - 1;
    while(t){
      offset += (t % shape.data[sz]) * strides.data[sz];
      t /= shape.data[sz];
      sz--;
    }
    out[offset] = a;
  }
}

void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   * 
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ScalarSetitemKernel<<<dim.grid, dim.block>>>(val, out->ptr, size, VecToCuda(shape),
                                              VecToCuda(strides), offset);
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

// __global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
//   size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
//   if (gid < size) out[gid] = a[gid] + b[gid];
// }

// void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
//   /**
//    * Add together two CUDA array
//    */
//   CudaDims dim = CudaOneDim(out->size);
//   EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
// }

// __global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
//   size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
//   if (gid < size) out[gid] = a[gid] + val;
// }

// void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
//   /**
//    * Add together a CUDA array and a scalar value.
//    */
//   CudaDims dim = CudaOneDim(out->size);
//   ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
// }

#define def_ewise_binop_cuda_kernel(name, op) \
__global__ void Ewise##name##Kernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) { \
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x; \
  if (gid < size) out[gid] = a[gid] op b[gid]; \
}

#define def_ewise_binop_cuda(name) \
void Ewise##name(const CudaArray& a, const CudaArray& b, CudaArray* out) { \
  CudaDims dim = CudaOneDim(out->size); \
  Ewise##name##Kernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size); \
}

#define def_scalar_binop_cuda_kernel(name, op) \
__global__ void Scalar##name##Kernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) { \
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x; \
  if (gid < size) out[gid] = a[gid] op val; \
}

#define def_scalar_binop_cuda(name) \
void Scalar##name(const CudaArray& a, scalar_t val, CudaArray* out) { \
  CudaDims dim = CudaOneDim(out->size); \
  Scalar##name##Kernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size); \
}

#define def_ewise_binfunc_cuda_kernel(name, binfunc) \
__global__ void Ewise##name##Kernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size){ \
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x; \
  if (gid < size) out[gid] = binfunc(a[gid], b[gid]); \
}

#define def_ewise_binfunc_cuda(name) \
void Ewise##name(const CudaArray& a, const CudaArray& b, CudaArray* out) { \
  CudaDims dim = CudaOneDim(out->size); \
  Ewise##name##Kernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size); \
}

#define def_scalar_binfunc_cuda_kernel(name, binfunc) \
__global__ void Scalar##name##Kernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size){ \
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x; \
  if (gid < size) out[gid] = binfunc(a[gid], val); \
}

#define def_scalar_binfunc_cuda(name) \
void Scalar##name(const CudaArray& a, scalar_t val, CudaArray* out) { \
  CudaDims dim = CudaOneDim(out->size); \
  Scalar##name##Kernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size); \
}

#define def_ewise_ufunc_cuda_kernel(name, ufunc) \
__global__ void Ewise##name##Kernel(const scalar_t* a, scalar_t* out, size_t size){ \
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x; \
  if (gid < size) out[gid] = ufunc(a[gid]); \
}

#define def_ewise_ufunc_cuda(name) \
void Ewise##name(const CudaArray& a, CudaArray* out) { \
  CudaDims dim = CudaOneDim(out->size); \
  Ewise##name##Kernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size); \
}

/**
 * In the code the follows, use the above template to create analogous elementise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */


////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

/* element wise binary operations */
def_ewise_binop_cuda_kernel(Add, +)
def_ewise_binop_cuda(Add)

def_ewise_binop_cuda_kernel(Mul, *)
def_ewise_binop_cuda(Mul)

def_ewise_binop_cuda_kernel(Div, /)
def_ewise_binop_cuda(Div)

def_ewise_binop_cuda_kernel(Eq, ==)
def_ewise_binop_cuda(Eq)

def_ewise_binop_cuda_kernel(Ge, >=)
def_ewise_binop_cuda(Ge)

/* element wise binary operations */
def_scalar_binop_cuda_kernel(Add, +)
def_scalar_binop_cuda(Add)

def_scalar_binop_cuda_kernel(Mul, *)
def_scalar_binop_cuda(Mul)

def_scalar_binop_cuda_kernel(Div, /)
def_scalar_binop_cuda(Div)

def_scalar_binop_cuda_kernel(Eq, ==)
def_scalar_binop_cuda(Eq)

def_scalar_binop_cuda_kernel(Ge, >=)
def_scalar_binop_cuda(Ge)

/* element wise call binary functions */
def_ewise_binfunc_cuda_kernel(Maximum, max)
def_ewise_binfunc_cuda(Maximum)

/* call binary functions with scalar */
def_scalar_binfunc_cuda_kernel(Maximum, max)
def_scalar_binfunc_cuda(Maximum)

def_scalar_binfunc_cuda_kernel(Power, powf)
def_scalar_binfunc_cuda(Power)

/* element wise call unary functions */
def_ewise_ufunc_cuda_kernel(Log, logf)
def_ewise_ufunc_cuda(Log)

def_ewise_ufunc_cuda_kernel(Exp, expf)
def_ewise_ufunc_cuda(Exp)

def_ewise_ufunc_cuda_kernel(Tanh, tanhf)
def_ewise_ufunc_cuda(Tanh)


#define min(a, b) ((a) < (b) ? (a) : (b))

__global__ void MatmulKernel_01(const scalar_t* a, const scalar_t* b, scalar_t* out, uint32_t M, uint32_t N, uint32_t P){
  const int blockRow = blockIdx.y;
  const int blockCol = blockIdx.x;

  // which block
  scalar_t* outSub = &out[(blockRow * P + blockCol) * TILE];
  size_t outDimX = min(TILE, P - blockCol * TILE);
  size_t outDimY = min(TILE, M - blockRow * TILE);

  scalar_t val = 0.;
  for(size_t k = 0; k < N; k += TILE){
    __syncthreads();
    size_t innerDim = min(TILE, N - k);
    const scalar_t* aSub = &a[blockRow * TILE * N + k];
    const scalar_t* bSub = &b[k * P + blockCol * TILE];
    __shared__ scalar_t as[TILE][TILE];
    __shared__ scalar_t bs[TILE][TILE];
    if(threadIdx.x < innerDim && threadIdx.y < outDimY){
      as[threadIdx.y][threadIdx.x] = aSub[threadIdx.y * N + threadIdx.x];
    }
    if(threadIdx.x < outDimX && threadIdx.y < innerDim){
      bs[threadIdx.y][threadIdx.x] = bSub[threadIdx.y * P + threadIdx.x];
    }
    __syncthreads();

    #pragma unroll
    for(size_t e = 0; e < innerDim; ++e){
      if(threadIdx.y < outDimY && threadIdx.x < outDimX){
        val += as[threadIdx.y][e] * bs[e][threadIdx.x];
      }
    }
  }
  if(threadIdx.y < outDimY && threadIdx.x < outDimX)
    outSub[threadIdx.y * P + threadIdx.x] = val;
}


#define BLOCK_SIZE (4*TILE)
#define INNER_STEP 256
__global__ void MatmulKernel_02(const scalar_t* a, const scalar_t* b, scalar_t* out, uint32_t M, uint32_t N, uint32_t P){
  size_t blockRow = blockIdx.y;
  size_t blockCol = blockIdx.x;

  scalar_t* outBlockBase = &out[(blockRow * P + blockCol) * BLOCK_SIZE];
  int outBlockDimX = min(BLOCK_SIZE, P - blockCol * BLOCK_SIZE);
  int outBlockDimY = min(BLOCK_SIZE, M - blockRow * BLOCK_SIZE);

  // using int for threads whose dim is negtive
  int tileDimX = min(TILE, outBlockDimX - (int)(threadIdx.x * TILE));
  int tileDimY = min(TILE, outBlockDimY - (int)(threadIdx.y * TILE));

  scalar_t c[TILE][TILE] = {0.};
  __shared__ scalar_t as[BLOCK_SIZE][INNER_STEP];
  __shared__ scalar_t bs[INNER_STEP][BLOCK_SIZE];
  for(size_t k = 0; k < N; k += INNER_STEP){
    __syncthreads();
    const scalar_t* aBlockBase = &a[blockRow * BLOCK_SIZE * N + k];
    const scalar_t* bBlockBase = &b[k * P + blockCol * BLOCK_SIZE];
    size_t innerDim = min(INNER_STEP, N - k);
    for(size_t aj_bi = threadIdx.x; aj_bi < innerDim; aj_bi += blockDim.x){
      for(size_t ai = threadIdx.y; ai < outBlockDimY; ai += blockDim.y){
        as[ai][aj_bi] = aBlockBase[ai * N + aj_bi];
      }
      for(size_t bj = threadIdx.y; bj < outBlockDimX; bj += blockDim.y){
        bs[aj_bi][bj] = bBlockBase[aj_bi * P + bj];
      }
    }
    __syncthreads();

    #pragma unroll
    for(int i = 0; i < tileDimY; ++i){
      #pragma unroll
      for(int j = 0; j < tileDimX; ++j){
        #pragma unroll
        for(int e = 0; e < innerDim; ++e){
          c[i][j] += as[threadIdx.y * TILE + i][e] * bs[e][threadIdx.x * TILE + j];
        }
      }
    }
  }

  scalar_t* outTileBase = &outBlockBase[threadIdx.y * TILE * P + threadIdx.x * TILE];
  #pragma unroll
  for(int i = 0; i < tileDimY; ++i){
    #pragma unroll
    for(int j = 0; j < tileDimX; ++j){
      outTileBase[i * P + j] = c[i][j];
    }
  }
}

void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   * 
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */

  /// BEGIN SOLUTION

  // invoking kernel_01
  // size_t BM = (M + TILE - 1) / TILE;
  // size_t BP = (P + TILE - 1) / TILE;
  // dim3 grid = dim3(BP, BM, 1);
  // dim3 block = dim3(TILE, TILE, 1);
  // MatmulKernel_01<<<grid, block>>>(a.ptr, b.ptr, out->ptr, M, N, P);

  // invoking kernel_02
  size_t BM = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
  size_t BP = (P + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 grid = dim3(BP, BM, 1);
  dim3 block = dim3(BLOCK_SIZE / TILE, BLOCK_SIZE / TILE, 1);
  MatmulKernel_02<<<grid, block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
  /// END SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////
__global__ void ReduceMaxKernel(const scalar_t* a, scalar_t* out, size_t reduce_size, size_t size){
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if(gid < size){
    size_t start = gid * reduce_size;
    scalar_t tmp = a[start];
    for(size_t i = 1; i < reduce_size; ++i){
      tmp = max(tmp, a[start + i]);
    }
    out[gid] = tmp;
  }
}

void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size, out->size);
  /// END SOLUTION
}

__global__ void ReduceSumKernel(const scalar_t* a, scalar_t* out, size_t reduce_size, size_t size){
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if(gid < size){
    size_t start = gid * reduce_size;
    scalar_t tmp = 0;
    for(size_t i = 0; i < reduce_size; ++i){
      tmp += a[start + i];
    }
    out[gid] = tmp;
  }
}


void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you 
   * can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size, out->size);
  /// END SOLUTION
}

}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
