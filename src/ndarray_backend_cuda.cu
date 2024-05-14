#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

std::unordered_map<size_t, std::vector<scalar_t*>> buffer;

struct CudaArray {
  CudaArray(const size_t size) {
    if(buffer[size].size()){
      ptr = buffer[size].back();
      buffer[size].pop_back();
    }
    else {

      cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
      if (err != cudaSuccess){
        throw std::runtime_error(cudaGetErrorString(err));
      }
    }
    this->size = size;
  }
  ~CudaArray() { 
    cudaFree(ptr);
    // buffer[size].push_back(ptr);
  }
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


// #define min(a, b) ((a) < (b) ? (a) : (b))

/*
  Naive version of matmul
*/

__global__ void MatmulKernel_00(const scalar_t* a, const scalar_t* b, scalar_t* out, uint32_t M, uint32_t N, uint32_t P){
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int ty = blockIdx.y * blockDim.y + threadIdx.y;
  if(ty < M && tx < P) {
      scalar_t c = 0;
      for(int i = 0; i < N; ++i){
          c += a[ty * N + i] * b[i * P + tx];
      }
      out[ty * P + tx] = c;
  }
}

/*
  simple using of shared memory
*/
#define KERNEL1_BLOCK_SIZE 16
__global__ void MatmulKernel_01(const scalar_t* a, const scalar_t* b, scalar_t* out, uint32_t M, uint32_t N, uint32_t P){
  const int blockRow = blockIdx.y;
  const int blockCol = blockIdx.x;

  // which block
  scalar_t* outSub = &out[(blockRow * P + blockCol) * KERNEL1_BLOCK_SIZE];
  size_t outDimX = min(KERNEL1_BLOCK_SIZE, P - blockCol * KERNEL1_BLOCK_SIZE);
  size_t outDimY = min(KERNEL1_BLOCK_SIZE, M - blockRow * KERNEL1_BLOCK_SIZE);

  __shared__ scalar_t as[KERNEL1_BLOCK_SIZE][KERNEL1_BLOCK_SIZE];
  __shared__ scalar_t bs[KERNEL1_BLOCK_SIZE][KERNEL1_BLOCK_SIZE];

  scalar_t val = 0.;
  for(size_t k = 0; k < N; k += KERNEL1_BLOCK_SIZE){
    __syncthreads();
    size_t innerDim = min((int)KERNEL1_BLOCK_SIZE, (int)(N - k));
    const scalar_t* aSub = &a[blockRow * KERNEL1_BLOCK_SIZE * N + k];
    const scalar_t* bSub = &b[k * P + blockCol * KERNEL1_BLOCK_SIZE];
    
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

/*
  each thread computes a tile * tile submatrix
*/
#define T 8
#define BLOCK_SIZE 128
#define INNER_STEP 8
__global__ void MatmulKernel_02(const scalar_t* a, const scalar_t* b, scalar_t* out, int M, int N, int P){
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int outBlockBaseOff = (by * P + bx) * BLOCK_SIZE;
  int outBlockDimX = min(BLOCK_SIZE, P - bx * BLOCK_SIZE);
  int outBlockDimY = min(BLOCK_SIZE, M - by * BLOCK_SIZE);

  // using int for threads whose dim is negtive
  int tileDimX = min(T, outBlockDimX - tx * T);
  int tileDimY = min(T, outBlockDimY - ty * T);

  scalar_t c[T][T] = {0.};
  __shared__ scalar_t as[INNER_STEP][BLOCK_SIZE];
  __shared__ scalar_t bs[INNER_STEP][BLOCK_SIZE];


  int tid = ty * blockDim.x + tx;
  int a_i = tid >> 1;
  int a_j = (tid & 1) << 2;
  int b_i = tid >> 5;
  int b_j = (tid & 31) << 2;
  for(int k = 0; k < N; k += INNER_STEP){
    int aBlockBaseOff = by * BLOCK_SIZE * N + k;
    int bBlockBaseOff = k * P + bx * BLOCK_SIZE;
    int innerDim = min(INNER_STEP, N - k);

    if(a_i < outBlockDimY && a_j < innerDim){
      if((N & 3) || innerDim - a_j < 4){
        // not aligned to 128 bytes
        // or no 4 more elements
        for(int t = 0; t < min(innerDim - a_j, 4); ++t){
          as[a_j + t][a_i] = a[aBlockBaseOff + a_i * N + a_j + t];
        }
      }
      else{
        float4 load_a_reg = reinterpret_cast<const float4*>(&a[aBlockBaseOff + a_i * N + a_j])[0];
        as[a_j][a_i] = load_a_reg.x;
        as[a_j + 1][a_i] = load_a_reg.y;
        as[a_j + 2][a_i] = load_a_reg.z;
        as[a_j + 3][a_i] = load_a_reg.w;
      }
    }
    
    if(b_i < innerDim && b_j < outBlockDimX){
      if((P & 3) || outBlockDimX - b_j < 4){
        // not aligned to 128 bytes
        // or no 4 more elements
        for(int t = 0; t < min(outBlockDimX - b_j, 4); ++t){
          bs[b_i][b_j + t] = b[bBlockBaseOff + b_i * P + b_j + t];
        }
      }
      else{
        reinterpret_cast<float4*>(&bs[b_i][b_j])[0] = reinterpret_cast<const float4*>(&b[bBlockBaseOff + b_i * P + b_j])[0];
      }
    }
    
    // sync for loading data into shared memory
    __syncthreads();
    float compute_a_reg[T];
    float compute_b_reg[T];

    for(int e = 0; e < innerDim; ++e){
        reinterpret_cast<float4*>(&compute_a_reg[0])[0] = reinterpret_cast<float4*>(&as[e][ty * T])[0];
        reinterpret_cast<float4*>(&compute_a_reg[4])[0] = reinterpret_cast<float4*>(&as[e][ty * T + 4])[0];
        reinterpret_cast<float4*>(&compute_b_reg[0])[0] = reinterpret_cast<float4*>(&bs[e][tx * T])[0];
        reinterpret_cast<float4*>(&compute_b_reg[4])[0] = reinterpret_cast<float4*>(&bs[e][tx * T + 4])[0];
        for(int i = 0; i < tileDimY; ++i){
            for(int j = 0; j < tileDimX; ++j){
                c[i][j] += compute_a_reg[i] * compute_b_reg[j];
            }
        }
    }
    __syncthreads();
  }

  int outTileBaseOff = outBlockBaseOff + ty * T * P + tx * T;
  if(P & 3) {
    for(int i = 0; i < tileDimY; ++i){
      for(int j = 0; j < tileDimX; ++j){
        out[outTileBaseOff + i * P + j] = c[i][j];
      }
    }
  }
  else {
    for(int i = 0; i < tileDimY; ++i){
      for(int j = 0; j < tileDimX; j+=4){
        reinterpret_cast<float4*>(&out[outTileBaseOff + i * P + j])[0] = reinterpret_cast<const float4*>(&c[i][j])[0];
      }
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
  // int kernel = 2;
  // switch(kernel){
  //   case 0: {
  //     // invoking kernel_00
  //     dim3 dimGrid((P + 15) / 16, (M + 15) / 16, 1);
  //     dim3 dimBlock(16, 16, 1);
  //     MatmulKernel_00<<<dimGrid, dimBlock>>>(a.ptr, b.ptr, out->ptr, M, N, P);
  //     break;
  //   }
  //   case 1: {
  //     // invoking kernel_01
  //     size_t BM = (M + KERNEL1_BLOCK_SIZE - 1) / KERNEL1_BLOCK_SIZE;
  //     size_t BP = (P + KERNEL1_BLOCK_SIZE - 1) / KERNEL1_BLOCK_SIZE;
  //     dim3 grid = dim3(BP, BM, 1);
  //     dim3 block = dim3(KERNEL1_BLOCK_SIZE, KERNEL1_BLOCK_SIZE, 1);
  //     MatmulKernel_01<<<grid, block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
  //     break;
  //   }
  //   case 2: {
  //     // invoking kernel_02
  //     size_t BM = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
  //     size_t BP = (P + BLOCK_SIZE - 1) / BLOCK_SIZE;
  //     dim3 grid = dim3(BP, BM);
  //     dim3 block = dim3(BLOCK_SIZE / T, BLOCK_SIZE / T);
  //     MatmulKernel_02<<<grid, block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
  //     break;
  //   }
  //   default: assert(0);
  // }

  // if(M % 8 == 0 && N % 8 == 0 && P % 8 == 0 && 
  //     M >= 512 && M == N && P == N){
  //   size_t BM = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
  //   size_t BP = (P + BLOCK_SIZE - 1) / BLOCK_SIZE;
  //   dim3 grid = dim3(BP, BM);
  //   dim3 block = dim3(BLOCK_SIZE / T, BLOCK_SIZE / T);
  //   MatmulKernel_02<<<grid, block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
  // }
  // else {
    size_t BM = (M + KERNEL1_BLOCK_SIZE - 1) / KERNEL1_BLOCK_SIZE;
    size_t BP = (P + KERNEL1_BLOCK_SIZE - 1) / KERNEL1_BLOCK_SIZE;
    dim3 grid = dim3(BP, BM, 1);
    dim3 block = dim3(KERNEL1_BLOCK_SIZE, KERNEL1_BLOCK_SIZE, 1);
    MatmulKernel_01<<<grid, block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
  // }

  cudaDeviceSynchronize();
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
