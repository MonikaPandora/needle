"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND 
from .ops_tuple import *

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * log(a)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * (self.scalar * power_scalar(node.inputs[0], self.scalar - 1))
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        # assert a.dtype == b.dtype, f'{a.dtype}, {b.dtype}'
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        lgrad = out_grad / rhs
        rgrad = negate(out_grad * lhs / power_scalar(rhs, 2))
        return lgrad, rgrad 
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        axes_ = list(range(a.ndim))
        if self.axes:
            axes_[self.axes[1]], axes_[self.axes[0]] = self.axes
        else:
            axes_[-2], axes_[-1] = axes_[-1], axes_[-2]
        return a.permute(tuple(axes_))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, axes=self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return reshape(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape
        self.same = False

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if a.shape == self.shape:
            self.same = True
            return a
        return a.broadcast_to(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        if self.same:
            return out_grad
        bias = len(self.shape) - len(node.inputs[0].shape)
        axes = list(range(bias))
        for i in range(len(node.inputs[0].shape)):
            if self.shape[bias + i] > node.inputs[0].shape[i]:
                axes.append(bias + i)
        return reshape(summation(out_grad, axes=tuple(axes)), node.inputs[0].shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if isinstance(axes, int):
            self.axes = (axes,)
        else:
            self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        shape = list(node.inputs[0].shape)
        if self.axes is not None:
            for axe in self.axes:
                shape[axe] = 1
        else:
            shape = [1] * len(node.inputs[0].shape)
        return broadcast_to(reshape(out_grad, tuple(shape)), node.inputs[0].shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        # assert a.dtype == b.dtype, f'{a.dtype}, {b.dtype}'
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        def get_size(shape):
            ret = 1
            for n in shape:
                ret *= n
            return ret

        lhs, rhs = node.inputs
        lgrad = matmul(out_grad, transpose(rhs))
        rgrad = matmul(transpose(lhs), out_grad)
        
        if (l := get_size(lhs.shape[:-2])) != (r := get_size(rhs.shape[:-2])):
            if l > r:
                node.inputs = node.inputs[::-1]
                rgrad = BroadcastTo(lhs.shape[:-2] + rhs.shape[-2:]).gradient(rgrad, node)
                node.inputs = node.inputs[::-1]
            else:
                lgrad = BroadcastTo(rhs.shape[:-2] + lhs.shape[-2:]).gradient(lgrad, node)
        
        return lgrad, rgrad
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return negate(out_grad)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / node.inputs[0]
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * exp(node.inputs[0])
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.maximum(0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * Tensor(array_api.array(node.inputs[0].realize_cached_data() > 0), dtype=out_grad.dtype, device=out_grad.device)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)

class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        exp_a = array_api.exp(a)
        exp_minus_a = array_api.exp(-a)
        return (exp_a - exp_minus_a) / (exp_a + exp_minus_a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * (1 - tanh(node.inputs[0])**2)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: List[NDArray]) -> NDArray:
        ### BEGIN YOUR SOLUTION
        # if not len(args):
        #     return args[0].reshape(tuple([1] + list(args[0].shape)))
        
        shape = args[0].shape
        fin_shape = [len(args)] + list(shape)

        ret = numpy.random.randn(*fin_shape)
        for i in range(len(args)):
            if args[i].shape != shape:
                raise ValueError('can not stack tensors of different shapes')
            ret[i] = args[i].numpy()
        ret = array_api.array(ret, device=args[0].device)

        if self.axis == 0:
            return ret
        
        permu = list(range(1, len(fin_shape)))
        permu = permu[:self.axis] + [0] + permu[self.axis:]
        return ret.permute(tuple(permu))
        ### END YOUR SOLUTION


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        device = A.device
        if self.axis:
            permu = list(range(len(A.shape)))
            permu = [self.axis] + permu[:self.axis] + permu[1 + self.axis:]
            A = A.permute(tuple(permu)).numpy()
        else:
            A = A.numpy()
        lst = [array_api.array(A[i], device=device) for i in range(A.shape[0])]
        return tuple(lst)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        out_grad.cached_data = out_grad.cached_data.flip(self.axes)
        return out_grad
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int, mode='extern'):
        self.axes = axes
        self.dilation = dilation
        self.mode = mode

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.dilation:
            if self.mode == 'extern':
                step = 1 + self.dilation
                new_shape = tuple((step * s if i in self.axes else s 
                            for i, s in enumerate(a.shape)))
                slices = tuple((slice(0, new_shape[i], step) if i in self.axes else slice(None)
                        for i in range(len(a.shape))))
                ret = a.device.full(new_shape, 0)
                ret[slices] = a
                return ret
            elif self.mode == 'inner':
                step = 1 + self.dilation
                new_shape = tuple((step * s - self.dilation if i in self.axes else s 
                            for i, s in enumerate(a.shape)))
                slices = tuple((slice(0, new_shape[i], step) if i in self.axes else slice(None)
                        for i in range(len(a.shape))))
                ret = a.device.full(new_shape, 0)
                ret[slices] = a
                return ret
            else:
                raise ValueError('invalid mode')
        return a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, axes=self.axes, dilation=self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation, mode='extern'):
    return Dilate(axes, dilation, mode=mode)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int, mode='extern'):
        self.axes = axes
        self.dilation = dilation
        self.mode = mode

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.dilation:
            step = 1 + self.dilation
            slices = tuple((slice(0, None, step) if i in self.axes else slice(None)
                    for i in range(len(a.shape))))
            return a[slices]
        return a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, axes=self.axes, dilation=self.dilation, mode=self.mode)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation, mode='extern'):
    return UnDilate(axes, dilation, mode=mode)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        def im2col(Z, kh, kw, s=1):
            n, h, w, c = Z.shape
            nh = (h - kh) // s + 1
            nw = (w - kw) // s + 1
            lst_strides = list(Z.strides)
            new_strides = lst_strides[:3] + lst_strides[1:3] + lst_strides[-1:]
            new_strides[1] *= s
            new_strides[2] *= s

            return array_api.NDArray.make(shape=(n, nh, nw, kh, kw, c), 
                                       strides=tuple(new_strides),
                                       device=Z._device,
                                       handle=Z._handle,
                                       offset=Z._offset)
        
        
        assert len(A.shape) == 4 and len(B.shape) == 4, "requires 4 dims of each operand"

        # do padding
        if pad := self.padding:
            A = A.pad(((0, 0), (pad, pad), (pad, pad), (0, 0)))
        
        # unpack metadatas
        n, h, w, c_in = A.shape
        kh, kw, _, c_out = B.shape
        
        # convolution via im2col
        nh = (h - kh) // self.stride + 1
        nw = (w - kw) // self.stride + 1
        A = im2col(A, kh, kw, s=self.stride).reshape((n * nh * nw, kh * kw * c_in))
        B = B.reshape((kh * kw * c_in, c_out))
        ret = A @ B
        return ret.reshape((n, nh, nw, c_out))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        A, B = node.inputs
        n, h, w, c_in = A.shape
        kh, kw, _, _ = B.shape
        n, nh, nw, _ = out_grad.shape

        assert kh == kw, 'backward pass only support for square kernels'
        assert nh == nw, 'backward pass only support for square images'

        k = kh

        flipped_kernel = flip(B, axes=(0, 1)).transpose()
        if self.stride > 1:
            mode = 'extern' if (h + 2 * self.padding - k) % self.stride else 'inner'
            out_grad = dilate(out_grad, axes=(1, 2), dilation=self.stride - 1, mode=mode)
        input_grad = conv(out_grad, 
                          flipped_kernel, 
                          stride=1, 
                          padding=k - 1 - self.padding)
        
        kernel_grad = conv(A.transpose((0, 3)), out_grad.transpose((0, 2)).transpose((0, 1)), stride=1, padding=self.padding)
        return input_grad, kernel_grad.transpose((0, 2)).transpose((0, 1))
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)
