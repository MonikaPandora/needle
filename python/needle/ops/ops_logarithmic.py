from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_z = Z.max(axis=self.axes, keepdims=True).broadcast_to(Z.shape)
        max_z_nodim = Z.max(axis=self.axes)
        return array_api.exp(Z - max_z).sum(axis=self.axes).log() + max_z_nodim
        # return array_api.log(array_api.sum(array_api.exp(Z - max_z), axis=self.axes)) + max_z_nodim
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        max_z = Tensor(Z)
        max_z.cached_data = max_z.cached_data.max(axis=self.axes, keepdims=True).broadcast_to(Z.shape)
        exp_z = exp(Z - max_z)

        shape = list(node.inputs[0].shape)
        if self.axes:
            for axe in self.axes:
                shape[axe] = 1
        else:
            shape = [1] * len(node.inputs[0].shape)

        return exp_z / broadcast_to(reshape(summation(exp_z, axes=self.axes) / out_grad, tuple(shape)), Z.shape)
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

