"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.device =device
        self.dtype = dtype
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features), device=device, dtype=dtype)
        if bias:
            self.bias = Parameter(init.kaiming_uniform(out_features, 1).transpose((0, 1)), device=device, dtype=dtype)
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        assert X.shape[-1] == self.weight.shape[0], f'{X.shape}, {self.weight.shape}'
        out = X @ self.weight
        if self.bias:
            out = out + self.bias.broadcast_to(out.shape)
        return out
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        shape = X.shape
        if len(shape) == 1: 
            return X.reshape((1, shape[0]))
        
        dim = 1
        for i in range(1, len(shape)):
            dim *= shape[i]
        
        return X.reshape((shape[0], dim))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION
    

class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        assert len(y.shape) == 1
        if len(logits.shape) == 1:
            assert y.shape[0] == 1
            return ops.logsumexp(logits) - y
        else:
            assert y.shape[0] == logits.shape[-2]
            lse = ops.logsumexp(logits, axes=(-1,))
            y_one_hot = init.one_hot(logits.shape[-1], y, device=logits.device, dtype=logits.dtype)
            return ops.summation(lse - ops.summation(y_one_hot * logits, axes=(-1,)), axes=(-1,)) / logits.shape[-2]
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.device = device
        self.dtype = dtype
        self.weight = Parameter(init.ones(dim), device=device, dtype=dtype)
        self.bias = Parameter(init.zeros(dim), device=device, dtype=dtype)
        self.running_mean = init.zeros(dim, device=device, dtype=dtype, requires_grad=False)
        self.running_var = init.ones(dim, device=device, dtype=dtype, requires_grad=False)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        assert len(x.shape) == 2
        m = x.shape[0]
        dim = x.shape[1]
        assert self.dim == dim

        if self.training:
            u_observed = x.sum(axes=(0,)) / m
            self.running_mean.data = (1 - self.momentum) * self.running_mean + self.momentum * u_observed
            u_observed = u_observed.reshape(shape=(1, dim))

            var_observed = ops.summation((x - u_observed.broadcast_to(x.shape)) ** 2, axes=(0,)) / m
            self.running_var.data = (1 - self.momentum) * self.running_var + self.momentum * var_observed
            var_observed = var_observed.reshape(shape=(1, dim))

            w = self.weight.broadcast_to(x.shape)
            b = self.bias.broadcast_to(x.shape)
            u_observed = u_observed.broadcast_to(x.shape)
            var_observed = var_observed.broadcast_to(x.shape)
            return w * (x - u_observed) / (var_observed + self.eps) ** 0.5 + b
        
        norm = (x - self.running_mean.broadcast_to(x.shape)) / ops.broadcast_to((self.running_var + self.eps) ** 0.5, x.shape) 
        return self.weight.broadcast_to(norm.shape) * norm + self.bias.broadcast_to(norm.shape)
        ### END YOUR SOLUTION

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weights = Parameter(init.ones(dim), device=device, dtype=dtype)
        self.bias = Parameter(init.zeros(dim), device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        assert x.shape[-1] == self.dim
        w = self.weights.broadcast_to(x.shape)
        b = self.bias.broadcast_to(x.shape)

        e_x = x.sum(axes=(-1,)) / x.shape[-1]
        e_x = e_x.reshape(tuple(list(x.shape)[:-1] + [1])).broadcast_to(x.shape)

        var_x = ops.summation((x - e_x) ** 2, axes=(-1,)) / x.shape[-1] + self.eps
        var_x_sqrt = var_x ** 0.5
        var_x_sqrt = var_x_sqrt.reshape(tuple(list(x.shape)[:-1] + [1])).broadcast_to(x.shape)

        x = (x - e_x) / var_x_sqrt

        return w * x + b
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if not self.training:
            return x
        probs = init.randb(*x.shape, p=1 - self.p)
        return x * probs / (1 - self.p)
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION
