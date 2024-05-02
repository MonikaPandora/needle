"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(
            init.kaiming_uniform(
                kernel_size**2 * in_channels,
                kernel_size**2 * out_channels,
                shape=(kernel_size, kernel_size, in_channels, out_channels),
                device=device,
                dtype=dtype,
            )
        )
        if bias:
            bound = 1.0 / (in_channels * kernel_size**2)**0.5
            self.bias = Parameter(init.rand(out_channels, low=-bound, high=bound, device=device, dtype=dtype), requires_grad=True)
        else:
            self.bias = None
        
        self.padding = padding
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        assert len(x.shape) in [3, 4]
        if len(x.shape) == 3:
            x = x.reshape(tuple([1] + list(x.shape)))
        n, c, h, w = x.shape

        assert h == w, 'currently only support for square images'

        pad = (self.kernel_size) // 2 if self.padding is None else self.padding
            
        conv_out = ops.conv(x.transpose((1, 3)).transpose((1, 2)), self.weight, stride=self.stride, padding=pad)
        bias = self.bias.reshape((1,1,1,self.out_channels)).broadcast_to(conv_out.shape)
        return (conv_out + bias).transpose((1, 3)).transpose((2, 3))
        ### END YOUR SOLUTION


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=1, padding=0, device=None):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.device = device
    
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError()
