"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module, ReLU


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return init.ones_like(x, device=x.device) / (1 + ops.exp(-x))
        ### END YOUR SOLUTION

class TanH(Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: Tensor) -> Tensor:
        return ops.tanh(x)

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        bound = (1 / hidden_size)**0.5
        self.W_ih = Parameter(init.rand(input_size, hidden_size, low=-bound, high=bound, 
                                        device=device, dtype=dtype, requires_grad=True))
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size, low=-bound, high=bound, 
                                        device=device, dtype=dtype, requires_grad=True))

        self.bias = bias

        nonlinears = {
            'tanh': TanH,
            'relu': ReLU,
            'sigmoid': Sigmoid
        }

        self.nonlinear = nonlinears[nonlinearity]()
        if bias:
            self.bias_ih = Parameter(init.rand(hidden_size, low=-bound, high=bound, 
                                               device=device, dtype=dtype, requires_grad=True))
            self.bias_hh = Parameter(init.rand(hidden_size, low=-bound, high=bound, 
                                               device=device, dtype=dtype, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h is None:
            h = init.zeros(X.shape[0], self.W_ih.shape[1], device=X.device, dtype=X.dtype)
        
        out = X @ self.W_ih + h @ self.W_hh
        if self.bias:
            out = out + self.bias_ih.broadcast_to(out.shape) + self.bias_hh.broadcast_to(out.shape)
        return self.nonlinear(out)
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        assert num_layers > 0
        self.d = hidden_size
        self.rnn_cells = [RNNCell(input_size, hidden_size, bias, nonlinearity, device, dtype)]
        self.rnn_cells += [RNNCell(hidden_size, hidden_size, bias, nonlinearity, device, dtype) for _ in range(num_layers - 1)]
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h0 is None:
            h0 = init.zeros(len(self.rnn_cells), X.shape[1], self.d, device=X.device, dtype=X.dtype)

        features = list(ops.split(X, axis=0).tuple())
        hiddens = list(ops.split(h0, axis=0).tuple())

        for k in range(len(self.rnn_cells)):
            cell = self.rnn_cells[k]
            h = hiddens[k]
            for i in range(len(features)):
                features[i] = cell(features[i], h)
                h = features[i]
            hiddens[k] = features[-1]

        features = ops.make_tuple(*features)
        hiddens = ops.make_tuple(*hiddens)
        return ops.stack(features, axis=0), ops.stack(hiddens, axis=0)
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        bound = (1 / hidden_size)**0.5

        self.bias = bias
        self.d = hidden_size
        self.W_ih = Parameter(init.rand(input_size, 4 * hidden_size, low=-bound, high=bound, 
                                        device=device, dtype=dtype, requires_grad=True))
        self.W_hh = Parameter(init.rand(hidden_size, 4 * hidden_size, low=-bound, high=bound, 
                                        device=device, dtype=dtype, requires_grad=True))
        
        if bias:
            self.bias_ih = Parameter(init.rand(4 * hidden_size, low=-bound, high=bound, 
                                        device=device, dtype=dtype, requires_grad=True))
            self.bias_hh = Parameter(init.rand(4 * hidden_size, low=-bound, high=bound, 
                                        device=device, dtype=dtype, requires_grad=True))
        
        self.sigmoid = Sigmoid()
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs, _ = X.shape
        if h is None:
            h0 = init.zeros(bs, self.d, device=X.device, dtype=X.dtype)
            c0 = init.zeros(bs, self.d, device=X.device, dtype=X.dtype)
        else:
            h0, c0 = h
        
        ifgo = X @ self.W_ih + h0 @ self.W_hh
        if self.bias:
            ifgo = ifgo + self.bias_hh.broadcast_to(ifgo.shape) + self.bias_ih.broadcast_to(ifgo.shape)
        
        ifgo = list(ops.split(ifgo.reshape((bs, 4, self.d)), axis=1).tuple())
        i, f, g, o = ifgo
        i = self.sigmoid(i)
        f = self.sigmoid(f)
        g = ops.tanh(g)
        o = self.sigmoid(o)

        c = c0 * f + i * g
        h = ops.tanh(c) * o
        return h, c
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        assert num_layers > 0
        self.d = hidden_size
        self.lstm_cells = [LSTMCell(input_size, hidden_size, bias, device, dtype)]
        self.lstm_cells += [LSTMCell(hidden_size, hidden_size, bias, device, dtype) for _ in range(num_layers - 1)]
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            c_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs = X.shape[1]
        if h is None:
            h0 = init.zeros(len(self.lstm_cells), bs, self.d, device=X.device, dtype=X.dtype)
            c0 = init.zeros(len(self.lstm_cells), bs, self.d, device=X.device, dtype=X.dtype)
        else:
            h0, c0 = h

        features = list(ops.split(X, axis=0).tuple())
        hiddens = list(ops.split(h0, axis=0).tuple())
        states = list(ops.split(c0, axis=0).tuple())

        for k in range(len(self.lstm_cells)):
            cell = self.lstm_cells[k]
            h = hiddens[k]
            c = states[k]
            for i in range(len(features)):
                h, c = cell(features[i], (h, c))
                features[i] = h
            hiddens[k] = h
            states[k] = c

        features = ops.make_tuple(*features)
        hiddens = ops.make_tuple(*hiddens)
        states = ops.make_tuple(*states)
        return ops.stack(features, axis=0), (ops.stack(hiddens, axis=0), ops.stack(states, axis=0))
        ### END YOUR SOLUTION

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.randn(num_embeddings, embedding_dim, mean=0, std=1,
                                           device=device, dtype=dtype, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        n = self.weight.shape[0]
        seq_len, bs = x.shape
        idx = x.reshape((seq_len * bs,))
        embed = init.one_hot(n, idx, device=self.weight.device, dtype=self.weight.dtype) @ self.weight
        return embed.reshape((seq_len, bs, self.weight.shape[1]))
        ### END YOUR SOLUTION