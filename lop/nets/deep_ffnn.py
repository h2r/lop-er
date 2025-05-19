import torch
import torch.nn as nn
import torch.nn.functional as F

# class CRATELayer(nn.Module):
#     """One learned ISTA‐step + ReLU, as in your Flax CRATE."""
#     def __init__(self, dim, step_size=0.1, lambd=0.1):
#         super().__init__()
#         self.dim = dim
#         self.step_size = step_size
#         self.lambd = lambd
#         # D ∈ R^{dim×dim}
#         self.weight = nn.Parameter(torch.empty(dim, dim))
#         nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')

#     def forward(self, x):
#         # x: (batch, dim)
#         x1 = x @ self.weight               # x @ D
#         grad_1 = x1 @ self.weight.t()      # (x @ D) @ D^T
#         grad_2 = x @ self.weight.t()       # x @ D^T
#         update = self.step_size * (grad_2 - grad_1) - self.step_size * self.lambd
#         return F.relu(x + update)


# class Layer(nn.Module):
#     def __init__(self, in_shape, out_shape, act_type='relu'):
#         super().__init__()
#         self.fc = nn.Linear(in_shape, out_shape, bias=True)
#         nn.init.kaiming_uniform_(self.fc.weight, nonlinearity=act_type)
#         self.fc.bias.data.fill_(0.0)

#         if act_type == 'linear':
#             self.act = None
#         else:
#             act_cls = {
#                 'sigmoid': nn.Sigmoid,
#                 'tanh':    nn.Tanh,
#                 'relu':    nn.ReLU,
#                 'selu':    nn.SELU,
#                 'swish':   nn.SiLU,
#                 'leaky_relu': nn.LeakyReLU,
#                 'elu':     nn.ELU
#             }[act_type]
#             self.act = act_cls()

#     def forward(self, x):
#         x = self.fc(x)
#         return x if self.act is None else self.act(x)

class Layer(nn.Module):
    def __init__(self, in_shape, out_shape, act_type='relu'):
        super(Layer, self).__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.act_type = act_type

        self.layers = nn.ModuleList()

        bias = True
        self.fc = nn.Linear(in_shape, out_shape, bias=bias)
        self.layers.append(self.fc)

        if self.act_type == 'linear':
            self.act_layer = None
        else:
            self.act_layer = {'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'relu': nn.ReLU, 'selu': nn.SELU,
                                      'swish': nn.SiLU, 'leaky_relu': nn.LeakyReLU, 'elu': nn.ELU}[self.act_type]
            self.act_layer = self.act_layer()
            self.layers.append(self.act_layer)

        # Initialize the weights
        if bias:
            self.fc.bias.data.fill_(0.0)
        nn.init.kaiming_uniform_(self.fc.weight, nonlinearity=self.act_type)

    def forward(self, x):
        x = self.fc(x)
        if self.act_layer is not None:
            x = self.act_layer(x)
        return x


class DeepFFNN(nn.Module):
    def __init__(self, input_size, num_features=2000, num_outputs=1, num_hidden_layers=2, act_type='relu'):
        super(DeepFFNN, self).__init__()
        self.num_inputs = input_size
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.num_hidden_layers = num_hidden_layers
        self.act_type = act_type
        self.layers_to_log = [-(i * 2 + 1) for i in range(num_hidden_layers + 1)]

        # define the architecture
        self.layers = nn.ModuleList()

        self.in_layer = Layer(in_shape=input_size, out_shape=num_features, act_type=self.act_type)
        self.layers.extend(self.in_layer.layers)

        self.hidden_layers = []
        for i in range(self.num_hidden_layers - 1):
            self.hidden_layers.append(Layer(in_shape=num_features, out_shape=num_features, act_type=self.act_type))
            self.layers.extend(self.hidden_layers[i].layers)

        self.out_layer = Layer(in_shape=num_features, out_shape=num_outputs, act_type='linear')
        self.layers.extend(self.out_layer.layers)

    def predict(self, x):
        """
        Forward pass
        :param x: input
        :return: estimated output
        """
        activations = []
        out = self.in_layer.forward(x=x)
        activations.append(out)
        for hidden_layer in self.hidden_layers:
            out = hidden_layer.forward(x=out)
            activations.append(out)
        out = self.out_layer.forward(x=out)
        return out, activations
