import torch
import torch.nn as nn
import torch.nn.functional as F

class CRATELayer(nn.Module):
    """One learned ISTA‐step + ReLU, as in your Flax CRATE."""
    def __init__(self, dim, step_size=0.1, lambd=0.1):
        super().__init__()
        self.dim = dim
        self.step_size = step_size
        self.lambd = lambd
        # D ∈ R^{dim×dim}
        self.weight = nn.Parameter(torch.empty(dim, dim))
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')

    def forward(self, x):
        # x: (batch, dim)
        x1 = x @ self.weight               # x @ D
        grad_1 = x1 @ self.weight.t()      # (x @ D) @ D^T
        grad_2 = x @ self.weight.t()       # x @ D^T
        update = self.step_size * (grad_2 - grad_1) - self.step_size * self.lambd
        return F.relu(x + update)


class Layer(nn.Module):
    def __init__(self, in_shape, out_shape, act_type='relu'):
        super().__init__()
        self.fc = nn.Linear(in_shape, out_shape, bias=True)
        nn.init.kaiming_uniform_(self.fc.weight, nonlinearity=act_type)
        self.fc.bias.data.fill_(0.0)

        if act_type == 'linear':
            self.act = None
        else:
            act_cls = {
                'sigmoid': nn.Sigmoid,
                'tanh':    nn.Tanh,
                'relu':    nn.ReLU,
                'selu':    nn.SELU,
                'swish':   nn.SiLU,
                'leaky_relu': nn.LeakyReLU,
                'elu':     nn.ELU
            }[act_type]
            self.act = act_cls()

    def forward(self, x):
        x = self.fc(x)
        return x if self.act is None else self.act(x)


class DeepFFNN(nn.Module):
    def __init__(self,
                 input_size,
                 num_features=2000,
                 num_outputs=1,
                 num_hidden_layers=2,
                 act_type='relu',
                 crate_step_size=0.1,
                 crate_lambd=0.1):
        super().__init__()
        self.num_hidden = num_hidden_layers

        # input layer + its CRATE
        self.in_layer = Layer(input_size, num_features, act_type)
        self.in_crate  = CRATELayer(num_features, crate_step_size, crate_lambd)

        # hidden layers + CRATEs
        self.hidden_layers = nn.ModuleList()
        self.hidden_crates = nn.ModuleList()
        for _ in range(num_hidden_layers - 1):
            self.hidden_layers.append(Layer(num_features, num_features, act_type))
            self.hidden_crates.append(CRATELayer(num_features, crate_step_size, crate_lambd))

        # output (no activation, no CRATE)
        self.out_layer = Layer(num_features, num_outputs, act_type='linear')

    def predict(self, x):
        activations = []

        # input → activation → CRATE
        out = self.in_layer(x)
        out = self.in_crate(out)
        activations.append(out)

        # each hidden block
        for layer, crate in zip(self.hidden_layers, self.hidden_crates):
            out = layer(out)
            out = crate(out)
            activations.append(out)

        # final output
        out = self.out_layer(out)
        return out, activations
