"""Neural network models (PyTorch).

GCN          : graph convolutional network; weights are shared across buses and
               each layer mixes information between neighbouring buses using the
               network's current topology.
MLP          : topology-blind baseline; flattens all bus features into one vector.
MLP+status   : the same MLP, also given a 0/1 vector saying which branches are
               in service, so it can in principle learn the effect of outages.
"""

import torch
from torch import nn


class GCNLayer(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)

    def forward(self, h, a_hat):
        # h: (batch, buses, d_in), a_hat: (batch, buses, buses)
        return a_hat @ self.lin(h)


class GCN(nn.Module):
    def __init__(self, d_in=7, hidden=64, layers=6, d_out=2):
        super().__init__()
        self.inp = nn.Linear(d_in, hidden)
        self.layers = nn.ModuleList([GCNLayer(hidden, hidden) for _ in range(layers)])
        self.out = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, d_out))

    def forward(self, x, a_hat, status=None):
        h = torch.relu(self.inp(x))
        for layer in self.layers:
            h = h + torch.relu(layer(h, a_hat))  # residual connection
        return self.out(h)


class MLP(nn.Module):
    def __init__(self, n_bus=30, d_in=7, d_out=2, hidden=256, n_status=0):
        super().__init__()
        self.n_bus, self.d_out, self.use_status = n_bus, d_out, n_status > 0
        self.net = nn.Sequential(
            nn.Linear(n_bus * d_in + n_status, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_bus * d_out),
        )

    def forward(self, x, a_hat=None, status=None):
        z = x.flatten(1)
        if self.use_status:
            z = torch.cat([z, status], dim=1)
        return self.net(z).view(-1, self.n_bus, self.d_out)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
