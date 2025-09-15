"""
Minimal model stubs: Edge-aware ECC and GIN baseline.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import NNConv, GlobalAttention, Set2Set, GINConv
from torch_geometric.nn.norm import BatchNorm

class EdgeAwareECC(nn.Module):
    def __init__(self, in_dim, edge_dim, hidden=64, num_layers=2, readout='gap', num_classes=9):
        super().__init__()
        self.num_layers = num_layers
        self.convs, self.bns, self.acts, self.edge_mlps = nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        last = in_dim
        for _ in range(num_layers):
            edge_mlp = nn.Sequential(nn.Linear(edge_dim, last * hidden), nn.ReLU(),
                                     nn.Linear(last * hidden, last * hidden))
            conv = NNConv(last, hidden, edge_mlp, aggr='mean')
            self.convs.append(conv); self.bns.append(BatchNorm(hidden)); self.acts.append(nn.ReLU())
            self.edge_mlps.append(edge_mlp); last = hidden
        if readout == 'gap':
            self.gate = nn.Sequential(nn.Linear(last, 1))
            self.read = GlobalAttention(self.gate); out_dim = last
        elif readout == 'set2set':
            self.read = Set2Set(last, processing_steps=2); out_dim = 2*last
        else:
            # fallback to simple mean; requires torch_scatter but PyG provides global_mean_pool elsewhere
            from torch_geometric.nn import global_mean_pool
            self.read = lambda x, batch: global_mean_pool(x, batch); out_dim = last
        self.head_cls = nn.Sequential(nn.Linear(out_dim, hidden), nn.ReLU(), nn.Linear(hidden, num_classes))
        self.head_reg = nn.Sequential(nn.Linear(out_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, data):
        x, ei, ea, batch = data.x, data.edge_index, data.edge_attr, data.batch
        for conv, bn, act in zip(self.convs, self.bns, self.acts):
            x = conv(x, ei, ea); x = bn(x); x = act(x)
        g = self.read(x, batch)
        return self.head_cls(g), self.head_reg(g)

class GINBaseline(nn.Module):
    def __init__(self, in_dim, hidden=64, num_layers=2, readout='gap', num_classes=9):
        super().__init__()
        self.layers = nn.ModuleList()
        last = in_dim
        for _ in range(num_layers):
            mlp = nn.Sequential(nn.Linear(last, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.layers.append(GINConv(mlp)); last = hidden
        if readout == 'gap':
            self.gate = nn.Sequential(nn.Linear(last, 1))
            self.read = GlobalAttention(self.gate); out_dim = last
        else:
            from torch_geometric.nn import global_mean_pool
            self.read = lambda x, batch: global_mean_pool(x, batch); out_dim = last
        self.head_cls = nn.Sequential(nn.Linear(out_dim, hidden), nn.ReLU(), nn.Linear(hidden, num_classes))
        self.head_reg = nn.Sequential(nn.Linear(out_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        for layer in self.layers:
            x = torch.relu(layer(x, ei))
        g = self.read(x, batch)
        return self.head_cls(g), self.head_reg(g)