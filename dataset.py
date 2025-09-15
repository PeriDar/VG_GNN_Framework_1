"""
PyG Dataset stub: windowed OHLCV -> Graph Data objects.
"""
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from vg_builder import assemble_graph

class WindowGraphDataset(InMemoryDataset):
    def __init__(self, df, indices, window, step, y9, yret, graph_kwargs, use_structural_features=True):
        self.df = df
        self.indices = indices
        self.window = window
        self.step = step
        self.y9 = y9
        self.yret = yret
        self.graph_kwargs = graph_kwargs
        self.use_structural_features = use_structural_features
        super().__init__(None)
        data_list = self._build()
        self.data, self.slices = self.collate(data_list)

    def _build(self):
        data_list = []
        idx = self.indices.values
        min_i, max_i = idx.min(), idx.max()
        for start in range(min_i, max_i - self.window + 1, self.step):
            end = start + self.window
            if not ((start in self.df.index) and (end-1 in self.df.index)):
                continue
            win = self.df.loc[start:end-1]
            y_cls = self.y9.loc[end-1]
            y_r = self.yret.loc[end-1]
            if np.isnan(y_cls) or np.isnan(y_r): 
                continue
            g = assemble_graph(win, **self.graph_kwargs)
            x = torch.tensor(g['X'], dtype=torch.float32)
            ei = torch.tensor(g['E'].T, dtype=torch.long)
            ea = torch.tensor(g['EA'], dtype=torch.float32)
            data = Data(x=x, edge_index=ei, edge_attr=ea,
                        y_cls9=torch.tensor(int(y_cls)),
                        y_ret=torch.tensor([float(y_r)], dtype=torch.float32))
            if self.use_structural_features:
                deg = torch.bincount(ei[0], minlength=x.size(0)).float().unsqueeze(1)
                data.x = torch.cat([data.x, deg], dim=1)
            data_list.append(data)
        return data_list