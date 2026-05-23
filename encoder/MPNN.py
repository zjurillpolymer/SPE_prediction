import rdkit
from rdkit import RDLogger, Chem
RDLogger.logger().setLevel(RDLogger.ERROR)

import pandas as pd
import numpy as np
from torch import nn
import molecule_to_graph
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_add_pool
from torch import optim
import os


df=pd.read_csv('../data/clean_train_data.csv')
np.random.seed(42)
n=len(df)
idx=np.random.permutation(n)
n_train=int(n*0.8)
n_valid=int(n*0.1)
train_idx = idx[:n_train]
valid_idx = idx[n_train:n_train + n_valid]
test_idx = idx[n_train + n_valid:]

train_df = df.iloc[train_idx]
valid_df = df.iloc[valid_idx]
test_df = df.iloc[test_idx]



class SPEdataset(Dataset):
    def __init__(self, df,task='conductivity',cache_path="graph_cache.pt"):
        self.df = df.reset_index(drop=True)
        self.task = task
        self.cache_path = cache_path
        if os.path.exists(cache_path):
            print(f"加载缓存: {cache_path}")
            self.data_list = torch.load(cache_path, weights_only=False)
        else:
            print(f"首次构建图，缓存到: {cache_path}")
            self.data_list = self._preprocess_all_graphs()
            torch.save(self.data_list, cache_path)
            print(f"缓存完成！")


    def _preprocess_all_graphs(self):
        data_list = []
        total = len(self.df)
        fail = 0

        for idx in range(total):
            smiles = self.df['smiles'].iloc[idx]
            labels = self.df[self.task].iloc[idx]

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                fail += 1
                continue

            label = np.nan_to_num(labels, nan=0.0).astype(np.float32)

            data = molecule_to_graph.mol_to_pyg_graph(mol)
            data.y = torch.tensor(label, dtype=torch.float).view(1, -1)

            data_list.append(data)
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx].clone()



train_dataset=SPEdataset(train_df,task='conductivity',cache_path="graph_train_cache.pt")
valid_dataset=SPEdataset(valid_df,task='conductivity',cache_path="graph_valid_cache.pt")
test_dataset=SPEdataset(test_df,task='conductivity',cache_path="graph_test_cache.pt")

print(train_dataset[0])

BATCH_SIZE = 32

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


class MPNNEdgeLayer(MessagePassing):
    def __init__(self, in_channels, edge_dims,out_channels):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.mlp_msg = nn.Sequential(
            nn.Linear(2*in_channels+edge_dims, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

        self.mlp_update=nn.Sequential(
            nn.Linear(in_channels+out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )


    def forward(self, x, edge_index,edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)


    def message(self, edge_attr,x_i,x_j):
        msg=torch.cat([x_i,x_j,edge_attr],dim=-1)
        return self.mlp_msg(msg)

    def update(self,aggr_out,x):
        update_input = torch.cat([x, aggr_out], dim=-1)
        return self.mlp_update(update_input)


class MPNNEmbedding(nn.Module):
    def __init__(self,node_in_dims=31,edge_dim=6,hidden_dim=128,num_layers=6):
        super().__init__()
        self.num_layers = num_layers

        self.node_encoder=nn.Linear(node_in_dims,hidden_dim)

        self.mpnn_layers = nn.ModuleList([
            MPNNEdgeLayer(hidden_dim, edge_dim, hidden_dim) for _ in range(num_layers)
        ])

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.node_encoder(x)

        # 2. 多轮消息传递与更新
        for layer in self.mpnn_layers:
            x = layer(x, edge_index, edge_attr)
            x = F.relu(x)

        # 3. Readout / Pooling：把所有节点汇聚成一张图的向量
        graph_embedding = global_add_pool(x, batch)

        return graph_embedding





