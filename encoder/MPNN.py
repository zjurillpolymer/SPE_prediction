
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import MessagePassing, global_mean_pool


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
        graph_embedding = global_mean_pool(x, batch)

        return graph_embedding





