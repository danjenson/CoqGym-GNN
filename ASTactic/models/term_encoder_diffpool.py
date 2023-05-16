from dataclasses import dataclass
from torch_geometric.nn.dense import dense_diff_pool, DenseGATConv
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_dense_adj, to_dense_batch
import torch.nn as nn
import torch.nn.functional as F
import warnings

# ignore tensorboard DeprecationWarnings
warnings.filterwarnings("ignore")


@dataclass(frozen=True)
class Cluster:
    n_nodes: int
    n_hops: int
    dense: bool


class TermEncoder(nn.Module):
    def __init__(
        self,
        opts,
        clusters=[
            Cluster(n_nodes=100, n_hops=6, dense=False),
            Cluster(n_nodes=10, n_hops=3, dense=True),
            Cluster(n_nodes=1, n_hops=3, dense=True),
        ],
        max_num_nodes=300,
        **kwargs,
    ):
        assert clusters[-1].n_nodes == 1, "Last cluster must reduce to 1 node."
        super(TermEncoder, self).__init__()
        self.max_num_nodes = max_num_nodes
        self.embed = nn.ModuleList()
        self.pool = nn.ModuleList()
        self.n_steps = len(clusters)
        for cluster in clusters:
            self.embed += [
                GATStack(
                    n_hops=cluster.n_hops,
                    dense=cluster.dense,
                    **kwargs,
                )
            ]
            self.pool += [
                GATStack(
                    n_hops=cluster.n_hops,
                    out_channels=cluster.n_nodes,
                    dense=cluster.dense,
                    postprocess=True,
                    **kwargs,
                )
            ]

    def forward(self, x, edge_index, batch):
        x = self.embed[0](x, edge_index)
        s = self.pool[0](x, edge_index)
        # TODO(danj): memory blowout w/o max_num_nodes, develop sparse_diff_pool?
        max_num_nodes = min(edge_index.max(), self.max_num_nodes)
        x, mask = to_dense_batch(x, batch, max_num_nodes=max_num_nodes)
        s, _ = to_dense_batch(s, batch, max_num_nodes=max_num_nodes)
        adj = to_dense_adj(edge_index, batch, max_num_nodes=max_num_nodes)
        x, adj, link_loss, entropy_loss = dense_diff_pool(x, adj, s, mask)
        for i in range(1, self.n_steps):
            x = self.embed[i](x, adj)
            s = self.pool[i](x, adj)
            x, adj, ll, el = dense_diff_pool(x, adj, s)
            link_loss += ll
            entropy_loss += el
        return x.squeeze(), link_loss, entropy_loss

    def loss(self, pred, label):
        return F.nll_loss(pred, label)


class GATStack(nn.Module):
    def __init__(
        self,
        in_channels=256,
        hidden_channels=256,
        out_channels=256,
        n_heads=4,
        n_hops=6,
        preprocess=False,
        postprocess=False,
        dense=False,
        **kwargs,
    ):
        super().__init__()
        Conv = DenseGATConv if dense else GATConv
        if preprocess:
            self.preprocess = nn.Linear(in_channels, hidden_channels)
        self.norms = nn.ModuleList()
        self.convs = nn.ModuleList()
        for _ in range(n_hops):
            self.norms += [nn.LayerNorm(hidden_channels)]
            self.convs += [
                Conv(hidden_channels, hidden_channels // n_heads, heads=n_heads)
            ]
        self.norms += [nn.LayerNorm(hidden_channels)]
        if postprocess:
            self.postprocess = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        if hasattr(self, "preprocess"):
            x = self.preprocess(x)
        for norm, conv in zip(self.norms, self.convs):
            x = norm(x)
            x = conv(x, edge_index)
            x = F.elu(x)
        x = self.norms[-1](x)
        if hasattr(self, "postprocess"):
            x = self.postprocess(x)
        return x
