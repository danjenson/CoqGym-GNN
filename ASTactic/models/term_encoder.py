from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
import torch_scatter
from torch import Tensor
from torch.nn import Linear, Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import (Adj, NoneType, OptPairTensor, OptTensor,
                                    Size)
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from torch_geometric.graphgym.models.encoder import IntegerFeatureEncoder
from torch_sparse import SparseTensor, set_diag

from .non_terminals import nonterminals


class TermEncoder(torch.nn.Module):  # StackGNN
    def __init__(self, opts):
        super(TermEncoder, self).__init__()
        self.opts = opts

        self.input_dim = opts.nonterminals_feature_dim
        self.hidden_dim = opts.term_embedding_dim
        self.output_dim = opts.term_embedding_dim

        self.feature_encoder = IntegerFeatureEncoder(self.input_dim, len(nonterminals))

        conv_model = self.build_conv_model(opts.model_type)
        self.convs = nn.ModuleList()
        self.convs.append(
            conv_model(self.input_dim, self.hidden_dim // opts.heads, heads=opts.heads)
        )
        assert opts.num_layers >= 1, "Number of layers is not >=1"
        for i in range(opts.num_layers - 1):
            self.convs.append(
                conv_model(
                    self.hidden_dim, self.hidden_dim // opts.heads, heads=opts.heads
                )
            )

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Dropout(opts.dropout),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

        self.dropout = opts.dropout
        self.num_layers = opts.num_layers

        # pooling
        self.pool = pyg_nn.global_max_pool  # self.pool = pyg_nn.dense_diff_pool
        self.post_pool = nn.Linear(self.output_dim, self.output_dim)

    def build_conv_model(self, model_type):
        if model_type == "GraphSage":
            return GraphSage
        elif model_type == "GAT":
            # When applying GAT with num heads > 1, you need to modify the
            # input and output dimension of the conv layers (self.convs),
            # to ensure that the input dim of the next layer is num heads
            # multiplied by the output dim of the previous layer.
            # HINT: In case you want to play with multiheads, you need to change the for-loop that builds up self.convs to be
            # self.convs.append(conv_model(hidden_dim * num_heads, hidden_dim)),
            # and also the first nn.Linear(hidden_dim * num_heads, hidden_dim) in post-message-passing.
            return GAT
        raise ValueError("Unknown model type: {}".format(model_type))

    def forward(self, proof_step):
        """"""
        proof_step = self.feature_encoder(proof_step)
        x, edge_index, batch = proof_step.x, proof_step.edge_index, proof_step.batch

        # preprocess x
        batch = batch.to(self.opts.device)
        edge_index = edge_index.to(self.opts.device)
        x = x.to(self.opts.device)

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.post_mp(x)

        # differential pooling
        x = self.pool(x, batch)

        return self.post_pool(x)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)


class GraphSage(MessagePassing):
    """"""

    def __init__(self, in_channels, out_channels, normalize=True, bias=False, **kwargs):
        super(GraphSage, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.node_dim = 0
        self.out_channels = out_channels
        self.normalize = normalize

        self.lin_l = torch.nn.Linear(in_channels, out_channels, bias=bias)
        self.lin_r = torch.nn.Linear(in_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x, edge_index, size=None):
        hv = self.propagate(edge_index=edge_index, x=(x, x), size=size)
        out = self.lin_l(x) + self.lin_r(hv)

        if self.normalize:  # L-2 normalization if set to true
            out = torch.nn.functional.normalize(out, p=2.0)

        return out

    def message(self, x_j):
        return x_j

    def aggregate(self, inputs, index, dim_size=None):
        node_dim = self.node_dim  # axis to index number of nodes
        out = torch_scatter.scatter(
            inputs, index=index, dim=node_dim, dim_size=dim_size, reduce="mean"
        )

        return out


class GAT(MessagePassing):
    """"""

    def __init__(
        self,
        in_channels,
        out_channels,
        heads=2,
        negative_slope=0.2,
        dropout=0.0,
        **kwargs
    ):
        super(GAT, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin_l = torch.nn.Linear(in_channels, out_channels * heads, bias=False)
        self.lin_r = self.lin_l

        self.att_l = nn.Parameter(torch.zeros(self.heads, self.out_channels))
        self.att_r = nn.Parameter(torch.zeros(self.heads, self.out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)

    def forward(self, x, edge_index, size=None):
        """"""
        H, C = self.heads, self.out_channels

        # pre-processing
        x_l = self.lin_l(x).view(-1, H, C)
        x_r = self.lin_r(x).view(-1, H, C)

        alpha_l = (x_l * self.att_l).sum(dim=-1)
        alpha_r = (x_r * self.att_r).sum(dim=-1)

        # message propagation
        out = self.propagate(
            edge_index=edge_index, alpha=(alpha_l, alpha_r), x=(x_l, x_r), size=size
        )

        # post-processing
        out = out.view(-1, H * C)

        return out

    def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i):
        """"""
        # attention weights
        alpha = F.leaky_relu(alpha_i + alpha_j, self.negative_slope)

        # softmax
        alpha = pyg_utils.softmax(alpha, index=index, ptr=ptr, num_nodes=size_i)

        # apply dropout
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # multiply embeddings
        out = alpha.unsqueeze(-1) * x_j

        return out

    def aggregate(self, inputs, index, dim_size=None):
        """"""
        out = torch_scatter.scatter(
            inputs, index=index, dim=self.node_dim, dim_size=dim_size, reduce="sum"
        )

        return out
