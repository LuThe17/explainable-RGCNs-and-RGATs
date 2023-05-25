import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module

from torch_geometric.nn.conv import MessagePassing
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, degree
from torch_geometric.typing import Adj, Size, OptTensor, Tensor
from torch_sparse import SparseTensor, set_diag

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super(GATLayer, self).__init__()

        self.dropout = dropout
        self.alpha = alpha

        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, h, adj):
        Wh = self.fc(h)
        N = Wh.size()[0]

        a_input = torch.cat([Wh.repeat(1, N).view(N*N, -1), Wh.repeat(N, 1)], dim=1).view(N, -1, 2*Wh.size(-1))
        e = F.leaky_relu(torch.matmul(a_input, self.a), negative_slope=self.alpha)
        attention = F.softmax(e, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training).squeeze(dim=2)
        h_prime = torch.matmul(attention, Wh)

        return h_prime

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.6, alpha=0.2):
        super(GAT, self).__init__()

        self.dropout = dropout

        self.gat1 = GATLayer(nfeat, nhid, dropout=dropout, alpha=alpha)
            #(nfeat, nhid, dropout=dropout, alpha=alpha)
        self.gat2 = GATLayer(nhid, nclass, dropout=dropout, alpha=alpha)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gat1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gat2(x, adj)
        x = F.log_softmax(x, dim=1)

        return x

class RGATLayer(nn.Module):
    """Layer implementing Relational Graph Attention of
    https://openreview.net/forum?id=Bklzkh0qFm with sparse supports.

    Must be called using both inputs and support:
        inputs = get_inputs()
        support = get_support()

        rgat_layer = RelationalGraphAttention(...)
        outputs = rgat_layer(inputs=inputs, support=support)

    Has alias of `RGAT`.

    Arguments:
        units (int): The dimensionality of the output space.
        relations (int): The number of relation types the layer will handle.
        heads (int): The number of attention heads to use (see
            https://arxiv.org/abs/1710.10903). Defaults to `1`.
        head_aggregation (str): The attention head aggregation method to use
            (see https://arxiv.org/abs/1710.10903). Can be one of `'mean'` or
            `'concat'`. Defaults to `'mean'`.
        attention_mode (str): The relational attention mode to to use (see
            https://openreview.net/forum?id=Bklzkh0qFm). Can be one of `'argat'`
            or `'wirgat'`. Defaults to `'argat'`.
        attention_style (str): The different types of attention to use. To use
            the transformer style multiplicative attention, set to `'dot'`.  To
            use the GAT style additive attention set to `'sum'`. Defaults to
            `'sum'`.
        attention_units (int): The dimensionality of the attention space. If
            using `'sum'` style attention, this must be set to `1`.
        attn_use_edge_features (bool): Whether the layer can use edge features.
            Defaults to `False`.
        kernel_basis_size (int): The number of basis kernels to create the
            relational kernels from, i.e. W_r = sum_i c_{i,r} W'_i, where
            r = 1, 2, ..., relations, and i = 1, 2 ..., kernel_basis_size.
            If `None` (default), these is no basis decomposition.
        attn_kernel_basis_size (int): The number of basis kernels to create the
            relational attention kernels from. Defaults to `None`.
        activation (callable): Activation function. Set it to `None` to maintain
            a linear activation.
        attn_activation (callable): Activation function to apply to the
            attention logits prior to feeding to softmax. Defaults to the leaky
            relu in https://arxiv.org/abs/1710.10903, however, when using
            `'dot'` style attention, this can be set to `None`.
        use_bias (bool): Whether the layer uses a bias. Defaults to `False`.
        batch_normalisation (bool): Whether the layer uses batch normalisation.
            Defaults to `False`.
        kernel_initializer (callable): Initializer function for the graph
            convolution weight matrix. If None (default), weights are
            initialized using the `glorot_uniform` initializer.
        bias_initializer (callable): Initializer function for the bias. Defaults
            to `zeros`.
        attn_kernel_initializer (callable): Initializer function for the
            attention weight matrix. If None (default), weights are
            initialized using the `glorot_uniform` initializer.
        kernel_regularizer (callable): Regularizer function for the graph
            convolution weight matrix. Defaults to `None`.
        bias_regularizer (callable): Regularizer function for the bias. Defaults
            to `None`.
        attn_kernel_regularizer (callable): Regularizer function for the graph
            attention weight matrix. Defaults to `None`.
        activity_regularizer (callable): Regularizer function for the output.
            Defaults to `None`.
        feature_dropout (float): The dropout rate for node feature
            representations, between 0 and 1. E.g. rate=0.1 would drop out 10%
            of node input units.
        support_dropout (float): The dropout rate for edges in the support,
            between 0 and 1. E.g. rate=0.1 would drop out 10%
            of the edges in the support.
        edge_feature_dropout (float): The dropout rate for edge feature
            representations, between 0 and 1.
        name (string): The name of the layer. Defaults to
            `rgat`.

    """
    def __init__(self,
                 units,
                 relations,
                 heads=1,
                 head_aggregation= 'mean',#HeadAggregation.MEAN,
                 attention_mode= 'ARGAT', #AttentionModes.ARGAT,
                 attention_style='sum',#AttentionStyles.SUM,
                 attention_units=1,
                 attn_use_edge_features=False,
                 kernel_basis_size=None,
                 attn_kernel_basis_size=None,
                 activation=None,
                 attn_activation=F.leaky_relu,
                 use_bias=False,
                 batch_normalisation=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 attn_kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 attn_kernel_regularizer=None,
                 activity_regularizer=None,
                 feature_dropout=None,
                 support_dropout=None,
                 edge_feature_dropout=None,
                 name='rgat',
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(RGATLayer, self).__init__(
            activity_regularizer=activity_regularizer,
            name=name, **kwargs)

        self.units = int(units)
        self.relations = int(relations)
        self.heads = int(heads)
        self.head_aggregation = head_aggregation
        self.attention_mode = attention_mode
        self.attention_style = attention_style
        self.attention_units = attention_units
        self.attn_use_edge_features = attn_use_edge_features

        self.kernel_basis_size = (int(kernel_basis_size)
                                  if kernel_basis_size else None)
        self.attn_kernel_basis_size = (int(attn_kernel_basis_size)
                                       if attn_kernel_basis_size else None)

        self.activation = activation
        self.attn_activation = attn_activation

        self.use_bias = use_bias
        self.batch_normalisation = batch_normalisation

        # if self.batch_normalisation: # is false
        #     self.batch_normalisation_layer = BatchNormalization()

        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.attn_kernel_initializer = attn_kernel_initializer

        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.attn_kernel_regularizer = attn_kernel_regularizer

        self.feature_dropout = feature_dropout
        self.support_dropout = support_dropout
        self.edge_feature_dropout = edge_feature_dropout

        self.supports_masking = True
        self.input_spec = InputSpec(min_ndim=2)

        self.dense_layer = rgat_layers.BasisDecompositionDense(
            units=self.relations * self.heads * self.units,
            basis_size=self.kernel_basis_size,
            coefficients_size=self.relations * self.heads,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            name=name + '_basis_decomposition_dense',
            **kwargs)
        self.attention_logits = RelationalGraphAttentionLogits(
            relations=self.relations,
            heads=self.heads,
            attention_style=self.attention_style,
            attention_units=self.attention_units,
            basis_size=self.attn_kernel_basis_size,
            activation=self.attn_activation,
            use_edge_features=self.attn_use_edge_features,
            kernel_initializer=self.attn_kernel_initializer,
            kernel_regularizer=self.attn_kernel_regularizer,
            feature_dropout=self.feature_dropout,
            edge_feature_dropout=self.edge_feature_dropout,
            batch_normalisation=self.batch_normalisation,
            name="logits",
            **kwargs)
        # if self.head_aggregation == HeadAggregation.PROJECTION:
        #     self.projection_layer = keras_layers.Dense(
        #         units=self.units,
        #         use_bias=False,
        #         kernel_initializer=self.kernel_initializer,
        #         kernel_regularizer=self.kernel_regularizer,
        #         name="projection",
        #         **kwargs)
        # if self.batch_normalisation:
        #     self.batch_normalisation_layer = BatchNormalization()

# class GAT(MessagePassing):
#     def __init__(self, in_features, out_features, heads=1, negative_slope=0.2, dropout=0.6, bias=True, concat: bool=True, add_self_loops: bool=True):
#         super(GAT, self).__init__(aggr='add')

#         self.in_features = in_features
#         self.out_features = out_features
#         self.heads = heads
#         self.negative_slope = negative_slope
#         self.dropout = dropout
#         self.concat = concat
#         self.add_self_loops = add_self_loops

#         if isinstance(in_features, int):
#             self.lin_l = nn.Linear(in_features, heads*out_features, bias=bias)
#             self.lin_r = nn.Linear(in_features, heads*out_features, bias=bias)
#         else:
#             self.lin_l = nn.Linear(in_features[0], heads*out_features, bias=bias)
#             self.lin_r = nn.Linear(in_features[1], heads*out_features, bias=bias)
        
        
#         self.att = nn.Parameter(torch.Tensor(1, heads, out_features))
#         #self.weight = nn.Parameter(torch.Tensor(in_features, heads*out_features))
        

#         if bias and not self.concat:
#             self.bias = nn.Parameter(torch.Tensor(out_features))
#         else:
#             self.register_parameter('bias', None)

#         self.reset_parameters()

#     def reset_parameters(self):
#         glorot(self.lin_l.weight)
#         glorot(self.lin_r.weight)
#         glorot(self.att)
#         zeros(self.bias)
#         # nn.init.xavier_uniform_(self.weight.data, gain=1.414)
#         # nn.init.xavier_uniform_(self.att.data, gain=1.414)
#         # if self.bias is not None:
#         #     self.bias.data.fill_(0)

#     def forward(self, x, edge_index: Adj, size: Size=None, return_attention_weights=None):

#         h, f = self.heads, self.out_features

#         x_l: OptTensor = None
#         x_r: OptTensor = None

#         if isinstance(x, Tensor):
#             assert x.dim() == 2, 'Static graphs not supported in `GATLayer`.'
#             x_l = x_r = self.lin_l(x).view(-1, h, f)
#         else:
#             x_l, x_r = x[0], x[1]
#             x_l = self.lin_l(x_l).view(-1, h, f)
#             if x_r is not None:
#                 x_r = self.lin_r(x_r).view(-1, h, f)

#         if self.add_self_loops:
#             if isinstance(edge_index, Tensor):
#                 num_nodes = x_l.size(0)
#                 if x_r is not None:
#                     num_nodes = min(num_nodes, x_r.size(0))
#                 if size is not None:
#                     num_nodes = min(size[0], size[1])
#                 edge_index, _ = remove_self_loops(edge_index)
#                 edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
#             elif isinstance(edge_index, SparseTensor):
#                 edge_index = set_diag(edge_index)

#         out = self.propagate(edge_index, x=(x_l, x_r), size=size)
#         alpha = self._alpha

#         if self.concat:
#             out = out.view(-1, self.heads*self.out_features)
#         else:
#             out = out.mean(dim=1)
        
#         if self.bias is not None:
#             out = out + self.bias
        
#         if isinstance(return_attention_weights, bool):
#             assert alpha is not None
#             if isinstance(edge_index, Tensor):
#                 return out, (edge_index, alpha)
#             elif isinstance(edge_index, SparseTensor):
#                 return out, edge_index.set_value(alpha, layout='coo')
#         else:
#             return out
            
        
#     def message(self, x_i, x_j, index, ptr, size_i):
#         x = x_i + x_j
#         x = F.leaky_relu(x, self.negative_slope)

#         alpha = (x * self.att).sum(dim=-1)
#         alpha = softmax(alpha, index, ptr, size_i)
#         self._alpha = alpha

#         alpha = F.dropout(alpha, p=self.dropout, training=self.training)

#         return x_j*alpha.unsqueeze(-1)

#     def update(self, aggr_out):
#         if self.bias is not None:
#             aggr_out = aggr_out + self.bias

#         return aggr_out

#     def __repr__(self):
#         return '{}({}, {}, heads={})'.format(self.__class__.__name__, self.in_channels, self.out_channels, self.heads)

# class GraphAttentionLayer(Module):
#     def __init__(self,
#                  in_features=None,
#                  out_features=None,
#                  nheads=None,
#                  alpha=None,
#                  dropout=None,
#                  concat=True):
#         super(GraphAttentionLayer, self).__init__()


#         self.dropout = dropout
#         self.in_features = in_features
#         self.out_features = out_features
#         self.nheads = nheads
#         self.alpha = alpha
#         self.concat = concat

#         self.project = nn.Linear(in_features, out_features*nheads)

#         #self.weights = torch.nn.Parameter(torch.empty((in_features, out_features)))
#         nn.init.xavier_uniform_(self.projection.weight.data, gain=1.414)
#         self.a = nn.Parameter(torch.empty(size=(nheads,2*out_features)))
#         nn.init.xavier_uniform_(self.a.data, gain=1.414)
#         self.leakyrelu = nn.LeakyReLU(alpha)

#     def forward(self, h, adj):
#         Wh = torch.mm(h, self.weights)
#         N = h.size()[0]
#         #a_input = torch.cat([h.repeat(1,N).view(N * N, -1), h.repeat(N,1)], dim=1).view(N, -1, 2 * self.out_features)#self._prepare_attentional_mechanism_input(Wh)
#         e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

#         zero_vec = -9e15*torch.ones_like(e)
#         attention = torch.where(adj > 0, e, zero_vec)
#         attention = F.softmax(attention, dim=1)
#         attention = F.dropout(attention, self.dropout, training=self.training)
#         h_prime = torch.matmul(attention, Wh)

#         if self.concat:
#             return F.elu(h_prime)
#         else:
#             return h_prime
        
#     # def _prepare_attentional_mechanism_input(self, Wh):

#     #     Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])#.transpose(0, 1))
#     #     Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])#.transpose(0, 1))
#     #     e = Wh1 + Wh2.T
#     #     return self.leakyrelu(e)
    
#     def __repr__(self):
#          return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

# class GAT(Module):
#     def __init__(self, numlayers, nheads, nfeat_per_layer, dropout, alpha):
#         super(GAT, self).__init__()
#         self.dropout = dropout

#         self.gat1 = GraphAttentionLayer(
#             in_features = nfeat, 
#             out_features= nhid,
#             heads = nheads, 
#             alpha = alpha, 
#             dropout=0.6, 
#             concat=True)
        
#         self.gat2 = GraphAttentionLayer(
#             in_features = nfeat, 
#             out_features = nclass, 
#             heads = 1
#             dropout=dropout, 
#             alpha=alpha, 
#             concat=False)
#         # self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
#         # for i, attention in enumerate(self.attentions):
#         #     self.add_module('attention_{}'.format(i), attention)

#         # self.out_att = GraphAttentionLayer(nhid*nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

#     def forward(self, x, adj):
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.att1(x, adj)
#         x = F.elu(x)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.att2(x, adj)
#         return F.log_softmax(x, dim=1)   

#         # x = F.dropout(x, self.dropout, training=self.training)
#         # x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
#         # x = F.dropout(x, self.dropout, training=self.training)
#         # x = F.elu(self.out_att(x, adj))
        