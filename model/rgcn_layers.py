#from torch_rgcn.utils import *
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch import nn
import math
import torch
import torch.nn.functional as F

    
class NodeClassifier(nn.Module):
    """ Node classification with R-GCN message passing """
    def __init__(self,
                 #triples=None,
                 nnodes=None,
                 nrel=None,
                 nfeat=None,
                 nhid=16,
                 nlayers=2,
                 nclass=None,
                 edge_dropout=None,
                 decomposition=None,
                 nemb=None):
        super(NodeClassifier, self).__init__()

        self.nlayers = nlayers
        self.nemb = nemb
        self.nnodes = nnodes
        self.nrel = nrel

        #assert (triples is not None or nnodes is not None or nrel is not None or nclass is not None), \
        #    "The following must be specified: triples, number of nodes, number of relations and number of classes!"
        assert 0 < nlayers < 3, "Only supports the following number of RGCN layers: 1 and 2."

        if nlayers == 1:
            nhid = nclass

        if nlayers == 2:
            assert nhid is not None, "Number of hidden layers not specified!"

        #triples = torch.tensor(triples, dtype=torch.long)


        self.rgcn_no_hidden = RelationalGraphConvolutionNC(
            #triples=self.triples_plus,
            num_nodes=nnodes,
            num_relations=nrel * 2 + 1,
            in_features=nfeat,
            out_features=nhid,
            edge_dropout=edge_dropout,
            decomposition=decomposition,
            vertical_stacking=False
        )
        if nlayers == 2:
            self.rgc1 = RelationalGraphConvolutionNC(
                #triples=self.triples_plus,
                num_nodes=nnodes,
                num_relations=nrel * 2 + 1,
                in_features=nhid,
                out_features=nclass,
                edge_dropout=edge_dropout,
                decomposition=decomposition,
                vertical_stacking=False
            )
        self.dense = nn.Linear(nhid, nclass, bias= False)

    def forward(self, triples_plus):
        """ Embed relational graph and then compute class probabilities """
        activation = {}
        if self.nemb is not None:
             x, adj, input = self.rgcn_no_hidden(self.nemb, triples_plus)
        else:
            x, adj, input = self.rgcn_no_hidden(None, triples_plus)
        if input is not None:
            activation['input'] = input.detach()
        if self.nlayers == 2:
            x = F.relu(x)
            activation['rgcn_no_hidden'] = x.detach()
            x, adj, input = self.rgc1(x, triples_plus)
            #x = F.relu(x)
            #x = self.dense(x)
            x = F.softmax(x)
            activation['rgc1'] = x.detach()
        return x, adj, activation


class RelationalGraphConvolutionNC(Module):
    """
    Relational Graph Convolution (RGC) Layer for Node Classification
    (as described in https://arxiv.org/abs/1703.06103)
    """
    def __init__(self,
                 #triples=None,
                 num_nodes=None,
                 num_relations=None,
                 in_features=None,
                 out_features=None,
                 edge_dropout=None,
                 edge_dropout_self_loop=None,
                 bias=True,
                 decomposition=None,
                 vertical_stacking=False,
                 diag_weight_matrix=False,
                 reset_mode='glorot_uniform'):
        super(RelationalGraphConvolutionNC, self).__init__()

        #assert (triples is not None or num_nodes is not None or num_relations is not None or out_features is not None), \
        #    "The following must be specified: triples, number of nodes, number of relations and output dimension!"

        # If featureless, use number of nodes instead as input dimension
        in_dim = in_features if in_features is not None else num_nodes
        out_dim = out_features

        # Unpack arguments
        weight_decomp = decomposition['type'] if decomposition is not None and 'type' in decomposition else None
        num_bases = decomposition['num_bases'] if decomposition is not None and 'num_bases' in decomposition else None
        num_blocks = decomposition['num_blocks'] if decomposition is not None and 'num_blocks' in decomposition else None

        #self.triples = triples
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.in_features = in_features
        self.out_features = out_features
        self.weight_decomp = weight_decomp
        self.num_bases = num_bases
        self.num_blocks = num_blocks
        self.vertical_stacking = vertical_stacking
        self.diag_weight_matrix = diag_weight_matrix
        self.edge_dropout = edge_dropout
        self.edge_dropout_self_loop = edge_dropout_self_loop
        self.node_emb = torch.nn.Parameter(torch.FloatTensor(num_nodes, in_dim))
        torch.nn.init.kaiming_normal_(self.node_emb, mode='fan_in')
        # If this flag is active, the weight matrix is a diagonal matrix
        if self.diag_weight_matrix:
            self.weights = torch.nn.Parameter(torch.empty((self.num_relations, self.in_features)), requires_grad=True)
            self.out_features = self.in_features
            self.weight_decomp = None
            bias = False

        # Instantiate weights
        elif self.weight_decomp is None:
            self.weights = Parameter(torch.FloatTensor(num_relations, in_dim, out_dim))
        elif self.weight_decomp == 'basis':
            # Weight Regularisation through Basis Decomposition
            assert num_bases > 0, \
                'Number of bases should be set to higher than zero for basis decomposition!'
            self.bases = Parameter(torch.FloatTensor(num_bases, in_dim, out_dim))
            self.comps = Parameter(torch.FloatTensor(num_relations, num_bases))
        elif self.weight_decomp == 'block':
            # Weight Regularisation through Block Diagonal Decomposition
            assert self.num_blocks > 0, \
                'Number of blocks should be set to a value higher than zero for block diagonal decomposition!'
            assert in_dim % self.num_blocks == 0 and out_dim % self.num_blocks == 0,\
                f'For block diagonal decomposition, input dimensions ({in_dim}, {out_dim}) must be divisible ' \
                f'by number of blocks ({self.num_blocks})'
            self.blocks = nn.Parameter(
                torch.FloatTensor(num_relations, self.num_blocks, in_dim // self.num_blocks, out_dim // self.num_blocks))
        else:
            raise NotImplementedError(f'{self.weight_decomp} decomposition has not been implemented')

        # Instantiate biases
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else: 
            self.register_parameter('bias', None)
            
        self.reset_parameters(reset_mode)
    
    def reset_parameters(self, reset_mode='glorot_uniform'):
        """ Initialise biases and weights (glorot_uniform or uniform) """

        if reset_mode == 'glorot_uniform':
            if self.weight_decomp == 'block':
                nn.init.xavier_uniform_(self.blocks, gain=nn.init.calculate_gain('relu'))
            elif self.weight_decomp == 'basis':
                nn.init.xavier_uniform_(self.bases, gain=nn.init.calculate_gain('relu'))
                nn.init.xavier_uniform_(self.comps, gain=nn.init.calculate_gain('relu'))
            else:
                nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain('relu'))

            if self.bias is not None:
                torch.nn.init.zeros_(self.bias)
        elif reset_mode == 'schlichtkrull':
            if self.weight_decomp == 'block':
                nn.init.xavier_uniform_(self.blocks, gain=nn.init.calculate_gain('relu'))
            elif self.weight_decomp == 'basis':
                nn.init.xavier_uniform_(self.bases, gain=nn.init.calculate_gain('relu'))
                nn.init.xavier_uniform_(self.comps, gain=nn.init.calculate_gain('relu'))
            else:
                nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain('relu'))

            if self.bias is not None:
                torch.nn.init.zeros_(self.bias)
        elif reset_mode == 'uniform':
            stdv = 1.0 / math.sqrt(self.weights.size(1))
            if self.weight_decomp == 'block':
                self.blocks.data.uniform_(-stdv, stdv)
            elif self.weight_decomp == 'basis':
                self.bases.data.uniform_(-stdv, stdv)
                self.comps.data.uniform_(-stdv, stdv)
            else:
                self.weights.data.uniform_(-stdv, stdv)

            if self.bias is not None:
                self.bias.data.uniform_(-stdv, stdv)
        else:
            raise NotImplementedError(f'{reset_mode} parameter initialisation method has not been implemented')

    def forward(self, features, triples):
        """ Perform a single pass of message propagation """

        #assert (features is None) == (self.in_features is None), "in_features not provided!"
        triples = torch.tensor(triples, dtype=torch.long)
        in_dim = self.in_features if self.in_features is not None else self.num_nodes
        out_dim = self.out_features
        edge_dropout = self.edge_dropout
        weight_decomp = self.weight_decomp
        num_nodes = self.num_nodes
        num_relations = self.num_relations
        vertical_stacking = self.vertical_stacking
        general_edge_count = int((triples.size(0) - num_nodes)/2)
        self_edge_count = num_nodes

        # Choose weights
        if weight_decomp is None:
            weights = self.weights
        elif weight_decomp == 'basis':
            weights = torch.einsum('rb, bio -> rio', self.comps, self.bases)
        elif weight_decomp == 'block':
            weights = block_diag(self.blocks)
        else:
            raise NotImplementedError(f'{weight_decomp} decomposition has not been implemented')

        # Determine whether to use cuda or not
        if weights.is_cuda:
            device = 'cuda'
        else:
            device = 'cpu'

        # Stack adjacency matrices either vertically or horizontally
        adj_indices, adj_size = stack_matrices(
            triples,
            num_nodes,
            num_relations,
            vertical_stacking=vertical_stacking,
            device=device
        )
        num_triples = adj_indices.size(0)
        vals = torch.ones(num_triples, dtype=torch.float, device=device)

        # Apply normalisation (vertical-stacking -> row-wise rum & horizontal-stacking -> column-wise sum)
        sums = sum_sparse(adj_indices, vals, adj_size, row_normalisation=vertical_stacking, device=device)
        if not vertical_stacking:
            # Rearrange column-wise normalised value to reflect original order (because of transpose-trick)
            n = general_edge_count
            i = self_edge_count
            sums = torch.cat([sums[n:2 * n], sums[:n], sums[-i:]], dim=0)

        vals = vals / sums

        # Construct adjacency matrix
        if device == 'cuda':
            adj = torch.cuda.sparse.FloatTensor(indices=adj_indices.t(), values=vals, size=adj_size)
        else:
            adj = torch.sparse.FloatTensor(indices=adj_indices.t(), values=vals, size=adj_size)

        if self.diag_weight_matrix:
            assert weights.size() == (num_relations, in_dim)
        else:
            assert weights.size() == (num_relations, in_dim, out_dim)

        if features is None:
            # Message passing if no features are given
            #output = torch.mm(adj, weights.view(num_relations * in_dim, out_dim))
            input = self.node_emb
            fw = torch.einsum('ni, rio -> rno', input, weights).contiguous()
            output = torch.mm(adj, fw.view(self.num_relations * self.num_nodes, out_dim))
        elif self.diag_weight_matrix:
            fw = torch.einsum('ij,kj->kij', features, weights)
            fw = torch.reshape(fw, (self.num_relations * self.num_nodes, in_dim))
            output = torch.mm(adj, fw)
        elif self.vertical_stacking:
            # Message passing if the adjacency matrix is vertically stacked
            af = torch.spmm(adj, features)
            af = af.view(self.num_relations, self.num_nodes, in_dim)
            output = torch.einsum('rio, rni -> no', weights, af)
        else:
            # Message passing if the adjacency matrix is horizontally stacked
            fw = torch.einsum('ni, rio -> rno', features, weights).contiguous()
            output = torch.mm(adj, fw.view(self.num_relations * self.num_nodes, out_dim))

        assert output.size() == (self.num_nodes, out_dim)
        
        if self.bias is not None:
            output = torch.add(output, self.bias)
        if features is None:
            return output, adj, input
        return output, adj, None


from math import floor, sqrt
import random
import torch


def schlichtkrull_std(shape, gain):
    """
    a = \text{gain} \times \frac{3}{\sqrt{\text{fan\_in} + \text{fan\_out}}}
    """
    fan_in, fan_out = shape[0], shape[1]
    return gain * 3.0 / sqrt(float(fan_in + fan_out))

def schlichtkrull_normal_(tensor, shape, gain=1.):
    """Fill the input `Tensor` with values according to the Schlichtkrull method, using a normal distribution."""
    std = schlichtkrull_std(shape, gain)
    with torch.no_grad():
        return tensor.normal_(0.0, std)

def schlichtkrull_uniform_(tensor, gain=1.):
    """Fill the input `Tensor` with values according to the Schlichtkrull method, using a uniform distribution."""
    std = schlichtkrull_std(tensor, gain)
    with torch.no_grad():
        return tensor.uniform_(-std, std)

def select_b_init(init):
    """Return functions for initialising biases"""
    init = init.lower()
    if init in ['zeros', 'zero', 0]:
        return torch.nn.init.zeros_
    elif init in ['ones', 'one', 1]:
        return torch.nn.init.ones_
    elif init == 'uniform':
        return torch.nn.init.uniform_
    elif init == 'normal':
        return torch.nn.init.normal_
    else:
        raise NotImplementedError(f'{init} initialisation has not been implemented!')

def select_w_init(init):
    """Return functions for initialising weights"""
    init = init.lower()
    if init in ['glorot-uniform', 'xavier-uniform']:
        return torch.nn.init.xavier_uniform_
    elif init in ['glorot-normal', 'xavier-normal']:
        return torch.nn.init.xavier_normal_
    elif init == 'schlichtkrull-uniform':
        return schlichtkrull_uniform_
    elif init == 'schlichtkrull-normal':
        return schlichtkrull_normal_
    elif init in ['normal', 'standard-normal']:
        return torch.nn.init.normal_
    elif init == 'uniform':
        return torch.nn.init.uniform_
    else:
        raise NotImplementedError(f'{init} initialisation has not been implemented!')


def sum_sparse(indices, values, size, row_normalisation=True, device='cpu'):
    """
    Sum the rows or columns of a sparse matrix, and redistribute the
    results back to the non-sparse row/column entries
    Arguments are interpreted as defining sparse matrix.

    Source: https://github.com/pbloem/gated-rgcn/blob/1bde7f28af8028f468349b2d760c17d5c908b58b/kgmodels/util/util.py#L304
    """

    assert len(indices.size()) == len(values.size()) + 1

    k, r = indices.size()

    if not row_normalisation:
        # Transpose the matrix for column-wise normalisation
        indices = torch.cat([indices[:, 1:2], indices[:, 0:1]], dim=1)
        size = size[1], size[0]

    ones = torch.ones((size[1], 1), device=device)
    if device == 'cuda':
        values = torch.cuda.sparse.FloatTensor(indices.t(), values, torch.Size(size))
    else:
        values = torch.sparse.FloatTensor(indices.t(), values, torch.Size(size))
    sums = torch.spmm(values, ones)
    sums = sums[indices[:, 0], 0]

    return sums.view(k)




def add_inverse_and_self(triples, num_nodes, num_rels, device='cpu'):
    """ Adds inverse relations and self loops to a tensor of triples """

    # Swap around head and tail. Create new relation ids for inverse relations.
    inverse_relations = torch.cat([triples[:, 2, None], triples[:, 1, None] + num_rels, triples[:, 0, None]], dim=1)
    assert inverse_relations.size() == triples.size()

    # Create a new relation id for self loop relation.
    all = torch.arange(num_nodes, device=device)[:, None]
    id  = torch.empty(size=(num_nodes, 1), device=device, dtype=torch.long).fill_(2*num_rels)
    self_loops = torch.cat([all, id, all], dim=1)
    assert self_loops.size() == (num_nodes, 3)

    # Note: Self-loops are appended to the end and this makes it easier to apply different edge dropout rates.
    return torch.cat([triples, inverse_relations, self_loops], dim=0)

def stack_matrices(triples, num_nodes, num_rels, vertical_stacking=True, device='cpu'):
    """
    Computes a sparse adjacency matrix for the given graph (the adjacency matrices of all
    relations are stacked vertically).
    """
    assert triples.dtype == torch.long

    r, n = num_rels, num_nodes
    size = (r * n, n) if vertical_stacking else (n, r * n)

    fr, to = triples[:, 0], triples[:, 2]
    offset = triples[:, 1] * n
    if vertical_stacking:
        fr = offset + fr
    else:
        to = offset + to

    indices = torch.cat([fr[:, None], to[:, None]], dim=1).to(device)

    assert indices.size(0) == triples.size(0)
    assert indices[:, 0].max() < size[0], f'{indices[0, :].max()}, {size}, {r}'
    assert indices[:, 1].max() < size[1], f'{indices[1, :].max()}, {size}, {r}'

    return indices, size

def block_diag(m):
    """
    Source: https://gist.github.com/yulkang/2e4fc3061b45403f455d7f4c316ab168
    Make a block diagonal matrix along dim=-3
    EXAMPLE:
    block_diag(torch.ones(4,3,2))
    should give a 12 x 8 matrix with blocks of 3 x 2 ones.
    Prepend batch dimensions if needed.
    You can also give a list of matrices.
    """

    device = 'cuda' if m.is_cuda else 'cpu'  # Note: Using cuda status of m as proxy to decide device

    if type(m) is list:
        m = torch.cat([m1.unsqueeze(-3) for m1 in m], -3)

    dim = m.dim()
    n = m.shape[-3]

    siz0 = m.shape[:-3]
    siz1 = m.shape[-2:]

    m2 = m.unsqueeze(-2)

    eye = attach_dim(torch.eye(n, device=device).unsqueeze(-2), dim - 3, 1)

    return (m2 * eye).reshape(
        siz0 + torch.Size(torch.tensor(siz1) * n)
    )

def attach_dim(v, n_dim_to_prepend=0, n_dim_to_append=0):
    return v.reshape(torch.Size([1] * n_dim_to_prepend) + v.shape + torch.Size([1] * n_dim_to_append))
