import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import numpy as np
import scipy.sparse as sp
import math
import os

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
        "The following must be specified: triples, number of nodes, number of relations and output dimension!"

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

    def forward(self, x, triples):
        '''
        Perform a single pass of message propagation
        :num_nodes: Number of nodes in the graph
        :num_relations: Number of relations in the graph
        :param vals: Tensor of shape (num_triples, 1) containing the value of each triple
        :param n: Tensor of shape (num_triples, 1) containing the head of each triple
        :param fw: Tensor of shape (num_triples, 1) containing the relation of each triple
        :param sums: Tensor of shape (num_triples, 1) containing the tail of each triple
        :triples: Tensor of shape (num_triples, 3) containing the head, relation and tail of each triple
        :features: Tensor of shape (num_nodes, in_features) containing the features of each node
        :return: Tensor of shape (num_nodes, out_features) containing the new features of each node
        '''

        #assert (features is None) == (self.in_features is None), "in_features not provided!"
        triples = torch.tensor(triples, dtype=torch.long)
        in_dim = self.in_features if self.in_features is not None else self.num_nodes
        #triples = self.triples
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

        # Apply normalisation (vertical-stacking -> row-wise sum & horizontal-stacking -> column-wise sum)
        
        sums = sum_sparse(adj_indices, vals, adj_size, row_normalisation=vertical_stacking, device=device)
        if not vertical_stacking:
            # Rearrange column-wise normalised value to reflect original order (because of transpose-trick)
            n = general_edge_count 
            i = self_edge_count
            sums = torch.cat([sums[n:2 * n], sums[:n], sums[-i:]], dim=0)

        vals = vals / sums # Normalise values by row/column sum (depending on vertical_stacking) to obtain transition probabilities (i.e. row/column stochastic matrix) 
                           

        # Construct adjacency matrix
        if device == 'cuda':
            adj = torch.cuda.sparse.FloatTensor(indices=adj_indices.t(), values=vals, size=adj_size)
        else:
            adj = torch.sparse.FloatTensor(indices=adj_indices.t(), values=vals, size=adj_size) # normale Adjazenzmatrix, nicht transponiert

        if self.diag_weight_matrix:
            assert weights.size() == (num_relations, in_dim)
        else:
            assert weights.size() == (num_relations, in_dim, out_dim)

        if x is None:
            # Message passing if no features are given
            input = torch.mm(adj, weights.view(num_relations * in_dim, out_dim))
            fw = torch.einsum('ni, rio -> rno', input, weights).contiguous() # (num_relations, num_nodes, out_dim) 
            output = torch.mm(adj, fw.view(self.num_relations * self.num_nodes, out_dim))
        elif self.diag_weight_matrix: # rgc_no_hidden
            fw = torch.einsum('ij,kj->kij', x, weights)
            fw = torch.reshape(fw, (self.num_relations * self.num_nodes, in_dim))
            output = torch.mm(adj, fw)
        elif self.vertical_stacking:
            # Message passing if the adjacency matrix is vertically stacked
            af = torch.spmm(adj, x)
            af = af.view(self.num_relations, self.num_nodes, in_dim)
            output = torch.einsum('rio, rni -> no', weights, af)
        else: #rgc1
            # Message passing if the adjacency matrix is horizontally stacked
            # Note: this is the same as the original R-GCN paper
            # relation-wise matrix multiplication (i.e. matrix multiplication with a batch of matrices)
            fw = torch.einsum('ni, rio -> rno', x, weights).contiguous() # (num_relations, num_nodes, out_dim) 
            output = torch.mm(adj, fw.view(self.num_relations * self.num_nodes, out_dim)) # (num_nodes, out_dim) # fw.view(self.num_relations * self.num_nodes, out_dim) absichern

        assert output.size() == (self.num_nodes, out_dim)
        
        if self.bias is not None:
            output = torch.add(output, self.bias)
        
        if x is None:
            return output, adj, input
        return output, adj, None
    
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
        #self.feat = feat

       # assert (triples is not None or nnodes is not None or nrel is not None or nclass is not None), \
        "The following must be specified: triples, number of nodes, number of relations and number of classes!"
        assert 0 < nlayers < 3, "Only supports the following number of RGCN layers: 1 and 2."

        if nlayers == 1:
            nhid = 50

        if nlayers == 2:
            assert nhid is not None, "Number of hidden layers not specified!"

        #triples = torch.tensor(triples, dtype=torch.long)
        #with torch.no_grad():
            #self.register_buffer('triples', triples)
            # Add inverse relations and self-loops to triples
            #self.register_buffer('triples_plus', add_inverse_and_self(triples, nnodes, nrel))

        self.rgcn_no_hidden = RelationalGraphConvolutionNC(
            #triples=self.triples_plus,
            num_nodes=nnodes,
            num_relations=nrel * 2 + 1,
            in_features=nfeat,
            out_features=nhid,
            edge_dropout=edge_dropout,
            decomposition=decomposition,
            vertical_stacking=False,
        )
        if nlayers == 2:
            self.rgc1 = RelationalGraphConvolutionNC(
                #triples=self.triples_plus,
                num_nodes=nnodes,
                num_relations=nrel * 2 + 1,
                in_features=nhid,
                out_features=nhid,
                edge_dropout=edge_dropout,
                decomposition=decomposition,
                vertical_stacking=False
            )
        self.dense = nn.Linear(nhid, nclass, bias = True)

    def forward(self, x, triples):
        """ Embed relational graph and then compute class probabilities """
        activation = {}
        x, adj, input = self.rgcn_no_hidden(x, triples)
        if input is not None:
            activation['input'] = input.detach()
        if self.nlayers == 2:
            x = F.relu(x)
            activation['rgcn_no_hidden'] = x.detach()
            x, adj, = self.rgc1(x, triples)
        x = F.relu(x)
        activation['rgc1'] = x.detach()
        x = self.dense(x)
        x = F.softmax(x, dim=1)
        activation['dense'] = x.detach()
        return x, adj, activation

class EmbeddingNodeClassifier(NodeClassifier):
    """ Node classification model with node embeddings as the feature matrix """
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
                 #emb=None):
        self.activations = []
        assert nemb is not None, "Size of node embedding not specified!"
        nfeat = nemb  # Configure RGCN to accept node embeddings as feature matrix

        assert nlayers == 2, "For this model only 2 layers are normally configured (for now)"
        nhid = nemb

        super(EmbeddingNodeClassifier, self)\
            .__init__(nnodes, nrel, nfeat, nhid, 1, nclass, edge_dropout, decomposition)

        # This model has a custom first layer
        self.rgcn_no_hidden = RelationalGraphConvolutionNC(#triples=self.triples_plus,
                                                         num_nodes=nnodes,
                                                         num_relations=nrel * 2 + 1,
                                                         in_features=nfeat,
                                                         out_features=nhid,
                                                         edge_dropout=edge_dropout,
                                                         decomposition=decomposition,
                                                         vertical_stacking=False,
                                                         diag_weight_matrix=False)
        self.rgc1 = RelationalGraphConvolutionNC(
            #triples=self.triples_plus,
            num_nodes=nnodes,
            num_relations=nrel * 2 + 1,
            in_features=nhid,
            out_features=nhid,
            edge_dropout=edge_dropout,
            decomposition=decomposition,
            vertical_stacking=False
            )

        self.dense = nn.Linear(nhid, nclass, bias = True)

        # Node embeddings
        #self.node_embeddings = emb# nn.Parameter(torch.FloatTensor(nnodes, nemb)) #emb

        # Initialise Parameters
        #nn.init.kaiming_normal_(self.x, mode='fan_in')
        

    def forward(self, x, triples):
        """ Embed relational graph and then compute class probabilities """
        activation = {}
        out, adj = self.rgcn_no_hidden(x, triples)
        out = F.relu(out)
        activation['rgcn_no_hidden'] = out.detach()
        out, adj = self.rgc1(out, triples) # features=x
        out = F.relu(out)
        activation['rgc1'] = out.detach()
        out = self.dense(out)
        out = F.softmax(out, dim=1)
        activation['dense'] = out.detach()
        return out, adj, activation


def attach_dim(v, n_dim_to_prepend=0, n_dim_to_append=0):
    return v.reshape(torch.Size([1] * n_dim_to_prepend) + v.shape + torch.Size([1] * n_dim_to_append))

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

def stack_matrices(triples, num_nodes, num_rels, vertical_stacking=True, device='cpu'):
    """
    Computes a sparse adjacency matrix for the given graph (the adjacency matrices of all
    relations are stacked vertically).
    :param triples: A tensor of shape (num_triples, 3) containing the triples of the graph.
    :param num_nodes: The number of nodes in the graph.
    :param num_rels: The number of relations in the graph.
    :param fr: A tensor of shape (num_triples,) containing the indices of the source nodes.
    :param to: A tensor of shape (num_triples,) containing the indices of the target nodes.
    :param rel: A tensor of shape (num_triples,) containing the indices of the relations.
    :param offset: A tensor of shape (num_triples,) containing the offsets to add to the
        indices of the non-zero elements of the adjacency matrix.
    :param indices: A tensor of shape (num_triples, 2) containing the indices of the non-zero
        elements of the adjacency matrix.

    :param vertical_stacking: If True, the adjacency matrices of all relations are stacked
        vertically. Otherwise, they are stacked horizontally.
    :param device: The device on which the adjacency matrix should be stored.
    :offset: The offset to add to the indices of the non-zero elements of the adjacency
    :indices: A tensor of shape (num_triples, 2) containing the indices of the non-zero
    :return: A tuple (indices, size) where indices is a tensor of shape (num_triples, 2)
        containing the indices of the non-zero elements of the adjacency matrix, and size
        is a tuple (num_rows, num_cols) containing the size of the adjacency matrix.
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

def sum_sparse(indices, values, size, row_normalisation=True, device='cpu'):
    """
    Sum the rows or columns of a sparse matrix, and redistribute the
    results back to the non-sparse row/column entries
    Arguments are interpreted as defining sparse matrix.
    :param indices: A tensor of shape (num_non_zero, 2) containing the indices of the non-zero	
        elements of the sparse matrix.
    :param values: A tensor of shape (num_non_zero,) containing the values of the non-zero
        elements of the sparse matrix.
    :param size: A tuple (num_rows, num_cols) containing the size of the sparse matrix.
    :param row_normalisation: If True, the rows of the sparse matrix are normalised. Otherwise,
        the columns are normalised.
    :param device: The device on which the sparse matrix should be stored.
    :param k: The number of non-zero elements in the sparse matrix.
    :param r: The number of relations in the graph.
    :return: A tensor of shape (num_rows, num_cols) containing the normalised sparse matrix.    

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

def locate_file(filepath):

    directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    return directory + '/' + filepath

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