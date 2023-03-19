import torch
import torch.nn as nn
from torch_geometric.data import Data
import pandas as pd
from torch.nn import functional as F
from collections import Counter
import numpy as np
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, add_self_loops
from torch_geometric.nn import GCNConv
import tqdm
import rdflib as rdf
import numpy as np
import numpy as np
import torch
from rdflib import URIRef
import torch.nn as nn
import torch.nn.functional as F


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

def st(node):
    """
    Maps an rdflib node to a unique string. We use str(node) for URIs (so they can be matched to the classes) and
    we use .n3() for everything else, so that different nodes don't become unified.

    Source: https://github.com/pbloem/gated-rgcn/blob/1bde7f28af8028f468349b2d760c17d5c908b58b/kgmodels/data.py#L16
    """
    if type(node) == URIRef:
        return str(node)
    else:
        return node.n3()

def load_data(homedir):
    
    labels_train = pd.read_csv(homedir + "/data/trainingSet.tsv", sep="\t")
    labels_test = pd.read_csv(homedir + "/data/testSet.tsv", sep="\t")
    labels = labels_train['label_affiliation'].astype('category').cat.codes
    train = {}
    for nod, lab in zip(labels_train['person'].values, labels):
        train[nod] = lab
    labels = labels_test['label_affiliation'].astype('category').cat.codes
    test = {}
    for nod, lab in zip(labels_test['person'].values, labels):
        test[nod] = lab
    print('Labels loaded.')
    graph = rdf.Graph()
    file = homedir + "/data/aifb_witho_complete.nt"
    graph.parse(file, format=rdf.util.guess_format(file))

    print('RDF loaded.')


    triples = graph

    nodes = set()
    relations = Counter()

    for s, p, o in triples:
        nodes.add(st(s))
        nodes.add(st(o))

        relations[st(p)] += 1

    i2n = list(nodes) # maps indices to labels
    n2i = {n:i for i, n in enumerate(i2n)} # maps labels to indices

    # the 'limit' most frequent labels are maintained, the rest are combined into label REST to save memory
   
    i2r =list(relations.keys())

    r2i = {r: i for i, r in enumerate(i2r)}

    # Collect all edges into a list: [from, relation, to] (only storing integer indices)
    edges = list()
    REST = '.rest'
    for s, p, o in tqdm.tqdm(triples):
        s, p, o = n2i[st(s)], st(p), n2i[st(o)]
        pf = r2i[p] if (p in r2i) else r2i[REST]
        edges.append([s, pf, o])

    print('Graph loaded.')
    # limit = None
    # enable_cache = True
    # cachefile = os.path.join(path, dataset + '.pkl.gz')
    # # Cache the results for fast loading next time
    # if limit is None and enable_cache:
    #     with open(cachefile, 'wb') as file:
    #         pickle.dump([edges, (n2i, i2n), (r2i, i2r), train, test], file)

    return edges, (n2i, i2n), (r2i, i2r), train, test

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

class GCN(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_dim, output_dim, num_layers):
        super(GCN, self).__init__()

        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(nn.Linear(input_dim, hidden_dim))
        for i in range(num_layers - 1):
            self.conv_layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.fc_layer = nn.Linear(hidden_dim*num_nodes, output_dim)

    def forward(self, adj, features):
        x = features

        for i in range(self.num_layers):
            x = self.conv_layers[i](x)

            # aggregate messages
            adj_t = adj.transpose(0,1).contiguous()
            x = torch.matmul(adj_t, x)

            # normalize node embeddings
            deg = adj_t.sum(dim=1, keepdim=True)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            x = x * deg_inv_sqrt

            # apply activation function
            x = F.relu(x)

        x = x.view(-1)
        x = self.fc_layer(x)

        return x



    

    
if __name__ == '__main__':
    homedir = "C:/Users/luisa/Projekte/Masterthesis/AIFB"
    df = pd.read_csv(homedir + "/data/aifb_without_literals.tsv", sep="\t")

    

# read graph from TSV file
    triples, (n2i, i2n), (r2i, i2r), train, test = load_data(homedir = "C:/Users/luisa/Projekte/Masterthesis/AIFB")

    # Check for available GPUs
    use_cuda =  torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Convert train and test datasets to torch tensors
    train_idx = [n2i[name] for name, _ in train.items()]
    train_lbl = [cls for _, cls in train.items()]
    train_idx = torch.tensor(train_idx, dtype=torch.long, device=device)
    train_lbl = torch.tensor(train_lbl, dtype=torch.long, device=device)

    test_idx = [n2i[name] for name, _ in test.items()]
    test_lbl = [cls for _, cls in test.items()]
    test_idx = torch.tensor(test_idx, dtype=torch.long, device=device)
    test_lbl = torch.tensor(test_lbl, dtype=torch.long, device=device)

    classes = set([int(l) for l in test_lbl] + [int(l) for l in train_lbl])
    num_classes = len(classes)
    num_nodes = len(n2i)
    num_relations = len(r2i)

    # create adjacency matrix

    # Stack adjacency matrices either vertically or horizontally
    adj_indices, adj_size = stack_matrices(
        triples,
        num_nodes,
        num_relations,
        vertical_stacking=False,
        device=device
        )
    vertical_stacking = False
    self_edge_count = num_nodes
    num_triples = adj_indices.size(0)
    general_edge_count = int((triples.size(0) - num_nodes)/2)
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

    # num_nodes = len(graph)
    # adj = np.zeros((num_nodes, num_nodes))
    # for head, tails in graph.items():
    #     for tail in tails:
    #         adj[head][tail] = 1

    # create feature matrix
    input_dim = 10
    features = np.random.rand(num_nodes, input_dim)

    # instantiate GCN model
    hidden_dim = 16
    output_dim = 8
    num_layers = 2

    model = GCN(num_nodes, input_dim, hidden_dim, output_dim, num_layers)
