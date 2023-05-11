import pickle
import torch
import numpy as np
import scipy.sparse as sp
import tqdm
import pandas as pd
import rdflib as rdf
from collections import Counter
from rdflib.term import URIRef



def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)



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
    '''
    Input: homedir: path to the directory where the data is stored
    Output: edges: list of edges
            n2i: dictionary mapping node labels to indices
            i2n: list of node labels
            r2i: dictionary mapping relation labels to indices
            i2r: list of relation labels
            train: dictionary mapping training node labels to indices
            test: dictionary mapping test node labels to indices
    '''
    labels_train = pd.read_csv(homedir + "/data/AIFB/trainingSet.tsv", sep="\t")
    labels_test = pd.read_csv(homedir + "/data/AIFB/testSet.tsv", sep="\t")
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
    file = homedir + "/data/AIFB/aifb_renamed_bn.nt"
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
   
    i2r =list(relations.keys()) # maps indices to labels

    r2i = {r: i for i, r in enumerate(i2r)} # maps labels to indices

    # Collect all edges into a list: [from, relation, to] (only storing integer indices)
    edges = list()
    REST = '.rest'
    for s, p, o in tqdm.tqdm(triples):
        s, p, o = n2i[st(s)], st(p), n2i[st(o)]
        pf = r2i[p] if (p in r2i) else r2i[REST]
        edges.append([s, pf, o])

    print('Graph loaded.')
    edges = torch.Tensor(edges)
    #create adjacency matrix
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(len(i2n), len(i2n)), dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = torch.FloatTensor(np.array(adj.todense()))
    return edges, (n2i, i2n), (r2i, i2r), train, test

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)