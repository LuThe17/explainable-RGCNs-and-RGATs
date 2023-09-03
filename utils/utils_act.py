import pickle
import torch
import numpy as np
import scipy.sparse as sp
import tqdm
import pandas as pd
from rdflib import Graph, Literal, URIRef
import rdflib as rdf
from collections import Counter
from rdflib.term import URIRef
import networkx as nx
import rdflib
import scipy
import gzip

def remove_literal_in_graph(g):
    l = []
    for s, p, o in g:
        if type(o) == rdflib.term.Literal:
            l.append(s)
            l.append(p)
            l.append(o)
            g.remove((s, p, o))
    return g

def rename_bnode_in_graph(g):
    new_iri = URIRef("http://bnode.org/")
    for s, p, o in g:
        if isinstance(o, rdflib.BNode):
            o_iri = URIRef(f"{new_iri}{o}")
            g.remove((s, p, o))#
            g.add((s, p, o_iri))
            #g.add((o_iri, RDF.subject, o))
            o = o_iri

        if isinstance(s, rdflib.BNode):
            s_iri = URIRef(f"{new_iri}{s}")
            g.remove((s, p, o))
            g.add((s_iri, p, o))
            #g.add((s_iri, RDF.subject, s))
            s = s_iri
    return g


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

def load_data(homedir,filename, emb_type, model_name):
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
    if filename == 'AIFB':
        kg_dir = 'data/AIFB/aifb_final.nt'
        train_dir = "data/AIFB/trainingSet.tsv"
        test_dir = "data/AIFB/testSet.tsv"
        pytest_dir = "data/AIFB/testSetpy.tsv"
        label_header = 'label_affiliation'
        nodes_header = 'person'
        sep="\t"
    elif filename == 'MUTAG':
        kg_dir = 'data/MUTAG/mutag_renamed_bn.nt'
        train_dir = "data/MUTAG/trainingSet.tsv"
        test_dir = "data/MUTAG/testSet.tsv"
        pytest_dir = "/data/MUTAG/testSetpy.tsv"
        label_header = 'label_mutagenic'
        nodes_header = 'bond'
        sep="\t"
    # elif filename == 'BGS':
    #     #homedir = '/pfs/work7/workspace/scratch/ma_luitheob-master/AIFB'
    #     kg_dir = 'data/BGS/bgs_renamed_bn_new.nt.gz'
    #     #kg_dir2= '/data/BGS/bgs_renamed_bn.nt.gz'
    #     train_dir = "data/BGS/trainingSet(lith).tsv"
    #     test_dir = "data/BGS/testSet(lith).tsv"
    #     pytest_dir = "data/BGS/testSetpy.tsv"
    #     label_header = 'label_lithogenesis'
    #     nodes_header = 'rock'
    #     sep="\t"
    elif filename == 'IMDB':
        kg_dir = 'data/IMDB/imdb_graph.nt'
        label_header = 'genre'
        nodes_header = 'movie'
        train_dir = 'data/IMDB/training_Set_new.csv'
        test_dir = 'data/IMDB/test_Set_new.csv'
        sep=','

    
    labels_train = pd.read_csv(homedir + train_dir, sep=sep)
    labels_test = pd.read_csv(homedir + test_dir, sep=sep)
    labels = labels_train[label_header].astype('category').cat.codes
    train = {}
    for nod, lab in zip(labels_train[nodes_header].values, labels):
        train[nod] = lab
    labels = labels_test[label_header].astype('category').cat.codes
    test = {}
    for nod, lab in zip(labels_test[nodes_header].values, labels):
        test[nod] = lab
    print('Labels loaded.')
    graph = rdf.Graph()
    file = homedir + kg_dir
    # if file.endswith('nt.gz'):
    #     with gzip.open(homedir + '/data/BGS/bgs_stripped.nt.gz', "rb") as out:
    #         graph.parse(file=out, format='nt')
    #     print('FINISH PARSING BGS STRIPPED.nt.gz')
    #     lith = rdflib.term.URIRef("http://data.bgs.ac.uk/ref/Lexicon/hasLithogenesis")
    #     graph.remove((None, lith, None))
    #     kg = remove_literal_in_graph(graph)
    #     print('################  RENAME BNODE IN GRAPH #############')
    #     kg = rename_bnode_in_graph(kg)
    #     graph = kg
    # else:
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
    edge_index = edges[:,[0,2]]
    #create adjacency matrix
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(len(i2n), len(i2n)), dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = torch.FloatTensor(np.array(adj.todense()))

    triples_plus = add_inverse_and_self(edges, len(i2n), len(i2r))
    if emb_type == None:
        with open (homedir + '/out/'+ filename+'/'+ model_name+'/triples_plus.pkl', 'wb') as fp:
            pickle.dump(triples_plus, fp)
        with open(homedir + '/out/'+ filename+'/'+ model_name+'/edges_list.pkl', 'wb') as fp:
            pickle.dump(edges, fp)
        with open (homedir + '/out/'+filename + '/'+ model_name + '/test_list.pkl','wb') as fp:
            pickle.dump(test, fp)
        with open (homedir + '/out/'+filename + '/'+ model_name + '/train_list.pkl','wb') as fp:
            pickle.dump(train,fp)
        with open (homedir + '/out/'+ filename+'/'+ model_name+'/i2r.pkl', 'wb') as fp:
            pickle.dump(i2r, fp)
        with open(homedir + '/out/'+ filename+'/'+ model_name+'/i2n.pkl', 'wb') as fp:
            pickle.dump(i2n, fp)
    else:
        with open (homedir + '/out/'+ filename+'/'+ model_name+ '/'+ emb_type+'/triples_plus.pkl', 'wb') as fp:
            pickle.dump(triples_plus, fp)
        with open(homedir + '/out/'+ filename+'/'+ model_name+'/'+ emb_type+'/edges_list.pkl', 'wb') as fp:
            pickle.dump(edges, fp)
        with open (homedir + '/out/'+filename + '/'+ model_name +'/'+ emb_type+ '/test_list.pkl','wb') as fp:
            pickle.dump(test,fp)
        with open (homedir + '/out/'+filename + '/'+ model_name + '/'+ emb_type+'/train_list.pkl','wb') as fp:
            pickle.dump(train,fp)
        with open (homedir + '/out/'+ filename+'/'+ model_name+'/'+ emb_type+'/i2r.pkl', 'wb') as fp:
            pickle.dump(i2r, fp)
        with open(homedir + '/out/'+ filename+'/'+ model_name+'/'+ emb_type+'/i2n.pkl', 'wb') as fp:  
            pickle.dump(i2n,fp)
    return adj, edges, (n2i, i2n), (r2i, i2r), train, test, triples, triples_plus

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


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
