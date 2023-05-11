import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.module import Module
import numpy as np
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, add_self_loops
from torch.nn.parameter import Parameter
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score
from torch import from_numpy
import csv
import time
import numpy as np
import scipy.sparse as sp
from copy import deepcopy
import torch
import pickle
from utils import utils
from model import rgcn, gat 


def get_relevance_for_dense_layer(a, w, b, rel_in):
    '''
    :param a: activations of the layer
    :param w: weights of the layer
    :param b: bias of the layer
    :param rel_in: relevance of the input of the layer
    :return: relevance of the output of the layer
    '''
    z = a.detach().numpy().dot(w.t().detach().numpy()) #2835x50 ; 4x50; 2835x4
    s = np.divide(rel_in,z) # 2835x4; 2835x4; 2835x4
    pre_res = s.detach().numpy().dot(w.detach().numpy()) # 2835x4; 4x50; 2835x50
    out = np.multiply(a, pre_res) # 2835x50; 2835x50; 2835x50
    return out


def lrp(activation, weights, adjacency, relevance):
    #1.Lrp Schritt
    adj = adjacency.to_dense().view(2835,49,2835)
    Xp = (adj @ activation).transpose(0,1)
    sumzk = (Xp @ weights.mT).sum(dim=0)+1e-9
    s = torch.div(relevance,sumzk)
    zkl = s @ weights 
    out = Xp * zkl


    Xp = Xp+1e-9 
    z = out / Xp
    f_out = adj.T.transpose(0,1) @ z
    rel = (activation * f_out).sum(dim=0)

    return rel

def lrp2 (activation,weights, adjacency,relevance):
    adj = adjacency.to_dense().view(2835,2835,49)
    Yp = activation @ weights
    sumzk = (adj.T @ Yp).sum(dim=0)+1e-9
    s = relevance / sumzk

    zkl = adj.T @ s
    out = zkl * Yp

    Yp = (activation @ weights)+1e-9
    z = out/Yp
    f_out = Yp.detach().numpy() * z
    rel = torch.Tensor(f_out).sum(dim=0)
    return rel

def lrp_rgcn(activation_m, activation_f, weights_m, weights_f, adjacency, relevance_m, variant='A'):
    if variant == 'A':
        rel = lrp(activation_m, weights_m, adjacency, relevance_m)
        out = lrp(activation_f, weights_f, adjacency, rel)
    else:
        rel = lrp2(activation_m, weights_m, adjacency, relevance_m)
        out = lrp2(activation_f, weights_f, adjacency, rel)
    return out

def tensor_max_value_to_1_else_0(tensor, x):
    max_value = tensor.argmax()
    idx = test_idx[x]
    t = torch.zeros(2835,4)
    t[idx,max_value] = 1
    return t


if __name__ == '__main__':
    homedir="C:/Users/luisa/Projekte/Masterthesis/AIFB/"
    triples, (n2i, i2n), (r2i, i2r), train, test = utils.load_data(homedir = "C:/Users/luisa/Projekte/Masterthesis/AIFB")
    pyk_emb = utils.load_pickle(homedir + "data/AIFB/embeddings/pykeen_embedding_TransH")
    pyk_emb = torch.tensor(pyk_emb, dtype=torch.float)
    lemb = len(pyk_emb[1])
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

    model = rgcn.EmbeddingNodeClassifier
    model = model(
        triples=triples,
        nnodes=num_nodes,
        nrel=num_relations,
        nclass=num_classes,
        nhid=16,
        nlayers=2,
        decomposition=None,
        nemb=lemb,
        emb=pyk_emb)
    

    optimiser = torch.optim.Adam
    optimiser = optimiser(
        model.parameters(),
        lr=0.1,
        weight_decay=0.0)
    epochs = 20

    #Training
    for epoch in range(1, epochs+1):
        t1 = time.time()
        criterion = nn.CrossEntropyLoss()
        model.train()
        optimiser.zero_grad()
        classes, adj_m, val_norm, activation,fw = model()
        classes = classes[train_idx, :]
        loss = criterion(classes, train_lbl)
        layer1_l2_penalty = 0.0
        decomposition = None
        # Apply l2 penalty on first layer weights
        if layer1_l2_penalty > 0.0:
            if decomposition is not None and decomposition['type'] == 'basis':
                layer1_l2 = model.rgc1.bases.pow(2).sum() + model.rgc1.comps.pow(2).sum()
            elif decomposition is not None and decomposition['type'] == 'block':
                layer1_l2 = model.rgc1.blocks.pow(2).sum()
            else:
                layer1_l2 = model.rgc1.weights.pow(2).sum()
            loss = loss + layer1_l2_penalty * layer1_l2
    print(classes)
    t2 = time.time()
    loss.backward()
    optimiser.step()
    t3 = time.time()

    torch.save(model.state_dict(), homedir +'out/pykeen_model/test.pth')
    params = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            params[name] = param.data


    act_dense = activation['dense']
    act_rgc1 = activation['rgc1']
    act_rgc1_no_hidden = activation['rgcn_no_hidden']
    bias_dense = model.dense.bias
    bias_rgc1 = model.rgc1.bias
    bias_rgc1_no_hidden = model.rgcn_no_hidden.bias
    weight_dense = model.dense.weight
    weight_rgc1 = model.rgc1.weights
    weight_rgc1_no_hidden = model.rgcn_no_hidden.weights
    relevance, adj, val_norm, activation, fw = model()

    # LRP
    x=19
    selected_rel = relevance[test_idx, :][x]    
    rel1 = tensor_max_value_to_1_else_0(selected_rel,x)
    rel2 = get_relevance_for_dense_layer(act_rgc1, weight_dense, bias_dense, rel1)
    rel = lrp_rgcn(act_rgc1_no_hidden, pyk_emb, weight_rgc1, weight_rgc1_no_hidden, adj_m, rel2, variant='A')

    

    with torch.no_grad(): #no backpropagation
        model.eval()
        #classes, adj, val_norm, activation = model()
        classes = classes[train_idx, :].argmax(dim=-1)
        train_accuracy = accuracy_score(classes.cpu(), train_lbl.cpu()) * 100  # Note: Accuracy is always computed on CPU
        classes, adj, val_norm, activation = model()
        classes = classes[test_idx, :].argmax(dim=-1)
        test_accuracy = accuracy_score(classes.cpu(), test_lbl.cpu()) * 100  # Note: Accuracy is always computed on CPU

        print(f'[Epoch {epoch}] Loss: {loss.item():.5f} Forward: {(t2 - t1):.3f}s Backward: {(t3 - t2):.3f}s '
              f'Train Accuracy: {train_accuracy:.2f} Test Accuracy: {test_accuracy:.2f}')

    print('Training is complete!')

    print("Starting evaluation...")
    model.eval()
    test_accuracy = accuracy_score(classes.cpu(), test_lbl.cpu()) * 100  # Note: Accuracy is always computed on CPU
    print(classes)
    print(test_lbl)
    print(f'[Evaluation] Test Accuracy: {test_accuracy:.2f}')