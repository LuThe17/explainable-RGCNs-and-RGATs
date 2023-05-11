import torch
from torch.nn import functional as F
import numpy as np
import torch
import pickle
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


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
    
    Xp = (adjacency.t() @ activation).reshape(49, 2835, 50)
    sumzk = (Xp @ weights).sum(dim=0)+1e-9
    s = torch.div(relevance,sumzk)
    zkl = s @ weights 
    out = Xp.detach().numpy() * zkl.detach().numpy()

    #2.Lrp Schritt
    Xp = Xp+1e-9 
    z = out / Xp
    adj = adjacency.to_dense().view(2835,2835,49)
    f_out = adj.T @ z
    rel = (activation * f_out).sum(dim=0) #3. LRP Schritt (sum)
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


if __name__ == '__main__':
    with open('variables.pkl', 'rb') as f:
        act_rgc1, weight_dense, bias_dense, rel1, act_rgc1_no_hidden, pyk_emb, weight_rgc1, weight_rgc1_no_hidden, adj_m = pickle.load(f)
    rel2 = get_relevance_for_dense_layer(act_rgc1, weight_dense, bias_dense, rel1)
    rel = lrp_rgcn(act_rgc1_no_hidden, pyk_emb, weight_rgc1, weight_rgc1_no_hidden, adj_m, rel2, variant='A')