import torch
import numpy as np
import scipy.sparse as sp
import tqdm
import pandas as pd
import rdflib as rdf
from collections import Counter
from rdflib.term import URIRef
from random import choice
import networkx as nx

def get_relevance_for_dense_layer(a, w, b, rel_in):
    '''
    :param a: activations of the layer
    :param w: weights of the layer
    :param b: bias of the layer
    :param rel_in: relevance of the input of the layer
    :return: relevance of the output of the layer
    '''
    z = a.detach().numpy().dot(w.t().detach().numpy()) + 1e-10#2835x50 ; 4x50; 2835x4
    s = np.divide(rel_in,z) # 2835x4; 2835x4; 2835x4
    pre_res = s.detach().numpy().dot(w.detach().numpy()) # 2835x4; 4x50; 2835x50
    out = np.multiply(a, pre_res) # 2835x50; 2835x50; 2835x50
    return out

def lrp_rgat_layer(x, w, alpha, rel, s1, s2, num_neighbors, lrp_step):
    if lrp_step == 'relevance_alphaandg':
        g = x @ w
        z = (((alpha @ g) * s1) + ((alpha @ g) * (1-s1)) + 1e-10 ).sum(dim=0)
        s = torch.div(rel, z)
        zkl = s @ g.mT
        out_alpha =alpha * zkl * s1

        zkl2 = alpha.mT @ s  
        out_g = (zkl2 * g) * (1-s1)
        out_g = out_g.sum(dim=0)
        print(out_g.to_dense().sum())
        return out_alpha, out_g
    
    elif lrp_step == 'relevance_h':
        rel_g = (alpha.sum(dim=0) + rel + s1.sum(dim=0))
        z = (x @ w.mT + 1e-10).sum(dim=0)
        s = torch.div(rel_g, z) 
        pre_res = s @ w 
        out = x * pre_res 
        return out.sum(dim=0)
    elif lrp_step == 'relevance_softmax':
        exponential = x
        sum_exp = w
        y = torch.where(sum_exp == 0, torch.zeros_like(sum_exp), torch.div(1,sum_exp))
        out_exp = rel * s2

        torch.count_nonzero(exponential, dim=0)
        softmax_output = torch.where(exponential == 0, torch.zeros_like(exponential), exponential / sum_exp)
        exp = torch.where(softmax_output == 0, torch.zeros_like(softmax_output), (1-softmax_output))
        exp = torch.where(softmax_output==1, exp+1, exp)
        num_neighbors = torch.zeros(24, 2835)
        # Iterate over the relations
        for relation in range(exp.size()[0]):
            sum_neighbor = torch.count_nonzero(exp[relation], dim = 1)
            sum_neighbor2 = torch.where(sum_neighbor==1, sum_neighbor, sum_neighbor - 1)
            sum_neighbor3 = torch.where(sum_neighbor==0, torch.zeros_like(sum_neighbor), sum_neighbor2)
            num_neighbors[relation] = sum_neighbor3
        num_neighbors2= num_neighbors.unsqueeze(2).expand(-1,-1,2835)
        ant = torch.where(num_neighbors2 == 0 , torch.zeros_like(num_neighbors2), exp/num_neighbors2)
    
        sum_rel = torch.zeros(24,2835)
        for relation in range(rel.size()[0]):
            sum_rel[relation]  = rel[relation].sum(dim=1)
        zkl = rel * ant
        ant_zkl = torch.zeros(24,2835)
        for relation in range(rel.size()[0]):
            ant_zkl[relation]  = zkl[relation].sum(dim=1)
        ant_zkl = torch.where(ant_zkl==0, torch.zeros_like(ant_zkl),sum_rel/ant_zkl).unsqueeze(2).expand(-1,-1,2835)

        out_y = zkl * (1-s2) * ant_zkl
        output = out_exp+ out_y
        return output
    elif lrp_step == 'relevance_q_k':
        G = out_l1 @ w_l2
        Gmi = M_l2.to_dense() @ G
        Gmj = M_l2.to_dense().transpose(1,2) @ G

        Q = torch.matmul(Gmi, q_l2).squeeze() #(3)
        K = torch.matmul(Gmj, k_l2).squeeze() #(3)
        E = torch.zeros(24,2835,2835)
        E[edge_type, edge_index[0], edge_index[1]] = Q[edge_type,edge_index[0]] + K[edge_type,edge_index[1]]
        rel_q = torch.where(E ==0, torch.zeros_like(E),(Q.unsqueeze(2).expand(-1,-1,2835)/E)*rel)
        rel_k = torch.where (E ==0 , torch.zeros_like(E), (K.unsqueeze(1).expand(-1,2835,-1)/E)*rel)
  
        Q = torch.matmul(Gmi, q_l2).squeeze() +1e-19
        s_q = (rel_q/Q.unsqueeze(2).expand(-1,-1,2835))
        zkl_q = s_q.mT @ Gmi
        rel_Q = zkl_q * q_l2.mT

        K = torch.matmul(Gmj, k_l2).squeeze() +1e-19
        s_k = (rel_k/(K.unsqueeze(1).expand(-1,2835,-1)))
        zkl_k = s_k @ Gmj
        rel_K = zkl_k * k_l2.mT

                                         
        return rel_Q, rel_K

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
    rel_edge = (activation * f_out)
    rel_node = rel_edge.sum(dim=0)
    return rel_node, rel_edge

def lrp_first_noemb_layer(adjacency,weights, relevance):
    adj = adjacency.to_dense().view(49,2835,2835)
    sumzk = (adj.mT @ weights).sum(dim=0)+1e-9
    s = torch.div(relevance,sumzk)
    zkl = adj @ s
    out = zkl * weights


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

def lrp_rgcn(act_rgc1, weight_dense, bias_dense, relevance, 
             act_rgc1_no_hidden, weight_rgc1, weight_rgc1_no_hidden, 
             adjacency, pyk_emb, test_idx, model_name, i, variant='A', ):
    x=i
    selected_rel = relevance[x[1],:]
    rel1 = tensor_max_value_to_1_else_0(selected_rel, x, test_idx)
    print(rel1.to_sparse_coo())
    rel2 = get_relevance_for_dense_layer(act_rgc1, weight_dense, bias_dense, rel1)
    print(rel2.to_sparse_coo())
    if variant == 'A':
        rel_node, rel_edge = lrp(act_rgc1_no_hidden, weight_rgc1, adjacency, rel2)
        print(rel_node.to_sparse_coo())
        if model_name == 'RGCN_emb':
            rel_node, rel_edge = lrp(pyk_emb, weight_rgc1_no_hidden, adjacency, rel_node)
            print(rel_node.to_sparse_coo())
        elif model_name == 'RGCN_no_emb':
            rel_node, rel_edge = lrp_first_noemb_layer(adjacency, weight_rgc1_no_hidden, rel_node)
    else:
        rel = lrp2(act_rgc1_no_hidden, weight_rgc1, adjacency, rel2)
        out = lrp2(pyk_emb, weight_rgc1_no_hidden, adjacency, rel)
    return rel_node, rel_edge

def lrp_rgat(parameter_list, input, weight_dense, relevance, s1, s2,x):
    #x=19
    selected_rel = relevance[test_idx,:][x[0]]
    rel1 = tensor_max_value_to_1_else_0(selected_rel,x[0])
    print(rel1.to_sparse_coo())
    for i in parameter_list:
        globals()[i[0]] = i[1]
    rel2 = get_relevance_for_dense_layer(out_l2, weight_dense, None, rel1)
    r_alpha, rg =  lrp_rgat_layer(out_l1, w_l2, alpha3_l2, rel2, s1, None, None, lrp_step='relevance_alphaandg')
    rel_softmax = lrp_rgat_layer(exponential_l2, resmat_l2, None, r_alpha, None, s2, num_neighbors_l2, lrp_step='relevance_softmax')
    rel_q, rel_k = lrp_rgat_layer(out_l1, w_l2, None, rel_softmax, None, None, None, 'relevance_q_k')
    rel_h = lrp_rgat_layer(input, w_l2, rel_q, rg, rel_k, None,None,lrp_step='relevance_h')
    return rel_h

def tensor_max_value_to_1_else_0(tensor, x, test_idx):
    max_value = tensor.argmax()
    idx = test_idx[x[0]]
    t = torch.zeros(2835,4)
    t[idx,max_value] = 1
    return t
def get_highest_relevance(rel):
    global high
    mask = rel == rel.max()
    indices = torch.nonzero(mask)
    high = rel.max()
    return high, indices

def analyse_lrp(emb, edge_index, edge_type, model, parameter_list, input, weight_dense, relevance, test_idx, model_name, s1, s2):
    homedir = "C:/Users/luisa/Projekte/Masterthesis/AIFB/"
    dataset_name = 'AIFB'
    pred, adj,act = model(emb,input)
    torch.save(pred, homedir + 'out/'+dataset_name+'/' 'pred_before.pt')
    if model_name == 'RGCN_emb':
        rel_nodes_list, min_nodes_list = {}, {}
        rel_edges_list, pos_min_nodes = {}, {}
        max_nodes_list, max_edges_list = {}, {}
        pos_max_nodes, pos_max_edges, min_edges_list, pos_min_edges = {}, {}, {}, {}
        rel_nodes_list_new, rel_edges_list_new = {}, {}
        pos_min_nodes_new, pos_min_edges_new = {}, {}
        max_nodes_list_new, max_edges_list_new = {}, {}
        pos_max_nodes_new, pos_max_edges_new = {}, {}
        min_nodes_list_new, min_edges_list_new = {}, {}
        for i in enumerate(test_idx):
            relevance, adj,activation = model(emb, input)
            rel_nodes, rel_edges = lrp_rgcn(activation['rgc1'], model.dense.weight, None, relevance, activation['rgcn_no_hidden'], model.rgc1.weights, 
                           model.rgcn_no_hidden.weights, adj,  emb, test_idx, model_name, i, 'A')
            rel_nodes = rel_nodes.sum(dim=1)
            rel_edges = rel_edges.sum(dim=2)
            rel_nodes_list[i] = rel_nodes#rel_nodes.argmax(), rel_nodes.max(), rel_nodes.argmin(), rel_nodes.min()
            max_nodes_list[i] = rel_nodes.max().item()
            pos_max_nodes[i] = rel_nodes.argmax().item()
            min_nodes_list[i] = rel_nodes.min().item()
            pos_min_nodes[i] = rel_nodes.argmin().item()
            rel_edges_list[i] = rel_edges#(rel_edges==torch.max(rel_edges)).nonzero(), rel_edges.max(), (rel_edges==torch.min(rel_edges)).nonzero(), rel_edges.min()
            max_edges_list[i] = rel_edges.max().item()
            pos_max_edges[i] = (rel_edges==torch.max(rel_edges)).nonzero()
            min_edges_list[i] = rel_edges.min().item()
            pos_min_edges[i] = (rel_edges==torch.min(rel_edges)).nonzero()
        nodes = {'tensor_nodes':rel_nodes_list, 'max_nodes':max_nodes_list, 'pos_max_nodes':pos_max_nodes, 'min_nodes':min_nodes_list, 'pos_min_nodes':pos_min_nodes}    
        nodes_table = pd.DataFrame(nodes)
        nodes_table.to_csv(homedir + 'out/'+dataset_name+'/' + model_name + '/LRP_nodes_table.csv')
        edges = {'tensor_edges':rel_edges_list, 'max_edges':max_edges_list, 'pos_max_edges':pos_max_edges, 'min_edges':min_edges_list, 'pos_min_edges':pos_min_edges}
        edges_table = pd.DataFrame(edges)
        edges_table.to_csv(homedir + 'out/'+dataset_name+'/' + model_name + '/LRP_edges_table.csv')
        n_largest_nodes = nodes_table.nlargest(3, 'max_nodes')
        for i in range(len(n_largest_nodes)):
            name_node = n_largest_nodes.index[0]
            node_indice = n_largest_nodes['pos_max_nodes'][i]
            lrp_node = node_indice.index.item().item()
            res_ind = torch.where((edge_index[0] == lrp_node) & (edge_index[1]== node_indice[0]))[0]
            exclude = [node_indice[0]]
            new_index = choice(list(set(range(0,edge_index.max().item())) - set(exclude)))
            edge_index_new = edge_index.clone()
            edge_index_new[1][res_ind] = new_index
            triples_new = torch.stack((edge_index_new[0], edge_type, edge_index_new[1]), 0).type(torch.long)
            
            classes, adj_m,activation = model(emb, triples_new.T)
            torch.save(classes, homedir + 'out/'+dataset_name+'/' + model_name + '/pred_after'+str(name_node)+'adapt_'+str(node_indice[0])+'_'+str(new_index)+'.pt')
            rel_nodes_new, rel_edges_new = lrp_rgcn(activation['rgc1'], model.dense.weight, None, classes, activation['rgcn_no_hidden'], model.rgc1.weights, 
                    model.rgcn_no_hidden.weights, adj,  emb, test_idx, model_name, i, 'A')
            rel_nodes_new = rel_nodes_new.sum(dim=1)
            rel_nodes_list_new[i] = rel_nodes_new#rel_nodes.argmax(), rel_nodes.max(), rel_nodes.argmin(), rel_nodes.min()
            max_nodes_list_new[i] = rel_nodes_new.max().item()
            pos_max_nodes_new[i] = rel_nodes_new.argmax().item()
            min_nodes_list_new[i] = rel_nodes_new.min().item()
            pos_min_nodes_new[i] = rel_nodes_new.argmin().item()
            
            rel_edges_new = rel_edges_new.sum(dim=2)
            rel_edges_list_new[i] = rel_edges_new#(rel_edges==torch.max(rel_edges)).nonzero(), rel_edges.max(), (rel_edges==torch.min(rel_edges)).nonzero(), rel_edges.min()
            max_edges_list_new[i] = rel_edges_new.max().item()
            pos_max_edges_new[i] = (rel_edges_new==torch.max(rel_edges_new)).nonzero()
            min_edges_list_new[i] = rel_edges_new.min().item()
            pos_min_edges_new[i] = (rel_edges_new==torch.min(rel_edges_new)).nonzero()
        rel_new = {'rel_nodes_new':rel_nodes_list_new, 'rel_edges_new':rel_edges_list_new, 'max_nodes_new':max_nodes_list_new, 
                   'pos_max_nodes_new':pos_max_nodes_new, 'min_nodes_new':min_nodes_list_new, 'pos_min_nodes_new':pos_min_nodes_new, 'max_edges_new':max_edges_list_new, 'pos_max_edges_new':pos_max_edges_new, 'min_edges_new':min_edges_list_new, 'pos_min_edges_new':pos_min_edges_new}
        rel_new_table = pd.DataFrame(rel_new)
        rel_new_table.to_csv(homedir + 'out/'+dataset_name+'/' + model_name + '/LRP_nodes_edges_after_adaptation.csv')


    elif model_name == 'RGCN_no_emb':
        for i in enumerate(test_idx):
            rel = lrp_rgcn(parameter_list, input, weight_dense, relevance, s1, s2,i)
            high, indices = get_highest_relevance(rel)
            analyse_highest_relevance(rel)
            change_highest_relevance(rel, high)
            predict_new_result(rel,high,change)
    elif model_name == 'RGAT_no_emb':
        for i in enumerate(test_idx):
            rel = lrp_rgat(parameter_list, input, weight_dense, relevance, s1, s2,i)
            high, indices = get_highest_relevance(rel)
            analyse_highest_relevance(rel)
            change_highest_relevance(rel, high)
            predict_new_result(rel,high,change)
    elif model_name == 'RGAT_emb':
        for i in enumerate(test_idx):
            rel = lrp_rgat(parameter_list, input, weight_dense, relevance, s1, s2,i)
            high, indices = get_highest_relevance(rel)
            analyse_highest_relevance(rel)
            change_highest_relevance(rel, high)
            predict_new_result(rel,high,change)