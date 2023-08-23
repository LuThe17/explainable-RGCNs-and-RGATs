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

def lrp_rgat_layer(x, w, alpha, rel, s1, s2, num_neighbors,num_relations, num_nodes, edge_type, edge_index,M_l2, lrp_step):
    if lrp_step == 'relevance_alphaandg':
        g = x @ w
        z = (((alpha @ g) * s1) + ((alpha @ g) * (1-s1)) + 1e-10 ).sum(dim=0)
        s = torch.div(rel, z)
        zkl = s @ g.mT
        out_alpha =alpha * zkl * s1

        zkl2 = alpha.mT @ s  
        out_g = (zkl2 * g) * (1-s1)
        out_g = out_g.sum(dim=0)
        #print(out_g.to_dense().sum())
        return out_alpha, out_g
    
    elif lrp_step == 'relevance_h':
        rel_g = (alpha.sum(dim=0) + rel + s1.sum(dim=0))
        z = (x @ w.mT + 1e-10).sum(dim=0)
        s = torch.div(rel_g, z) 
        pre_res = s @ w 
        rel_edges = x * pre_res 
        rel_nodes = rel_edges.sum(dim=0)
        return rel_edges, rel_nodes
    elif lrp_step == 'relevance_softmax':
        exponential = x
        sum_exp = w
        y = torch.where(sum_exp == 0, torch.zeros_like(sum_exp), torch.div(1,sum_exp))
        out_exp = rel * s2

        torch.count_nonzero(exponential, dim=0)
        softmax_output = torch.where(exponential == 0, torch.zeros_like(exponential), exponential / sum_exp)
        exp = torch.where(softmax_output == 0, torch.zeros_like(softmax_output), (1-softmax_output))
        exp = torch.where(softmax_output==1, exp+1, exp)
        num_neighbors = torch.zeros(num_relations, num_nodes)
        # Iterate over the relations
        for relation in range(exp.size()[0]):
            sum_neighbor = torch.count_nonzero(exp[relation], dim = 1)
            sum_neighbor2 = torch.where(sum_neighbor==1, sum_neighbor, sum_neighbor - 1)
            sum_neighbor3 = torch.where(sum_neighbor==0, torch.zeros_like(sum_neighbor), sum_neighbor2)
            num_neighbors[relation] = sum_neighbor3
        num_neighbors2= num_neighbors.unsqueeze(2).expand(-1,-1,num_nodes)
        ant = torch.where(num_neighbors2 == 0 , torch.zeros_like(num_neighbors2), exp/num_neighbors2)
        sum_rel = torch.zeros(num_relations,num_nodes)
        for relation in range(rel.size()[0]):
            sum_rel[relation]  = rel[relation].sum(dim=1)
        zkl = rel * ant
        ant_zkl = torch.zeros(num_relations,num_nodes)
        for relation in range(rel.size()[0]):
            ant_zkl[relation]  = zkl[relation].sum(dim=1)
        ant_zkl = torch.where(ant_zkl==0, torch.zeros_like(ant_zkl),sum_rel/ant_zkl).unsqueeze(2).expand(-1,-1,num_nodes)

        out_y = zkl * (1-s2) * ant_zkl
        output = out_exp+ out_y
        return output
    elif lrp_step == 'relevance_q_k':
        G = out_l1 @ w_l2
        print(M_l2.shape, G.shape)
        Gmi = M_l2 @ G
        Gmj = M_l2.transpose(1,2) @ G

        Q = torch.matmul(Gmi, q_l2).squeeze() #(3)
        K = torch.matmul(Gmj, k_l2).squeeze() #(3)
        E = torch.zeros(num_relations,num_nodes,num_nodes)
        E[edge_type, edge_index[0], edge_index[1]] = Q[edge_type,edge_index[0]] + K[edge_type,edge_index[1]]
        rel_q = torch.where(E ==0, torch.zeros_like(E),(Q.unsqueeze(2).expand(-1,-1,num_nodes)/E)*rel)
        rel_k = torch.where (E ==0 , torch.zeros_like(E), (K.unsqueeze(1).expand(-1,num_nodes,-1)/E)*rel)
  
        Q = torch.matmul(Gmi, q_l2).squeeze() +1e-19
        s_q = (rel_q/Q.unsqueeze(2).expand(-1,-1,num_nodes))
        zkl_q = s_q.mT @ Gmi
        rel_Q = zkl_q * q_l2.mT

        K = torch.matmul(Gmj, k_l2).squeeze() +1e-19
        s_k = (rel_k/(K.unsqueeze(1).expand(-1,num_nodes,-1)))
        zkl_k = s_k @ Gmj
        rel_K = zkl_k * k_l2.mT

                                         
        return rel_Q, rel_K

def lrp(activation, weights, adjacency, relevance):
    #1.Lrp Schritt
    # num_relations = weights.shape[0]
    # num_nodes= adjacency.shape[0]
    Xp = (adjacency @ activation).transpose(0,1)
    sumzk = (Xp @ weights).sum(dim=0)+1e-9
    s = torch.div(relevance,sumzk)
    zkl = s @ weights.mT 
    out = Xp * zkl

    Xp = Xp+1e-9 
    z = out / Xp
    f_out = adjacency.T.transpose(0,1) @ z
    rel_edge = (activation * f_out)
    rel_node = rel_edge.sum(dim=0)
    return rel_node, rel_edge

def lrp2 (activation,weights, adjacency,relevance):
    adj = adjacency.to_dense().view(num_nodes,num_nodes,num_relations)
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
             adjacency, pyk_emb, test_idx, model_name, i, num_nodes, variant='A' ):
    x=i
    selected_rel = relevance[x[1],:]
    num_classes = len(selected_rel)
    rel1 = tensor_max_value_to_1_else_0(selected_rel, x, test_idx, num_nodes, num_classes)
    print(rel1.to_sparse_coo())
    #rel2 = get_relevance_for_dense_layer(act_rgc1, weight_dense, bias_dense, rel1)
    #print(rel2.to_sparse_coo())
    if variant == 'A':
        rel_node, rel_edge = lrp(act_rgc1_no_hidden, weight_rgc1, adjacency, rel1)
        print(rel_node.to_sparse_coo())
        if model_name == 'RGCN_emb':
            rel_node, rel_edge = lrp(pyk_emb, weight_rgc1_no_hidden, adjacency, rel_node)
            print(rel_node.to_sparse_coo())
        elif model_name == 'RGCN_no_emb':
            rel_node, rel_edge = lrp(pyk_emb, weight_rgc1_no_hidden, adjacency, rel_node)
            #rel_node, rel_edge = lrp_first_noemb_layer(adjacency, weight_rgc1_no_hidden, rel_node)
    else:
        rel = lrp2(act_rgc1_no_hidden, weight_rgc1, adjacency, rel1)
        out = lrp2(pyk_emb, weight_rgc1_no_hidden, adjacency, rel)
    return rel_node, rel_edge

def lrp_rgat(parameter_list, input, weight_dense, relevance, s1, s2,x, test_idx, num_nodes, num_relations, edge_type, edge_index,M_l2):
    selected_rel = relevance[x[1],:]
    num_classes = len(selected_rel)
    rel1 = tensor_max_value_to_1_else_0(selected_rel,x, test_idx, num_nodes, num_classes)
    print(rel1.to_sparse_coo())
    # for d in parameter_list:
    #      globals()[d[0]] = d[1]
    rel2 = get_relevance_for_dense_layer(out_l2, weight_dense, None, rel1)
    #llrp_rgat_layer(x, w, alpha, rel, s1, s2, num_neighbors, lrp_step)
    #Second RGAT layer
    r_alpha, rg =  lrp_rgat_layer(out_l1, w_l2, alpha3_l2, rel2, s1, None, None, num_relations, num_nodes, edge_type, edge_index,None, lrp_step='relevance_alphaandg')
    rel_softmax = lrp_rgat_layer(exponential_l2, resmat_l2, None, r_alpha, None, s2, num_neighbors_l2,num_relations, num_nodes, edge_type, edge_index,None, lrp_step='relevance_softmax')
    rel_q, rel_k = lrp_rgat_layer(out_l1, w_l2, None, rel_softmax, None, None, None, num_relations, num_nodes, edge_type, edge_index,M_l2,'relevance_q_k')
    rel_edges, rel_nodes = lrp_rgat_layer(out_l1, w_l2, rel_q, rg, rel_k, None,None,num_relations, num_nodes, edge_type, edge_index,None, lrp_step='relevance_h')
    # First RGAT Layer
    r_alpha, rg =  lrp_rgat_layer(inp_l1, w_l1, alpha3_l1, rel_nodes, s1, None, None, num_relations, num_nodes, edge_type, edge_index,None,lrp_step='relevance_alphaandg')
    rel_softmax = lrp_rgat_layer(exponential_l1, resmat_l1, None, r_alpha, None, s2, num_neighbors_l1,num_relations, num_nodes, edge_type, edge_index,None, lrp_step='relevance_softmax')
    rel_q, rel_k = lrp_rgat_layer(inp_l1, w_l1, None, rel_softmax, None, None, None, num_relations, num_nodes, edge_type, edge_index,M_l2,'relevance_q_k')
    rel_edges, rel_nodes = lrp_rgat_layer(inp_l1, w_l1, rel_q, rg, rel_k, None,None,num_relations, num_nodes, edge_type, edge_index,None,lrp_step='relevance_h')
    return rel_edges, rel_nodes

def tensor_max_value_to_1_else_0(tensor, x, test_idx, num_nodes, num_classes):
    max_value = tensor.argmax()
    idx = test_idx[x[0]]
    t = torch.zeros(num_nodes,num_classes)
    t[idx,max_value] = 1
    return t
def get_highest_relevance(rel):
    global high
    mask = rel == rel.max()
    indices = torch.nonzero(mask)
    high = rel.max()
    return high, indices

def analyse_nodes(homedir, mode, node_table, edge_index, edge_type, model, emb, 
                       parameter_list, input, weight_dense, relevance, adja, activation, 
                       test_idx, model_name,dataset_name,params, num_nodes, num_relations,emb_type, s1, s2,M_l2):
    rel_nodes_list_new, rel_edges_list_new = {}, {}
    pos_min_nodes_new, pos_min_edges_new = {}, {}
    max_nodes_list_new, max_edges_list_new = {}, {}
    pos_max_nodes_new, pos_max_edges_new = {}, {}
    min_nodes_list_new, min_edges_list_new = {}, {}

    for i in node_table.index:
        name_node = str(i)
        name_e = name_node.split('tensor(')[1].split(',')[0]
        print(name_e)
        #name_node for torch.save
        if mode == 'large':
            node_indice = node_table['pos_max_nodes'][i]
        elif mode == 'small':
            node_indice = node_table['pos_min_nodes'][i]
        lrp_node = node_indice
        res_ind = torch.where((input[:,0] == lrp_node) & (input[:,2]== node_indice))[0]
        if res_ind.shape[0] == 0:
            res_ind = torch.where((input[:,2] == lrp_node) & (input[:,0]== node_indice))[0]
            exclude = [node_indice]
            new_index = choice([a for a in range(0,int(input[:,1].max().item())) if a not in [exclude]])
            input_new = input.clone()
            input_new[res_ind,0] = new_index
            edge_index_new = torch.cat([input_new[:,0, None], input_new[:,2, None]], dim=1).T
            print(edge_index_new.shape)
        else:
            exclude = [node_indice]
            new_index = choice([a for a in range(0,int(input[:,1].max().item())) if a not in [exclude]])
            input_new = input.clone()
            input_new[res_ind,2] = new_index
            edge_index_new = torch.cat([input_new[:,0, None], input_new[:,2,None]], dim=1).T
            print(edge_index_new.shape)
            
        if model_name == 'RGCN_emb':
            classes, adj_m, act = model(input_new)
            torch.save(classes, homedir + 'out/'+dataset_name+'/' + model_name + '/pred_after'+str(name_e)+'adapt_'+str(node_indice)+'_'+str(new_index)+'.pt')
            rel_nodes_new, rel_edges_new = lrp_rgcn(activation['rgc1'], model.dense.weight, None, relevance, activation['rgcn_no_hidden'], model.rgc1.weights, 
                        model.rgcn_no_hidden.weights, adja,  emb, test_idx, model_name, i, num_nodes,'A')
        elif model_name == 'RGCN_no_emb':
                
            classes, adj_m, act = model(input_new)
            torch.save(classes, homedir + 'out/'+dataset_name+'/' + model_name + '/pred_after'+str(name_e)+'adapt_'+str(node_indice)+'_'+str(new_index)+'.pt')
            rel_nodes_new, rel_edges_new = lrp_rgcn(activation['rgc1'], model.dense.weight, None, relevance, activation['rgcn_no_hidden'], model.rgc1.weights, 
                        model.rgcn_no_hidden.weights, adja,  activation['input'], test_idx, model_name, i, num_nodes,'A')
        elif model_name == 'RGAT_no_emb':
            pred, params, inp = model(None, edge_index_new, edge_type)
            torch.save(pred, homedir + 'out/'+dataset_name+'/' + model_name + '/pred_after'+str(name_e)+'adapt_'+str(node_indice)+'_'+str(new_index)+'.pt')
            rel_edges_new, rel_nodes_new = lrp_rgat(params, input, weight_dense, relevance, s1, 
                                                    s2, i, test_idx, num_nodes, num_relations, edge_type, edge_index,M_l2) 
        elif model_name == 'RGAT_emb':
            pred, params, inp = model(emb, edge_index_new, edge_type)
            torch.save(pred, homedir + 'out/'+dataset_name+'/' + model_name + '/pred_after'+str(name_e)+'adapt_'+str(node_indice)+'_'+str(new_index)+'.pt')
            rel_edges_new, rel_nodes_new = lrp_rgat(params, input, weight_dense, relevance, s1, 
                                                    s2, i, test_idx, num_nodes, num_relations, edge_type, edge_index,M_l2)
        rel_nodes_new = rel_nodes_new.sum(dim=1)
        rel_nodes_list_new[i] = rel_nodes_new#rel_nodes.argmax(), rel_nodes.max(), rel_nodes.argmin(), rel_nodes.min()
        max_nodes_list_new[i] = rel_nodes_new.max().item()
        pos_max_nodes_new[i] = rel_nodes_new.argmax().item()
        min_nodes_list_new[i] = rel_nodes_new.min().item()
        pos_min_nodes_new[i] = rel_nodes_new.argmin().item()
        
        rel_edges_new = rel_edges_new.sum(dim=2)
        if model_name =='RGAT_emb' or model_name == 'RGCN_emb':
            torch.save(rel_edges_new, homedir + 'out/'+dataset_name+'/' + model_name + '/'+ emb_type+'/relevances/relevances_after_node_edge_'+str(name_e)+'adapt_'+str(node_indice)+'_'+str(new_index)+'.pt')
            torch.save(rel_nodes_new, homedir +'out/' + dataset_name + '/' + model_name + '/'+ emb_type+'/relevances/relevances_after_node_node_'+str(name_e)+'adapt_'+str(node_indice)+'_'+str(new_index)+'.pt')
        else:
            print(name_e, node_indice, new_index)
            torch.save(rel_edges_new, homedir + 'out/'+dataset_name+'/' + model_name +  '/relevances/relevances_after_node_edge_'+str(name_e)+'adapt_'+str(node_indice)+'_'+str(new_index)+'.pt')
            torch.save(rel_nodes_new, homedir +'out/' + dataset_name + '/' + model_name  +'/relevances/relevances_after_node_node_'+str(name_e)+'adapt_'+str(node_indice)+'_'+str(new_index)+'.pt')
        rel_edges_list_new[i] = rel_edges_new
        max_edges_list_new[i] = rel_edges_new.max().item()
        pos_max_edges_new[i] = (rel_edges_new==torch.max(rel_edges_new)).nonzero()
        min_edges_list_new[i] = rel_edges_new.min().item()
        pos_min_edges_new[i] = (rel_edges_new==torch.min(rel_edges_new)).nonzero()
    rel_new = {'rel_nodes_new':rel_nodes_list_new, 'rel_edges_new':rel_edges_list_new, 'max_nodes_new':max_nodes_list_new, 
                'pos_max_nodes_new':pos_max_nodes_new, 'min_nodes_new':min_nodes_list_new, 'pos_min_nodes_new':pos_min_nodes_new, 'max_edges_new':max_edges_list_new, 'pos_max_edges_new':pos_max_edges_new, 'min_edges_new':min_edges_list_new, 'pos_min_edges_new':pos_min_edges_new}
    rel_new_table = pd.DataFrame(rel_new)
    rel_new_table.to_csv(homedir + 'out/'+dataset_name+'/' + model_name + '/'+mode+ '_LRP_nodes_edges_after_node_adaptation.csv')   


def analyse_edges(homedir, mode, edge_table, edge_index, edge_type, model, emb,
                          parameter_list, input, weight_dense, relevance, adja, activation,
                            test_idx, model_name,dataset_name, params,num_nodes, num_relations,emb_type, s1, s2,M_l2):
    rel_nodes_list_new, rel_edges_list_new = {}, {}
    pos_min_nodes_new, pos_min_edges_new = {}, {}
    max_nodes_list_new, max_edges_list_new = {}, {}
    pos_max_nodes_new, pos_max_edges_new = {}, {}
    min_nodes_list_new, min_edges_list_new = {}, {}

    for i in edge_table.index:
        name_edge = str(i)
        #name_edge for torch.save
        name_e = name_edge.split('tensor(')[1].split(',')[0]
        print(name_e)
        if mode == 'large':
            edge_indice = edge_table['pos_max_edges'][i] 
        elif mode == 'small':
            edge_indice = edge_table['pos_min_edges'][i]
        edge_relation = edge_indice[0][0].item()
        edge_node = edge_indice[0][1].item()
        lrp_edge = i[1].item()

        res_ind = torch.where((input[:,0] == edge_node) &  (input[:,1] == edge_relation))[0]
        if res_ind.shape[0] == 0:
            res_ind = torch.where((input[:,2] == edge_node) &  (input[:,1] == edge_relation))[0]
            exclude = edge_relation
            new_type = choice([a for a in range(0,int(input[:,1].max().item())) if a not in [exclude]])
            input_new = input.clone()
            input_new[res_ind,1] = new_type
            edge_type_new = input_new[:,1].T
        else:
            exclude = edge_relation
            new_type = choice([a for a in range(0,int(input[:,1].max().item())) if a not in [exclude]])
            input_new = input.clone()
            input_new[res_ind,1] = new_type
            edge_type_new = input_new[:,1].T

        if model_name == 'RGCN_emb':
            model.eval()
            classes, adj_m, act = model(input_new)
            torch.save(classes, homedir + 'out/'+dataset_name+'/' + model_name + '/pred_after'+str(name_e)+'adapt_'+str(edge_indice[0])+'_'+str(new_type)+'.pt')
            rel_nodes_new, rel_edges_new = lrp_rgcn(activation['rgc1'], model.dense.weight, None, relevance, activation['rgcn_no_hidden'], model.rgc1.weights, 
                        model.rgcn_no_hidden.weights, adja,  emb, test_idx, model_name, i,num_nodes, 'A')
        elif model_name == 'RGCN_no_emb':
            model.eval()
            classes, adj_m, act = model(input_new)
            torch.save(classes, homedir + 'out/'+dataset_name+'/' + model_name + '/pred_after'+str(name_e)+'adapt_'+str(edge_indice[0])+'_'+str(new_type)+'.pt')
            rel_nodes_new, rel_edges_new = lrp_rgcn(activation['rgc1'], model.dense.weight, None, relevance, activation['rgcn_no_hidden'], model.rgc1.weights, 
                        model.rgcn_no_hidden.weights, adja,  activation['input'], test_idx, model_name, i,num_nodes, 'A')
        elif model_name == 'RGAT_no_emb':
            model.eval()
            pred, params, inp = model(None, edge_index, edge_type_new)
            torch.save(pred, homedir + 'out/'+dataset_name+'/' + model_name + '/pred_after'+str(nname_e)+'adapt_'+str(edge_indice[0])+'_'+str(new_type)+'.pt')
            rel_edges_new, rel_nodes_new = lrp_rgat(params, input, weight_dense, relevance, s1, 
                                                    s2, i, test_idx, num_nodes, num_relations, edge_type, edge_index,M_l2) 
        elif model_name == 'RGAT_emb':
            model.eval()
            pred, params, inp = model(emb, edge_index, edge_type_new)
            torch.save(pred, homedir + 'out/'+dataset_name+'/' + model_name + '/pred_after'+str(name_e)+'adapt_'+str(edge_indice[0])+'_'+str(new_type)+'.pt')
            rel_edges_new, rel_nodes_new = lrp_rgat(params, input, weight_dense, relevance, s1, 
                                                    s2, i, test_idx, num_nodes, num_relations, edge_type, edge_index,M_l2)
            
        rel_nodes_new = rel_nodes_new.sum(dim=1)
        rel_nodes_list_new[i] = rel_nodes_new#rel_nodes.argmax(), rel_nodes.max(), rel_nodes.argmin(), rel_nodes.min()
        max_nodes_list_new[i] = rel_nodes_new.max().item()
        pos_max_nodes_new[i] = rel_nodes_new.argmax().item()
        min_nodes_list_new[i] = rel_nodes_new.min().item()
        pos_min_nodes_new[i] = rel_nodes_new.argmin().item()
        
        rel_edges_new = rel_edges_new.sum(dim=2)
        if model_name == 'RGAT_emb' or model_name == 'RGCN_emb':
            torch.save(rel_edges_new, homedir + 'out/'+dataset_name+'/' + model_name + '/'+ emb_type +'/relevances/relevances_after_edge_edge_'+str(name_e)+'_'+ str(edge_node)+'adapt_'+str(edge_indice[0])+'_'+str(new_type)+'.pt')
            torch.save(rel_nodes_new, homedir +'out/' + dataset_name + '/' + model_name+'/'+emb_type  +'/relevances/relevances_after_edge_node_'+str(name_e)+'_'+str(edge_node)+'adapt_'+str(edge_indice[0])+'_'+str(new_type)+'.pt')
        else:
            torch.save(rel_edges_new, homedir + 'out/'+dataset_name+'/' + model_name + '/relevances/relevances_after_edge_edge_'+str(name_e)+'_'+ str(edge_node)+'adapt_'+str(edge_indice[0])+'_'+str(new_type)+'.pt')
            torch.save(rel_nodes_new, homedir +'out/' + dataset_name + '/' + model_name  +'/relevances/relevances_after_edge_node_'+str(name_e)+'_'+str(edge_node)+'adapt_'+str(edge_indice[0])+'_'+str(new_type)+'.pt')
        rel_edges_list_new[i] = rel_edges_new#(rel_edges==torch.max(rel_edges)).nonzero(), rel_edges.max(), (rel_edges==torch.min(rel_edges)).nonzero(), rel_edges.min()
        max_edges_list_new[i] = rel_edges_new.max().item()
        pos_max_edges_new[i] = (rel_edges_new==torch.max(rel_edges_new)).nonzero()
        min_edges_list_new[i] = rel_edges_new.min().item()
        pos_min_edges_new[i] = (rel_edges_new==torch.min(rel_edges_new)).nonzero()
    rel_new = {'rel_nodes_new':rel_nodes_list_new, 'rel_edges_new':rel_edges_list_new, 
                'max_nodes_new':max_nodes_list_new, 
                'pos_max_nodes_new':pos_max_nodes_new, 'min_nodes_new':min_nodes_list_new, 
                'pos_min_nodes_new':pos_min_nodes_new, 'max_edges_new':max_edges_list_new, 
                'pos_max_edges_new':pos_max_edges_new, 'min_edges_new':min_edges_list_new, 
                'pos_min_edges_new':pos_min_edges_new}
    rel_new_table = pd.DataFrame(rel_new)
    rel_new_table.to_csv(homedir + 'out/'+dataset_name+'/' + model_name + '/'+mode+ '_LRP_nodes_edges_after_edge_adaptation.csv') 

def analyse_lrp(emb, edge_index, edge_type, model, parameter_list, input, weight_dense, relevance, test_idx, model_name,dataset_name, num_nodes,num_relations, homedir,emb_type, s1, s2):
    if model_name.startswith('RGCN'):
        pred, adj, activation = model(input)
        adja = adj.to_dense().view(num_nodes,(num_relations*2+1),num_nodes)
        torch.save(pred, homedir + 'out/'+dataset_name+'/'+model_name+ '/pred_before.pt')
        rel_nodes_list, min_nodes_list = {}, {}
        rel_edges_list, pos_min_nodes = {}, {}
        max_nodes_list, max_edges_list = {}, {}
        pos_max_nodes, pos_max_edges, min_edges_list, pos_min_edges = {}, {}, {}, {}
        count_edges_self, count_nodes_self = 0,0
        for i in enumerate(test_idx):
            #name_test_idx for torch.save
            print(i)
            name_e = str(i).split('tensor(')[1].split(',')[0]
            print(name_e)
            relevance, adj,act = model(input)
            if model_name == 'RGCN_emb':
                rel_nodes, rel_edges = lrp_rgcn(activation['rgc1'], model.dense.weight, None, relevance, activation['rgcn_no_hidden'], model.rgc1.weights, 
                            model.rgcn_no_hidden.weights, adja,  emb, test_idx, model_name, i,num_nodes, 'A')
            elif model_name == 'RGCN_no_emb':
                rel_nodes, rel_edges = lrp_rgcn(activation['rgc1'], model.dense.weight, None, relevance, activation['rgcn_no_hidden'], model.rgc1.weights, 
                            model.rgcn_no_hidden.weights, adja,  activation['input'], test_idx, model_name, i,num_nodes, 'A')
            print(rel_nodes.shape, rel_nodes.sum(dim=-1))
            rel_nodes = rel_nodes.sum(dim=1)
            rel_edges = rel_edges.sum(dim=2)
            if emb_type is not None:
                torch.save(rel_nodes, homedir + 'out/'+dataset_name+'/' + model_name + '/'+ emb_type+'/relevances/relevances_node_'+str(name_e)+'.pt')
                torch.save(rel_edges, homedir +'out/' + dataset_name + '/' + model_name +'/' + emb_type +'/relevances/relevances_edge_'+str(name_e)+'.pt')
            else:
                torch.save(rel_nodes, homedir + 'out/'+dataset_name+'/' + model_name + '/relevances/relevances_node_'+str(name_e)+'.pt')
                torch.save(rel_edges, homedir +'out/' + dataset_name + '/' + model_name  +'/relevances/relevances_edge_'+str(name_e)+'.pt')
            rel_nodes_list[i] = rel_nodes.to_sparse_coo()
            rel_edges_list[i] = rel_edges.to_sparse_coo()
            if rel_nodes.argmax().item() == i[1].item():
                count_nodes_self+=1
                print(count_nodes_self)
                rel_nodes[i[1].item()] = 0
                max_nodes_list[i] = rel_nodes.max().item()
                pos_max_nodes[i] = rel_nodes.argmax().item()
                if rel_nodes.argmin().item() == i[1].item():
                    count_nodes_self+=1
                    print(count_nodes_self)
                    rel_nodes[i[1].item()] = 0
                    min_nodes_list[i] = rel_nodes.min().item()
                    pos_min_nodes[i] = rel_nodes.argmin().item()
                else:
                    min_nodes_list[i] = rel_nodes.min().item()
                    pos_min_nodes[i] = rel_nodes.argmin().item()
            else:
                max_nodes_list[i] = rel_nodes.max().item()
                pos_max_nodes[i] = rel_nodes.argmax().item()
                min_nodes_list[i] = rel_nodes.min().item()
                pos_min_nodes[i] = rel_nodes.argmin().item()
            
            if (rel_edges==torch.max(rel_edges)).nonzero()[0][1].item() == i[1].item():
                count_edges_self +=1
                print(count_edges_self)
                rel_edges[(rel_edges==torch.max(rel_edges)).nonzero()[0][0].item(), i[1].item()] = 0
                max_edges_list[i] = rel_edges.max().item()
                pos_max_edges[i] = (rel_edges==torch.max(rel_edges)).nonzero()
                if (rel_edges==torch.min(rel_edges)).nonzero()[0][1].item() == i[1].item():
                    rel_edges[(rel_edges==torch.min(rel_edges)).nonzero()[0][0].item(), i[1].item()] = 0
                    min_edges_list[i] = rel_edges.min().item()
                    pos_min_edges[i] = (rel_edges==torch.min(rel_edges)).nonzero()
                else:
                    min_edges_list[i] = rel_edges.min().item()
                    pos_min_edges[i] = (rel_edges==torch.min(rel_edges)).nonzero()
            else:
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
        n_smallest_nodes = nodes_table.nsmallest(3, 'min_nodes')
        n_largest_edges = edges_table.nlargest(3, 'max_edges')
        n_smallest_edges = edges_table.nsmallest(3, 'min_edges')
        mode = 'large'
        analyse_edges(homedir, mode, n_largest_edges, edge_index, edge_type, model,emb,
                            parameter_list, input, weight_dense, relevance, adja, activation,
                            test_idx, model_name,dataset_name,None, num_nodes,num_relations,emb_type, s1, s2,None)
        analyse_nodes(homedir,mode,  n_largest_nodes, edge_index, edge_type, model,emb, 
                           parameter_list, input, weight_dense, relevance, adja, activation,
                           test_idx, model_name,dataset_name,None, num_nodes,num_relations,emb_type, s1, s2,None)
        mode = 'small'
        analyse_edges(homedir, mode, n_smallest_edges, edge_index, edge_type, model,emb,
                            parameter_list, input, weight_dense, relevance, adja, activation,
                            test_idx, model_name,dataset_name,None, num_nodes,num_relations, emb_type,s1, s2,None)
        analyse_nodes(homedir,mode,  n_smallest_nodes, edge_index, edge_type, model,emb, 
                           parameter_list, input, weight_dense, relevance, adja, activation,
                           test_idx, model_name,dataset_name,None, num_nodes,num_relations, emb_type,s1, s2,None)
        print('count_nodes_self: ',count_nodes_self)
        print('count_edges_self: ', count_edges_self)

    elif model_name.startswith('RGAT'):
        print('Start RGAT_emb')
        for d in parameter_list:
            globals()[d[0]] = d[1]
            if d[0] =='M_l2':
                M_l2 = d[1]
        M_l2 = M_l2.to_dense()
        print('M_l2: ', M_l2)
        pred, params, inp = model(emb, edge_index, edge_type)
        torch.save(pred, homedir + 'out/'+dataset_name+'/'+model_name+ '/pred_before.pt')
        rel_nodes_list, min_nodes_list = {}, {}
        rel_edges_list, pos_min_nodes = {}, {}
        max_nodes_list, max_edges_list = {}, {}
        pos_max_nodes, pos_max_edges, min_edges_list, pos_min_edges = {}, {}, {}, {}
        count_edges_self, count_nodes_self = 0,0
        for i in enumerate(test_idx):
            name_e = i.split('tensor(')[1].split(',')[0]
            print(name_e)
            pred, params, inp = model(emb, edge_index, edge_type)
            rel_edges, rel_nodes = lrp_rgat(params, input, weight_dense, pred, s1, s2, i, test_idx, num_nodes, num_relations, edge_type, edge_index,M_l2)
            rel_nodes = rel_nodes.sum(dim=1)
            rel_edges = rel_edges.sum(dim=2)
            if emb_type is not None:
                torch.save(rel_nodes, homedir + 'out/'+dataset_name+'/' + model_name + '/'+ emb_type+'/relevances/relevances_node_'+str(name_e)+'.pt')
                torch.save(rel_edges, homedir +'out/' + dataset_name + '/' + model_name +'/' + emb_type +'/relevances/relevances_edge_'+str(name_e)+'.pt')
            else:
                torch.save(rel_nodes, homedir + 'out/'+dataset_name+'/' + model_name + '/relevances/relevances_node_'+str(name_e)+'.pt')
                torch.save(rel_edges, homedir +'out/' + dataset_name + '/' + model_name  +'/relevances/relevances_edge_'+str(name_e)+'.pt')
            rel_nodes_list[i] = rel_nodes.to_sparse_coo()
            rel_edges_list[i] = rel_edges.to_sparse_coo()
            if rel_nodes.argmax().item() == i[1].item():
                count_nodes_self+=1
                print(count_nodes_self)
                rel_nodes[i[1].item()] = 0
                max_nodes_list[i] = rel_nodes.max().item()
                pos_max_nodes[i] = rel_nodes.argmax().item()
                if rel_nodes.argmin().item() == i[1].item():
                    count_nodes_self+=1
                    print(count_nodes_self)
                    rel_nodes[i[1].item()] = 0
                    min_nodes_list[i] = rel_nodes.min().item()
                    pos_min_nodes[i] = rel_nodes.argmin().item()
                else:
                    min_nodes_list[i] = rel_nodes.min().item()
                    pos_min_nodes[i] = rel_nodes.argmin().item()
            else:
                max_nodes_list[i] = rel_nodes.max().item()
                pos_max_nodes[i] = rel_nodes.argmax().item()
                min_nodes_list[i] = rel_nodes.min().item()
                pos_min_nodes[i] = rel_nodes.argmin().item()
            if (rel_edges==torch.max(rel_edges)).nonzero()[0][1].item() == i[1].item():
                count_edges_self +=1
                print(count_edges_self)
                rel_edges[(rel_edges==torch.max(rel_edges)).nonzero()[0][0].item(), i[1].item()] = 0
                max_edges_list[i] = rel_edges.max().item()
                pos_max_edges[i] = (rel_edges==torch.max(rel_edges)).nonzero()
                if (rel_edges==torch.min(rel_edges)).nonzero()[0][1].item() == i[1].item():
                    rel_edges[(rel_edges==torch.min(rel_edges)).nonzero()[0][0].item(), i[1].item()] = 0
                    min_edges_list[i] = rel_edges.min().item()
                    pos_min_edges[i] = (rel_edges==torch.min(rel_edges)).nonzero()
                else:
                    min_edges_list[i] = rel_edges.min().item()
                    pos_min_edges[i] = (rel_edges==torch.min(rel_edges)).nonzero()
            else:
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
        n_smallest_nodes = nodes_table.nsmallest(3, 'min_nodes')
        n_largest_edges = edges_table.nlargest(3, 'max_edges')
        n_smallest_edges = edges_table.nsmallest(3, 'min_edges')
        mode = 'large'
        print('Start Analyzing')
        analyse_edges(homedir, mode, n_largest_edges, edge_index, edge_type, model, None,
                            parameter_list, input, weight_dense, pred, None, None,
                            test_idx, model_name,dataset_name,params, num_nodes, num_relations,emb_type, s1, s2,M_l2)
        analyse_nodes(homedir,mode,  n_largest_nodes, edge_index, edge_type, model,None, 
                           parameter_list, input, weight_dense, pred, None, None,
                           test_idx, model_name,dataset_name, params, num_nodes, num_relations,emb_type, s1, s2,M_l2)
        mode = 'small'
        analyse_edges(homedir, mode, n_largest_edges, edge_index, edge_type, model, None,
                            parameter_list, input, weight_dense, pred, None, None,
                            test_idx, model_name,dataset_name,params, num_nodes, num_relations,emb_type, s1, s2,M_l2)
        analyse_nodes(homedir,mode,  n_largest_nodes, edge_index, edge_type, model,None, 
                           parameter_list, input, weight_dense, relevance, None, None,
                           test_idx, model_name,dataset_name,params, num_nodes, num_relations,emb_type, s1, s2,M_l2)
        print('count_nodes_self: ',count_nodes_self)
        print('count_edges_self: ', count_edges_self)