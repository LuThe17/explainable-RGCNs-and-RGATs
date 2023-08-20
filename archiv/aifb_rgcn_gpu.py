import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from sklearn.metrics import accuracy_score
import time
import numpy as np
import torch
from utils import utils_gpu
from model import gat, rgcn_gpu
from model.rgat import RGAT, RGATLayer
from data.entities import Entities
from gtn_dataset import IMDBDataset, ACMDataset, DBLPDataset
import os.path as osp
import torch_geometric
from torch_geometric.transforms import NormalizeFeatures

def get_data():
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Entities')
    dataset = Entities(path, 'AIFB')
    data = dataset[0]
    data.x = torch.randn(data.num_nodes, 16)


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

def lrp_rgcn(act_rgc1, weight_dense, bias_dense, relevance, act_rgc1_no_hidden, weight_rgc1, weight_rgc1_no_hidden, adjacency, variant='A'):
    x=10
    selected_rel = relevance[test_idx, :][x]
    rel1 = tensor_max_value_to_1_else_0(selected_rel,x)
    print(rel1.to_sparse_coo())
    rel2 = get_relevance_for_dense_layer(act_rgc1, weight_dense, bias_dense, rel1)
    print(rel2.to_sparse_coo())
    if variant == 'A':
        rel = lrp(act_rgc1_no_hidden, weight_rgc1, adjacency, rel2)
        print(rel.to_sparse_coo())
        out = lrp(pyk_emb, weight_rgc1_no_hidden, adjacency, rel)
        print(out.to_sparse_coo())
    else:
        rel = lrp2(act_rgc1_no_hidden, weight_rgc1, adjacency, rel2)
        out = lrp2(pyk_emb, weight_rgc1_no_hidden, adjacency, rel)
    return out

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
def tensor_max_value_to_1_else_0(tensor, x):
    max_value = tensor.argmax()
    idx = test_idx[x]
    t = torch.zeros(2835,4)
    t[idx,max_value] = 1
    return t
def get_highest_relevance(rel):
    global high
    mask = rel == rel.max()
    indices = torch.nonzero(mask)
    high = rel.max()
    return high, indices

def analyse_lrp(emb, edge_index, edge_type, model, parameter_list, input, weight_dense, relevance, s1, s2):
    for i in enumerate(test_idx):
        rel = lrp_rgat(parameter_list, input, weight_dense, relevance, s1, s2,i)
        high, indices = get_highest_relevance(rel)
        analyse_highest_relevance(rel)
        change_highest_relevance(rel, high)
        predict_new_result(rel,high,change)
def gat_train(epochs):
    optimiser = torch.optim.Adam
    optimiser = optimiser(
        model.parameters(),
        lr=0.1,
        weight_decay=0.0)

    #Training
    for epoch in range(1, epochs+1):
        t1 = time.time()
        #criterion = nn.CrossEntropyLoss()
        model.train()
        optimiser.zero_grad()
        output = model(pyk_emb, adj)
        loss_train = F.nll_loss(output[train_idx], train_lbl)
        acc_train = utils_gpu.accuracy(output[train_idx], train_lbl)
        loss_train.backward()
        optimiser.step()
        t3 = time.time()
        print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'time: {:.4f}s'.format(time.time() - t1))
    with torch.no_grad():
        model.eval()
        output = model(pyk_emb, adj)
        loss_test = F.nll_loss(output[test_idx], test_lbl)
        acc_test = utils_gpu.accuracy(output[test_idx], test_lbl)
        print("Test set results:",
        "loss= {:.4f}".format(loss_test.data.item()),
        "accuracy= {:.4f}".format(acc_test.data.item()))
    print('Training is complete!')
    return 

def gat_evaluation():
        print("Starting evaluation...")
        model.eval()
        output = model(pyk_emb, adj)
        test_accuracy = utils_gpu.accuracy(output[test_idx], test_lbl) # Note: Accuracy is always computed on CPU
        print(f'[Evaluation] Test Accuracy: {test_accuracy:.2f}')

def rgcn_train(epochs):
    optimiser = torch.optim.Adam
    optimiser = optimiser(
    model.parameters(),
    lr=0.01,
    weight_decay=0.05)
    #train_idx= train_idx.to(device)
    for epoch in range(1, epochs+1):
        t1 = time.time()
        criterion = nn.CrossEntropyLoss()
        model.train()
        optimiser.zero_grad()
        classes, adj_m, val_norm, activation, fw = model()
        classes = classes.to(device)
        
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
        acc_train = utils_gpu.accuracy(classes, train_lbl)
        print(classes)
        t2 = time.time()

        loss.backward()
        optimiser.step()
        t3 = time.time()
        print('Epoch: {:04d}'.format(epoch+1),
        'loss: {:.4f}'.format(loss.data.item()),
        'acc_train: {:.4f}'.format(acc_train.data.item()),
        'time: {:.4f}s'.format(time.time() - t1))

    torch.save(model.state_dict(), homedir +'out/pykeen_model/test.pth')
    params = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            params[name] = param.data

    with torch.no_grad(): #no backpropagation
        model.eval()
        classes, adj, val_norm, activation, fw = model()
        classes = classes.to(device)
        classes = classes[train_idx, :].argmax(dim=-1)
        train_accuracy = accuracy_score(classes.cpu(), train_lbl.cpu()) * 100  # Note: Accuracy is always computed on CPU
        classes, adj, val_norm, activation, fw = model()
        classes = classes.to(device)
        classes = classes[test_idx, :].argmax(dim=-1)
        test_accuracy = accuracy_score(classes.cpu(), test_lbl.cpu()) * 100  # Note: Accuracy is always computed on CPU

        print(f'[Epoch {epoch}] Loss: {loss.item():.5f} Forward: {(t2 - t1):.3f}s Backward: {(t3 - t2):.3f}s '
              f'Train Accuracy: {train_accuracy:.2f} Test Accuracy: {test_accuracy:.2f}')

    print('Training is complete!')
    return 

def rgcn_evaluation():
    print("Starting evaluation...")
    model.eval()
    classes, adj_m, val_norm, activation, fw = model()
    classes = classes.to(device)
    classes= classes[test_idx].argmax(dim=-1)
    test_accuracy = accuracy_score(classes.cpu(), test_lbl.cpu()) * 100  # Note: Accuracy is always computed on CPU
    print(f'[Evaluation] Test Accuracy: {test_accuracy:.2f}')


def rgat_train(epochs, pyk_emb, edge_index, edge_type, train_idx, train_y, test_idx, test_y, model):
    #model = RGAT(16, 16, dataset.num_classes, dataset.num_relations).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)
    for epoch in range(1, epochs+1):
        criterion = nn.CrossEntropyLoss()
        model.train()
        optimizer.zero_grad()
        out, parameter_list, input = model(pyk_emb, edge_index, edge_type)
        out = out.to(device)
        loss = criterion(out[train_idx], train_lbl)
        print("Loss, epoch: ", loss, epoch)
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        loss = float(loss)
    with torch.no_grad():
        model.eval()
        pred, parameter_list, input = model(pyk_emb, edge_index, edge_type)

        weight_dense = model.dense.weight
        analyse_lrp(pyk_emb, edge_index, edge_type, model, parameter_list, input, weight_dense, pred, s1 = 0.8, s2 = 0.2)
        lrp_rgat(parameter_list, input, weight_dense, pred, s1 = 0.8)
        pred2 = pred.argmax(dim=-1)
        
        train_acc = float((pred2[train_idx] == train_y).float().mean())
        test_acc = float((pred2[test_idx] == test_y).float().mean())
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f} '
          f'Test: {test_acc:.4f}')
    return loss, pred, parameter_list
        



def get_lrp_variables(model):
    classes, adj_m, val_norm, activation, fw = model()
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
    return act_dense, act_rgc1, act_rgc1_no_hidden, bias_dense, bias_rgc1, bias_rgc1_no_hidden, weight_dense, weight_rgc1, weight_rgc1_no_hidden, relevance, adj, val_norm, activation, fw



if __name__ == '__main__':
    homedir='/pfs/work7/workspace/scratch/ma_luitheob-master/AIFB/'
    adj, edges, (n2i, i2n), (r2i, i2r), train, test, triples, triples_plus = utils_gpu.load_data(homedir = '/pfs/work7/workspace/scratch/ma_luitheob-master/AIFB', filename= 'BGS')
    pyk_emb = utils_gpu.load_pickle(homedir + "data/AIFB/embeddings/pykeen_embedding_DistMult.pickle")
    pyk_emb = torch.tensor(pyk_emb, dtype=torch.float)
    #dataet = IMDBDataset()
    #g = dataet[0]
    lemb = len(pyk_emb[1])
    #dataset = torch_geometric.datasets.IMDB(root='data/IMDB')
    # Check for available GPUs
    use_cuda =  torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print('shape edges: ', edges.shape)
    edge_index = edges[:,[0,2]].T
    edge_index = edge_index.type(torch.long)
    edge_index_plus = triples_plus[:,[0,2]].T
    edge_index_plus = edge_index_plus.type(torch.long)
    edge_type = edges[:,1].T
    edge_type = edge_type.to(torch.long)
    edge_type_plus = triples_plus[:,1].T
    edge_type_plus = edge_type_plus.to(torch.long)



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
    print('num_classes: ', num_classes)
    num_nodes = len(n2i)
    print('num_nodes: ', num_nodes)
    num_relations = len(r2i)
    print('num_relations: ', num_relations)
    num_rel_plus = len(triples_plus[:,1].unique())
    hidden = 50
    dropout = 0.6
    nb_heads = 1
    alpha = 0.2

    model = 'RGCN_no_emb'
    if model == 'RGCN_emb':
        model = rgcn_gpu.EmbeddingNodeClassifier
        model = model(
            triples=edges,
            nnodes=num_nodes,
            nrel=num_relations,
            nclass=num_classes,
            nhid=50,
            nlayers=2,
            decomposition=None,
            nemb=lemb,
            emb=pyk_emb)
        loss = rgcn_train(epochs=50)
        (act_dense, act_rgc1, act_rgc1_no_hidden, bias_dense, bias_rgc1, 
         bias_rgc1_no_hidden, weight_dense, weight_rgc1, weight_rgc1_no_hidden, 
         relevance, adj, val_norm, activation, fw) =  get_lrp_variables(model)
        lrp_rgcn(act_rgc1, weight_dense, bias_dense, relevance, act_rgc1_no_hidden, weight_rgc1, weight_rgc1_no_hidden, adj, 'A')
        rgcn_evaluation()
    elif model == 'RGCN_no_emb':
        model = rgcn_gpu.NodeClassifier
        model = model(
            triples = edges,
            nnodes=num_nodes,
            nrel=num_relations,
            nclass=num_classes,
            nhid=16,
            nlayers=2,
            decomposition=None)
        loss = rgcn_train(epochs=50)
        rgcn_evaluation()
    elif model == 'GAT':
        model = gat.GAT
        model = model(nfeat=lemb, 
                nhid=hidden, 
                nclass=num_classes, 
                dropout=dropout, 
                alpha=alpha)
        loss = gat_train(epochs=4)
        gat_evaluation()
    
    elif model== 'RGAT':
        epochs= 2
        model = RGAT
        model = model(50, 50, num_classes, num_relations)
        loss, pred, parameter_list = rgat_train(epochs, pyk_emb, edge_index, edge_type, train_idx, train_lbl,test_idx, test_lbl, model)