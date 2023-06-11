import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from sklearn.metrics import accuracy_score
import time
import numpy as np
import torch
from utils import utils
from model import rgcn, gat 
from model.rgat import RGAT, RGATLayer
from data.entities import Entities
import os.path as osp

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

def lrp_rgat_layer(x, w, alpha, rel, s1, s2, lrp_step):
    if lrp_step == 'relevance_alphaandg':
        ralpha, rgj = rel* s1, rel*(1-s1)
        g = x @ w
        
        z = alpha @ g.view(24*2835,50) + 1e-10
        s = torch.div(ralpha, z)
        zkl = (s @ g.mT).mT
        out_alpha = alpha * zkl.reshape(24*2835,2835).T

        a = alpha.to_dense().view(2835,2835,24)
        z2 =(a.T @ g).sum(dim=0)+1e-10
        s2 = torch.div(rgj, z2)
        zkl2 = a.T.mT @ s2
        out_g = (zkl2 * g).sum(dim=0)
        print(out_g.to_dense().sum())
        return out_alpha, out_g
    elif lrp_step == 'relevance_h':
        z = (x @ w.mT + 1e-10).sum(dim=0)
        s = torch.div(rel, z) 
        pre_res = s @ w 
        out = x * pre_res 
        return out.sum(dim=0)
    elif lrp_step == 'relevance_softmax':
        reij, reik = rel * s2, rel * (1-s2)

        z = x.to_dense() @ (1-w.to_dense().T) + 1e-10

        return None
    else:

        return None    

    
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
    x=19
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

def lrp_rgat(parameter_list, input, weight_dense, relevance, s1):
    x=19
    selected_rel = relevance[test_idx,:][x]
    rel1 = tensor_max_value_to_1_else_0(selected_rel,x)
    print(rel1.to_sparse_coo())
    for i in parameter_list:
        globals()[i[0]] = i[1]
    rel2 = get_relevance_for_dense_layer(out_l2, weight_dense, None, rel1)
    r_alpha, rg =  lrp_rgat_layer(out_l1, w_l2, alpha_l2,rel2, s1, None, lrp_step='relevance_alphaandg')
    rh = lrp_rgat_layer(input, w_l2, None, rg, None, None, lrp_step='relevance_h')
    rel_softmax = lrp_rgat_layer(adj_eij_l2, adj_index_l2, None, r_alpha, None, 0.2, lrp_step='relevance_softmax')
def tensor_max_value_to_1_else_0(tensor, x):
    max_value = tensor.argmax()
    idx = test_idx[x]
    t = torch.zeros(2835,4)
    t[idx,max_value] = 1
    return t

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
        acc_train = utils.accuracy(output[train_idx], train_lbl)
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
        acc_test = utils.accuracy(output[test_idx], test_lbl)
        print("Test set results:",
        "loss= {:.4f}".format(loss_test.data.item()),
        "accuracy= {:.4f}".format(acc_test.data.item()))
    


    print('Training is complete!')
    return 

def gat_evaluation():
        print("Starting evaluation...")
        model.eval()
        output = model(pyk_emb, adj)
        test_accuracy = utils.accuracy(output[test_idx], test_lbl) # Note: Accuracy is always computed on CPU
        print(f'[Evaluation] Test Accuracy: {test_accuracy:.2f}')

def rgcn_train(epochs):
    optimiser = torch.optim.Adam
    optimiser = optimiser(
    model.parameters(),
    lr=0.1,
    weight_decay=0.0)

    for epoch in range(1, epochs+1):
        t1 = time.time()
        criterion = nn.CrossEntropyLoss()
        model.train()
        optimiser.zero_grad()
        classes, adj_m, val_norm, activation, fw = model()
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
        acc_train = utils.accuracy(classes, train_lbl)
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
        classes = classes[train_idx, :].argmax(dim=-1)
        train_accuracy = accuracy_score(classes.cpu(), train_lbl.cpu()) * 100  # Note: Accuracy is always computed on CPU
        classes, adj, val_norm, activation, fw = model()
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
    classes= classes[test_idx].argmax(dim=-1)
    test_accuracy = accuracy_score(classes.cpu(), test_lbl.cpu()) * 100  # Note: Accuracy is always computed on CPU
    print(f'[Evaluation] Test Accuracy: {test_accuracy:.2f}')


def rgat_train(epochs, pyk_emb, edge_index, edge_type, train_idx, train_y, test_idx, test_y, model):
    #model = RGAT(16, 16, dataset.num_classes, dataset.num_relations).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)
    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad()
        out, parameter_list, input = model(pyk_emb, edge_index, edge_type)
        loss = F.nll_loss(out[train_idx], train_lbl)
        loss.backward()
        optimizer.step()
        loss = float(loss)
    with torch.no_grad():
        model.eval()
        pred, parameter_list, input = model(pyk_emb, edge_index, edge_type)
        weight_dense = model.dense.weight
        lrp_rgat(parameter_list, input, weight_dense, pred, s1 = 0.8)
        pred = pred.argmax(dim=-1)
        
        train_acc = float((pred[train_idx] == train_y).float().mean())
        test_acc = float((pred[test_idx] == test_y).float().mean())
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
    #get_data()
    homedir="C:/Users/luisa/Projekte/Masterthesis/AIFB/"
    adj, triples, (n2i, i2n), (r2i, i2r), train, test, triples_plus = utils.load_data(homedir = "C:/Users/luisa/Projekte/Masterthesis/AIFB")
    pyk_emb = utils.load_pickle(homedir + "data/AIFB/embeddings/pykeen_embedding_TransH")
    pyk_emb = torch.tensor(pyk_emb, dtype=torch.float)
    lemb = len(pyk_emb[1])
    # Check for available GPUs
    use_cuda =  torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    edge_index = triples[:,[0,2]].T
    edge_index = edge_index.type(torch.long)
    edge_index_plus = triples_plus[:,[0,2]].T
    edge_index_plus = edge_index_plus.type(torch.long)
    edge_type = triples[:,1].T
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
    num_nodes = len(n2i)
    num_relations = len(r2i)
    num_rel_plus = len(triples_plus[:,1].unique())
    hidden = 50
    dropout = 0.6
    nb_heads = 4
    alpha = 0.2

    model = 'RGAT'
    if model == 'RGCN':
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
        loss = rgcn_train(epochs=10)
        (act_dense, act_rgc1, act_rgc1_no_hidden, bias_dense, bias_rgc1, 
         bias_rgc1_no_hidden, weight_dense, weight_rgc1, weight_rgc1_no_hidden, 
         relevance, adj, val_norm, activation, fw) =  get_lrp_variables(model)
        lrp_rgcn(act_rgc1, weight_dense, bias_dense, relevance, act_rgc1_no_hidden, weight_rgc1, weight_rgc1_no_hidden, adj, 'A')
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
        epochs= 5
        model = RGAT
        model = model(50, 50, num_classes, num_relations)
        loss, pred, parameter_list = rgat_train(epochs, pyk_emb, edge_index, edge_type, train_idx, train_lbl,test_idx, test_lbl, model)