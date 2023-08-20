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
from model import rgcn_gpu_lrp, gat 
from archiv import lrp
from model.rgat import RGAT, RGATLayer
from data.entities import Entities
#from gtn_dataset import IMDBDataset, ACMDataset, DBLPDataset
import os.path as osp
import torch_geometric
from torch_geometric.transforms import NormalizeFeatures
import networkx as ntx
import matplotlib.pyplot as plt

def get_data():
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Entities')
    dataset = Entities(path, 'AIFB')
    data = dataset[0]
    data.x = torch.randn(data.num_nodes, 16)

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

def rgcn_train(epochs, emb, triples):
    optimiser = torch.optim.Adam
    optimiser = optimiser(
    model.parameters(),
    lr=0.02,
    weight_decay=0.05)

    for epoch in range(1, epochs+1):
        t1 = time.time()
        criterion = nn.CrossEntropyLoss()
        model.train()
        optimiser.zero_grad()
        classes, adj_m, activation = model(emb,triples)
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
        classes, adj,  activation = model(emb, triples)
        classes = classes.to(device)
        classes = classes[train_idx, :].argmax(dim=-1)
        train_accuracy = accuracy_score(classes.cpu(), train_lbl.cpu()) * 100  # Note: Accuracy is always computed on CPU
        classes, adj, activation = model(emb, triples)
        classes = classes.to(device)
        classes = classes[test_idx, :].argmax(dim=-1)
        test_accuracy = accuracy_score(classes.cpu(), test_lbl.cpu()) * 100  # Note: Accuracy is always computed on CPU

        print(f'[Epoch {epoch}] Loss: {loss.item():.5f} Forward: {(t2 - t1):.3f}s Backward: {(t3 - t2):.3f}s '
              f'Train Accuracy: {train_accuracy:.2f} Test Accuracy: {test_accuracy:.2f}')

    print('Training is complete!')
    return 

def rgcn_evaluation(x, triples):
    print("Starting evaluation...")
    model.eval()
    classes, adj_m, activation = model(x, triples)
    classes = classes.to(device)
    classes= classes[test_idx].argmax(dim=-1)
    test_accuracy = accuracy_score(classes.cpu(), test_lbl.cpu()) * 100  # Note: Accuracy is always computed on CPU
    print(f'[Evaluation] Test Accuracy: {test_accuracy:.2f}')


def rgat_train(epochs, pyk_emb, edge_index, edge_type, train_idx, train_y, test_idx, test_y, triples, model):
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
        lrp.analyse_lrp(pyk_emb, edge_index, edge_type, model, parameter_list, 
                        triples, weight_dense, pred, test_idx, model_name, 
                        dataset_name,num_nodes, num_relations, s1 = 0.8, s2 = 0.2)
        lrp.lrp_rgat(parameter_list, input, weight_dense, pred, s1 = 0.8)
        pred2 = pred.argmax(dim=-1)
        
        train_acc = float((pred2[train_idx] == train_y).float().mean())
        test_acc = float((pred2[test_idx] == test_y).float().mean())
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f} '
          f'Test: {test_acc:.4f}')
    return loss, pred, parameter_list
        

def get_lrp_variables(model, emb, triples_plus):
    global act_dense, act_rgc1, act_rgc1_no_hidden 
    global bias_dense, bias_rgc1, bias_rgc1_no_hidden, weight_dense 
    global weight_rgc1, weight_rgc1_no_hidden, relevance, adj
    global  activation
    classes, adj, activation = model(emb, triples_plus)
    act_dense = activation['dense']
    act_rgc1 = activation['rgc1']
    act_rgc1_no_hidden = activation['rgcn_no_hidden']
    bias_dense = model.dense.bias
    bias_rgc1 = model.rgc1.bias
    bias_rgc1_no_hidden = model.rgcn_no_hidden.bias
    weight_dense = model.dense.weight
    weight_rgc1 = model.rgc1.weights
    weight_rgc1_no_hidden = model.rgcn_no_hidden.weights
    relevance, adj, activation = model(emb, triples_plus)



if __name__ == '__main__':
    homedir='C:/Users/luisa/Projekte/Masterthesis/AIFB/'
    dataset_name = 'AIFB'
    global test_idx, test_y, train_idx, train_y, edge_index, edge_type, pyk_emb
    adj, edges, (n2i, i2n), (r2i, i2r), train, test, triples, triples_plus = utils_gpu.load_data(homedir = 'C:/Users/luisa/Projekte/Masterthesis/AIFB/', filename= dataset_name)
    pyk_emb = utils_gpu.load_pickle(homedir + "data/AIFB/embeddings/pykeen_embedding_DistMult.pickle")
    pyk_emb = torch.tensor(pyk_emb, dtype=torch.float)
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
    epochs = 2
    
    model_name = 'RGAT_no_emb'
    if model_name == 'RGCN_emb':
        model = rgcn_gpu_lrp.EmbeddingNodeClassifier
        model = model(
            nnodes=num_nodes,
            nrel=num_relations,
            nclass=num_classes,
            nhid=50,
            nlayers=2,
            decomposition=None,
            nemb=lemb)
        loss = rgcn_train(epochs, pyk_emb, triples_plus)
        lrp.analyse_lrp(pyk_emb, edge_index, edge_type, model, None, triples_plus, None, None, test_idx, model_name, dataset_name, num_nodes, num_relations,None, None)
        rgcn_evaluation(pyk_emb, triples_plus)

    elif model_name == 'RGCN_no_emb':
        model = rgcn_gpu_lrp.EmbeddingNodeClassifier
        model = model(
            nnodes=num_nodes,
            nrel=num_relations,
            nclass=num_classes,
            nhid=50,
            nlayers=2,
            decomposition=None,
            nemb=lemb)
        loss = rgcn_train(epochs, None, triples_plus)
        get_lrp_variables(model, None, triples_plus)
        lrp.analyse_lrp(None, edge_index, edge_type, model, None, triples_plus, None, None, test_idx, model_name, dataset_name, num_nodes, num_relations,None, None)
        #lrp.lrp_rgcn(act_rgc1, weight_dense, bias_dense, relevance, act_rgc1_no_hidden, weight_rgc1, weight_rgc1_no_hidden, adj,  pyk_emb, test_idx, model_name, 'A')
        rgcn_evaluation(None, triples_plus)
    elif model_name == 'GAT':
        model = gat.GAT
        model = model(nfeat=lemb, 
                nhid=hidden, 
                nclass=num_classes, 
                dropout=dropout, 
                alpha=alpha)
        loss = gat_train(epochs=4)
        gat_evaluation()
    
    elif model_name== 'RGAT_emb':
        epochs= 1
        model = RGAT
        model = model(50, 50, num_classes, num_relations, num_nodes)
        loss, pred, parameter_list = rgat_train(epochs, pyk_emb, edge_index, edge_type, train_idx, train_lbl,test_idx, test_lbl,edges, model)
    
    elif model_name== 'RGAT_no_emb':
        epochs= 1
        model = RGAT
        model = model(50, 50, num_classes, num_relations, num_nodes)
        loss, pred, parameter_list = rgat_train(epochs, None, edge_index, edge_type, train_idx, train_lbl,test_idx, test_lbl, edges, model)