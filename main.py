import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from sklearn.metrics import accuracy_score
import time
import numpy as np
import torch
from utils import utils_act
from model import gat, rgcn_layers 
import os 
from model import lrp_act
from model.rgat_act import RGAT#, RGATLayer
import pickle
from data.entities import Entities
#from gtn_dataset import IMDBDataset, ACMDataset, DBLPDataset
import os.path as osp
#import torch_geometric
#from torch_geometric.transforms import NormalizeFeatures

# os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'


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
        acc_train = utils_act.accuracy(output[train_idx], train_lbl)
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
        acc_test = utils_act.accuracy(output[test_idx], test_lbl)
        print("Test set results:",
        "loss= {:.4f}".format(loss_test.data.item()),
        "accuracy= {:.4f}".format(acc_test.data.item()))
    print('Training is complete!')
    return 

def gat_evaluation():
        print("Starting evaluation...")
        model.eval()
        output = model(pyk_emb, adj)
        test_accuracy = utils_act.accuracy(output[test_idx], test_lbl) # Note: Accuracy is always computed on CPU
        print(f'[Evaluation] Test Accuracy: {test_accuracy:.2f}')

def rgcn_train(epochs, triples):
    optimiser = torch.optim.Adam
    optimiser = optimiser(
    model.parameters(),
    lr=0.01,
    weight_decay=0.05)

    for epoch in range(1, epochs+1):
        t1 = time.time()
        criterion = nn.CrossEntropyLoss()
        model.train()
        optimiser.zero_grad()
        classes, adj, activation = model(triples)
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
        acc_train = utils_act.accuracy(classes, train_lbl)
        print(classes)
        t2 = time.time()

        loss.backward()
        optimiser.step()
        t3 = time.time()
        print('Epoch: {:04d}'.format(epoch+1),
        'loss: {:.4f}'.format(loss.data.item()),
        'acc_train: {:.4f}'.format(acc_train.data.item()),
        'time: {:.4f}s'.format(time.time() - t1))

    #torch.save(model.state_dict(), homedir +'out/pykeen_model/test.pth')
    params = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            params[name] = param.data

    with torch.no_grad(): #no backpropagation
        model.eval()
        classes, adj,  activation= model(triples)
        classes = classes.to(device)
        classes = classes[train_idx, :].argmax(dim=-1)
        train_accuracy = accuracy_score(classes.cpu(), train_lbl.cpu()) * 100  # Note: Accuracy is always computed on CPU
        classes, adj,  activation= model(triples)
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
    classes, adj_m, activation = model(triples_plus)
    classes = classes.to(device)
    classes= classes[test_idx].argmax(dim=-1)
    test_accuracy = accuracy_score(classes.cpu(), test_lbl.cpu()) * 100  # Note: Accuracy is always computed on CPU
    print(f'[Evaluation] Test Accuracy: {test_accuracy:.2f}')


def rgat_train(epochs, pyk_emb, edge_index, edge_type, train_idx, train_y, test_idx, test_y, triples, model, homedir,emb_type):
    #model = RGAT(16, 16, dataset.num_classes, dataset.num_relations).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)
    for epoch in range(1, epochs+1):
        criterion = nn.CrossEntropyLoss()
        model.train()
        optimizer.zero_grad()
        out, parameter_list, input = model(pyk_emb, edge_index, edge_type)
        out = out.to(device)
        loss = criterion(out[train_idx], train_lbl)
        acc_train = utils_act.accuracy(out[train_idx], train_lbl)
        #print("Loss, epoch: ", loss, epoch)
        print('Epoch: {:04d}'.format(epoch),
        'loss: {:.4f}'.format(loss),
        'acc_train: {:.4f}'.format(acc_train.data.item()))
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        loss = float(loss)
    with torch.no_grad():
        model.eval()
        pred, parameter_list, input = model(pyk_emb, edge_index, edge_type)

        weight_dense = model.dense.weight
        lrp_act.analyse_lrp(pyk_emb, edge_index, edge_type, model, parameter_list, 
                        triples, weight_dense, pred, test_idx, model_name, 
                        dataset_name,num_nodes, num_relations, homedir,emb_type, s1 = 0.8, s2 = 0.2)
        pred2 = pred.argmax(dim=-1)

        #pred2 = pred2.to(device)
        pred2 = pred2[test_idx.cpu()]
        test_accuracy = accuracy_score(pred2.cpu(), test_y.cpu()) * 100 
        print(f'[Evaluation] Test Accuracy: {test_accuracy:.2f}')
        #train_acc.to(device)
        #test_acc.to(decive)
        #train_acc = float((pred2[train_idx] == train_y).float().mean()).to(decive)
        #test_acc = float((pred2[test_idx.cpu()] == test_y.cpu()).float().mean())
        # print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f} '
        #   f'Test: {test_acc:.4f}')
    return loss, pred, parameter_list
        

def get_lrp_variables(model, emb, triples_plus):
    global act_dense, act_rgc1, act_rgc1_no_hidden 
    global bias_dense, bias_rgc1, bias_rgc1_no_hidden, weight_dense 
    global weight_rgc1, weight_rgc1_no_hidden, relevance, adj
    global  activation
    classes, adj, activation = model()
    act_dense = activation['dense']
    act_rgc1 = activation['rgc1']
    act_rgc1_no_hidden = activation['rgcn_no_hidden']
    bias_dense = model.dense.bias
    bias_rgc1 = model.rgc1.bias
    bias_rgc1_no_hidden = model.rgcn_no_hidden.bias
    weight_dense = model.dense.weight
    weight_rgc1 = model.rgc1.weights
    weight_rgc1_no_hidden = model.rgcn_no_hidden.weights
    relevance, adj, activation = model()


if __name__ == '__main__':
    homedir= '/home/luitheob/AIFB/'#C:/Users/luisa/Projekte/Masterthesis/AIFB/'
    datasets = ['AIFB','MUTAG']
    models =   ['RGCN_no_emb', 'RGCN_emb', 'RGAT_no_emb','RGAT_emb'] #['RGCN_no_emb', 'RGCN_emb', 
    embs=[ 'TransE','TransH','DistMult']# ,
    global test_idx, test_y, train_idx, train_y, edge_index, edge_type, pyk_emb
    for dataset_name in datasets:
        print('dataset: ', dataset_name)
        for model_name in models:
            print('model: ', model_name)
            if model_name == 'RGCN_emb' or model_name == 'RGAT_emb':
                for emb_type in embs:
                    print('emb_type: ', emb_type)
                    pyk_emb = utils_act.load_pickle(homedir + "data/"+dataset_name+"/embeddings/pykeen_embedding_"+emb_type+".pickle")
                    pyk_emb = torch.tensor(pyk_emb, dtype=torch.float)
                    lemb = len(pyk_emb[1])
                    if model_name.startswith('RGCN'):
                        epochs = 50
                    elif model_name.startswith('RGAT'):
                         epochs = 25
                    
                    adj, edges, (n2i, i2n), (r2i, i2r), train, test, triples, triples_plus = utils_act.load_data(homedir, dataset_name, emb_type, model_name)

                    use_cuda =  torch.cuda.is_available()
                    print('cuda: ',use_cuda)

                    device = torch.device('cuda:7' if use_cuda else 'cpu')
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
                    with open(homedir + '/out/'+ dataset_name+'/'+ model_name+ '/' + emb_type+'/edge_type_plus.pkl', 'wb') as fp:
                        pickle.dump(edge_type_plus, fp)
                    with open(homedir + '/out/'+ dataset_name+'/'+ model_name+ '/' + emb_type+'/edge_index_plus.pkl', 'wb') as fp:
                        pickle.dump(edge_index_plus, fp)
                    with open (homedir + '/out/'+ dataset_name+'/'+ model_name+ '/'+ emb_type+'/test_idx.pkl', 'wb') as fp:
                        pickle.dump(test_idx, fp)
                    with open (homedir + '/out/'+ dataset_name+'/'+ model_name+ '/'+ emb_type+'/test_lbl.pkl', 'wb') as fp:
                        pickle.dump(test_lbl, fp)
                    with open (homedir + '/out/'+ dataset_name+'/'+ model_name+ '/'+ emb_type+'/train_idx.pkl', 'wb') as fp:
                        pickle.dump(test_idx, fp)
                    with open (homedir + '/out/'+ dataset_name+'/'+ model_name+ '/'+ emb_type+'/train_lbl.pkl', 'wb') as fp:
                        pickle.dump(test_lbl, fp)
                    print('test_idx: ',test_idx)
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
                    
                    if model_name == 'RGCN_emb':
                        model = rgcn_layers.NodeClassifier
                        model = model(
                            nnodes=num_nodes,
                            nrel=num_relations,
                            nfeat = len(pyk_emb[0]),
                            nclass=num_classes,
                            nhid=50,
                            nlayers=2,
                            decomposition=None,
                            nemb=pyk_emb)
                        loss = rgcn_train(epochs, triples_plus)
                        lrp_act.analyse_lrp(pyk_emb, edge_index, edge_type, model, None, triples_plus, None, None, test_idx, model_name, dataset_name, num_nodes, num_relations, homedir, emb_type, None, None)
                        rgcn_evaluation(pyk_emb, triples_plus)
                    elif model_name== 'RGAT_emb':
                        model = RGAT
                        model = model(50, 50, num_classes, num_relations, num_nodes)
                        loss, pred, parameter_list = rgat_train(epochs, pyk_emb, edge_index, edge_type, train_idx, train_lbl,test_idx, test_lbl,edges, model, homedir,emb_type)

            else:
                pyk_emb = None
                lemb = None
                adj, edges, (n2i, i2n), (r2i, i2r), train, test, triples, triples_plus = utils_act.load_data(homedir, dataset_name, None, model_name)
                if model_name.startswith('RGCN'):
                    epochs = 50
                elif model_name.startswith('RGAT'):
                    epochs = 25
                # Check for available GPUs
                use_cuda =  torch.cuda.is_available()
                print('cuda: ',use_cuda)
                device = torch.device('cuda:7' if use_cuda else 'cpu')
                print('shape edges: ', edges.shape)
                edge_index = edges[:,[0,2]].T
                edge_index = edge_index.type(torch.long)
                edge_index_plus = triples_plus[:,[0,2]].T
                with open(homedir + '/out/'+ dataset_name+'/'+ model_name+'/edge_index_plus.pkl', 'wb') as fp:
                    pickle.dump(edge_index_plus, fp)
                edge_index_plus = edge_index_plus.type(torch.long)
                edge_type = edges[:,1].T
                edge_type = edge_type.to(torch.long)
                edge_type_plus = triples_plus[:,1].T
                edge_type_plus = edge_type_plus.to(torch.long)
                with open(homedir + '/out/'+ dataset_name+'/'+ model_name+'/edge_type_plus.pkl', 'wb') as fp:
                    pickle.dump(edge_type_plus, fp)

                # Convert train and test datasets to torch tensors
                train_idx = [n2i[name] for name, _ in train.items()]
                train_lbl = [cls for _, cls in train.items()]
                train_idx = torch.tensor(train_idx, dtype=torch.long, device=device)
                train_lbl = torch.tensor(train_lbl, dtype=torch.long, device=device)

                test_idx = [n2i[name] for name, _ in test.items()]
                test_lbl = [cls for _, cls in test.items()]
                test_idx = torch.tensor(test_idx, dtype=torch.long, device=device)
                test_lbl = torch.tensor(test_lbl, dtype=torch.long, device=device)

                with open (homedir + '/out/'+ dataset_name+'/'+ model_name+'/test_idx.pkl', 'wb') as fp:
                    pickle.dump(test_idx, fp)
                with open (homedir + '/out/'+ dataset_name+'/'+ model_name+'/test_lbl.pkl', 'wb') as fp:
                    pickle.dump(test_lbl, fp)
                with open (homedir + '/out/'+ dataset_name+'/'+ model_name+'/train_idx.pkl', 'wb') as fp:
                    pickle.dump(test_idx, fp)
                with open (homedir + '/out/'+ dataset_name+'/'+ model_name+'/train_lbl.pkl', 'wb') as fp:
                    pickle.dump(test_lbl, fp)





                print('test_idx: ',test_idx)

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

                if model_name == 'RGCN_no_emb':
                    model = rgcn_layers.NodeClassifier
                    model = model(
                        nnodes=num_nodes,
                        nrel=num_relations,            
                        nfeat = 50,
                        nclass=num_classes,
                        nhid=50,
                        nlayers=2,
                        decomposition=None,
                        nemb=None)
                    loss = rgcn_train(epochs, triples_plus)
                    lrp_act.analyse_lrp(None, edge_index, edge_type, model, None, triples_plus, None, None, test_idx, model_name, dataset_name, num_nodes, num_relations, homedir, None, None, None)
                    rgcn_evaluation(None, triples_plus)
                

                elif model_name== 'RGAT_no_emb':
                    model = RGAT
                    model = model(50, 50, num_classes, num_relations, num_nodes)
                    loss, pred, parameter_list = rgat_train(epochs, None, edge_index, edge_type, train_idx, train_lbl,test_idx, test_lbl, edges, model, homedir,None)
