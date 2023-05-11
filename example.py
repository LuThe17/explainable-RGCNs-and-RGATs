import numpy as np
import torch
import os
from torch.nn.parameter import Parameter
from torch.nn import functional as F

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def get_relevance_for_dense_layer(a, w, b, rel_in):
    '''
    :param a: activations of the layer
    :param w: weights of the layer
    :param b: bias of the layer
    :param rel_in: relevance of the input of the layer
    :return: relevance of the output of the layer
    '''
    z = a.dot(w.T)
    s = np.divide(rel_in,z)
    pre_res = s.detach().numpy().dot(w)
    out = np.multiply(a, pre_res)
    return out

# Input
X = np.array([[1, 2], [4, 5], [3,6]])

# Adjazenzmatrizen
A1 = np.array([[[0, 0, 1], [1, 0, 1], [0, 1, 0]],[[0, 1, 1], [0,0,0], [1,0,0]]])

# Gewichte
W1 = torch.FloatTensor([[[0.3,0.1],[0.4,0.1]],[[0.1,0.5],[0.1, 0.1]]])#Parameter(torch.FloatTensor(2, 2, 2))
W2 = torch.FloatTensor([[[0.2,0.1],[0.1,0.3]],[[0.2,0.1],[0.4, 0.2]]])#Parameter(torch.FloatTensor(2,2,2))

# Dense Layer Gewichte
#W_dense = np.array([[0.1, 0.2, 0.1], [0.3, 0.4, 0.1], [0.5, 0.6, 0.1]])

# Softmax Gewichte
W_softmax = np.array([[0.5, 0.6, 0.1]])

# Berechnung der ersten Schicht mit RGCN
H1 = np.dot(X,W1.detach().numpy()).reshape(3*2,2)
H1 = np.dot(A1.reshape(3,3*2), H1)
H1_relu = np.maximum(H1, 0)

# Berechnung der zweiten Schicht mit RGCN
H2 = np.dot(H1_relu, W2.detach().numpy()).reshape(3*2,2)
H2 = np.dot(A1.reshape(3,3*2), H2)
#H2_relu = np.maximum(H2, 0)


# Berechnung der Softmax Schicht
softmax_out = F.softmax(torch.Tensor(H2),dim=1)
# softmax_out = np.dot(H2_relu, W_softmax.transpose())
# softmax_out_exp = np.exp(softmax_out)
# softmax_out_sum = np.sum(softmax_out_exp)
# softmax_out_norm = softmax_out_exp / softmax_out_sum

print(softmax_out)

probs = torch.Tensor(softmax_out)
selected_rel=probs[0]

def tensor_max_value_to_1_else_0(tensor, x=0):
        max_value = tensor.argmax()
        idx = 0
        t = torch.zeros(3,2)
        t[idx,max_value] = 1
        return t

rel1 =tensor_max_value_to_1_else_0(selected_rel,0)


def lrp(activation, weights, adjacency, relevance):
    #1.Lrp Schritt
    #adj = adjacency.to_dense().reshape(2835,49,2835)
    Xp = (adjacency @ activation)#.reshape(49, 2835, 50)
    sumzk = (torch.Tensor(Xp) @ weights.mT).sum(dim=0)+1e-9
    s = torch.div(relevance,sumzk)
    zkl = s @ weights 
    out = Xp*(zkl.detach().numpy()) 

    #2.Lrp Schritt
    Xp = Xp+1e-9 
    z = out / Xp
    f_out = adjacency.transpose(0,2,1) @ z
    rel = (activation * f_out)#.sum(dim=0)

    # Xp2 = (adjacency.mT @ activation)
    # sumzk2 = (Xp2 @ weights).sum(dim=0)+1e-9
    # s2 = torch.div(relevance,sumzk2)
    # zkl2 = s2 @ weights 
    # out2 = Xp2.mul(zkl2) 

    

    # Xp2 = Xp2+1e-9 
    # z2 = out2 / Xp2
    # f_out2 = adjacency.T.transpose(0,1) @ z2
    # rel2 = (activation * f_out2).sum(dim=0)
    return rel

out = lrp(H1_relu, W2, A1, rel1)
res = lrp(X,W1,A1, torch.Tensor(out))

print("Output probabilities:", probs)
