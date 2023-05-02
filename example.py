import numpy as np
import torch
import os
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
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Adjazenzmatrizen
A1 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
#A2 = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])

# Gewichte
W1 = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.4]])
W2 = np.array([[0.2, 0.3], [0.4, 0.5], [0.6, 0.7]])

# Dense Layer Gewichte
W_dense = np.array([[0.1, 0.2, 0.1], [0.3, 0.4, 0.1], [0.5, 0.6, 0.1]])

# Softmax Gewichte
W_softmax = np.array([[0.5, 0.6, 0.1]])

# Berechnung der ersten Schicht mit RGCN
H1 = np.dot(A1, X).dot(W1)
H1_relu = np.maximum(H1, 0)

# Berechnung der zweiten Schicht mit RGCN
H2 = (A1 @ H1_relu) @ W2.transpose()
H2_relu = np.maximum(H2, 0)


# Berechnung der Softmax Schicht
softmax_out = np.dot(H2_relu, W_softmax.transpose())
softmax_out_exp = np.exp(softmax_out)
softmax_out_sum = np.sum(softmax_out_exp)
softmax_out_norm = softmax_out_exp / softmax_out_sum

print(softmax_out_norm)

probs = torch.Tensor(softmax_out_norm)
def tensor_max_value_to_1_else_0(tensor):
        max_value = tensor.argmax()
        idx = softmax_out_norm[0]
        t = torch.zeros(3,1)
        t[idx,max_value] = 1
        #torch.where(tensor == max_value, torch.tensor(1), torch.tensor(0.0))
        return t
rel1 =tensor_max_value_to_1_else_0(probs)

get_relevance_for_dense_layer(H2_relu, W_softmax, 0, rel1)

print("Output probabilities:", probs)
