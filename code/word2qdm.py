import numpy as np

import pennylane as qml
from pennylane.templates.layers import BasicEntanglerLayers

import torch
import torch.nn as nn
import torch.nn.functional as F


class Word2QDM(nn.Module):
    def __init__(self, vocab_size, layers, num_qubits, num_dm_qubits):
        super(Word2QDM, self).__init__()
        self.vocab_size = vocab_size
        self.layers = layers
        self.num_qubits = num_qubits
        self.num_dm_qubits = num_dm_qubits
        self.dm_shape = (num_dm_qubits**2, num_dm_qubits**2)
        self.target_params = nn.Parameter(
            torch.empty((vocab_size, layers, num_qubits)).normal_(mean=0, std=np.pi)
        )
        self.context_params = nn.Parameter(
            torch.empty((vocab_size, layers, num_qubits)).normal_(mean=0, std=np.pi)
        )
        self.device = qml.device("default.qubit", wires=num_qubits)
        self.circuit = qml.QNode(
            self.qmodel, self.device, interface="torch", diff_method="backprop"
        )

    def qmodel(self, weights):
        BasicEntanglerLayers(weights, wires=range(self.num_qubits))
        return qml.density_matrix(range(self.num_dm_qubits))

    def forward(self, target_indices, context_indices, neg_indices):
        loss = 0
        for i, t in enumerate(target_indices):
            A_target = self.circuit(self.target_params[t]).reshape(self.dm_shape)
            A_context = self.circuit(self.context_params[context_indices[i]]).reshape(
                self.dm_shape
            )
            # TODO: enable multiple neg_samples
            A_neg = self.circuit(self.context_params[neg_indices[i][0]]).reshape(
                self.dm_shape
            )

            tc_mul = torch.mm(A_target, A_context)
            tc_traces = torch.trace(tc_mul)
            tc_log_sigmoid = F.logsigmoid(tc_traces.double())

            neg_mul = torch.mm(A_target, A_neg)
            neg_trace = torch.trace(neg_mul)
            neg_log_sigmoid = F.logsigmoid(-neg_trace.double())

            loss += -tc_log_sigmoid - neg_log_sigmoid

        return loss / len(target_indices)

    def get_density_matrices(self, parameter):

        dms = torch.empty(
            (len(parameter), self.vocab_size, self.dm_shape[0], self.dm_shape[1])
        )

        for j, params in enumerate(parameter):
            for i, p in enumerate(params):
                dms[j, i, :, :] = self.circuit(p).reshape(self.dm_shape)

        return dms
