"""
Deep Learning on Graphs - ALTEGRAD - Dec 2019
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GNN(nn.Module):
    """Simple GNN model"""
    def __init__(self, n_feat, n_hidden_1, n_hidden_2, n_class, dropout):
        super(GNN, self).__init__()

        self.fc1 = nn.Linear(n_feat, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.fc3 = nn.Linear(n_hidden_2, n_class)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x_in, adj):
        ############## Tasks 10 and 13
        
        output = self.fc1(x_in)
        output = self.relu(torch.mm(adj,output))
        output = self.fc2(output)
        t = self.relu(torch.mm(adj,output))
        output = self.fc3(t)

        return F.log_softmax(output, dim=1),t