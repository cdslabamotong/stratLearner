import torch.nn as nn
import torch
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nclass, dropout):
        super(GCN, self).__init__()
        
        
        self.gc1 = GraphConvolution(1, 1)
        self.gc2 = GraphConvolution(1, 1)
        self.dropout = dropout
        self.MLP = nn.Sequential(
            nn.Linear(nclass,  512), 
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.ReLU(),
            
            nn.Linear(512, 256),
            nn.ReLU(),
  
            nn.Linear(256, nclass),
        )

        
    def forward(self, x, adj):
        x=x.sum(1)
        x=x.unsqueeze(1)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x=x.squeeze()
        
        x=self.MLP(x)
        return x

