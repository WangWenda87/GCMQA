import torch as t
import torch.nn as nn
from scripts.utils import distance, chain_info, inter_chains_mask, judge_contact, contact_mask


class NodeReadout(nn.Module) : 
    def __init__(self, hidden_size, lddt_mlp_dim1, lddt_mlp_dim2, score_mlp_dim1, score_mlp_dim2) -> None:
        super(NodeReadout, self).__init__()
        
        self.lddt_mlp_layer1 = nn.Linear(hidden_size, lddt_mlp_dim1)
        self.lddt_mlp_layer2 = nn.Linear(lddt_mlp_dim1, lddt_mlp_dim2)
        self.lddt_mlp_layer3 = nn.Linear(lddt_mlp_dim2, 1)
        
        self.contact_embedding = nn.Embedding(2, hidden_size)
        self.score_mlp_layer1 = nn.Linear(hidden_size, score_mlp_dim1)
        self.score_mlp_layer2 = nn.Linear(score_mlp_dim1, score_mlp_dim2)
        self.score_mlp_layer3 = nn.Linear(score_mlp_dim2, 2)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, c) : 
        
        output = {}
        
        y = self.lddt_mlp_layer1(x)
        y = self.lddt_mlp_layer2(y)
        y = self.lddt_mlp_layer3(y)
        
        # self.BN = nn.BatchNorm1d(x.size(1))
        # x = self.BN(x)
        
        y = y.squeeze(-1)
        
        # x = t.mean(x, dim=1)
        y = self.sigmoid(y)
        output['plddt'] = y
        
        z = self.contact_embedding(c.to(t.int32))
        z = x.mul(z)
        z = t.mean(z, dim=1)
        
        z = self.score_mlp_layer1(z)
        z = self.score_mlp_layer2(z)
        z = self.score_mlp_layer3(z)
        
        z = self.sigmoid(z)
        
        output['score'] = z
        
        return output

    
# from multihead_attention import u1, u2

# temp = NodeReadout(512, 256, 128, 256, 128)
# c = contact_mask('2BNQ_tidy.pdb').squeeze(0)
# x = temp(u1, c)

# print(x)

class EdgeReadout(nn.Module) : 
    def __init__(self, hidden_size, edge_o_h_size, knn=20) -> None:
        super(EdgeReadout, self).__init__()
        
        self.knn = knn
        
        
        self.edge_linear = nn.Linear(hidden_size, edge_o_h_size)
        self.int_embedding = nn.Embedding(2, edge_o_h_size)
        self.relu = nn.ReLU()
        
    def forward(self, x, y) : 
        
        n_list = t.split(x, self.knn, dim=1)
        edge_n = t.empty(x.size(0), len(n_list), x.size(-1))
        
        for n in range(len(n_list)) : 
            node_rep = t.sum(n_list[n], dim=1)
            edge_n[:, n, :] = node_rep
        
        x = self.edge_linear(edge_n)
        y = self.int_embedding(y.to(t.int32))
        y = t.sum(y, dim=2).transpose(2, 1)
        x = t.matmul(x, y)
        
        x = self.relu(x).squeeze(0)

        return x

# y = judge_contact('2BNQ_tidy.pdb').unsqueeze(0)
# temp = EdgeReadout(512, 512)
# print(temp(u2, y).size())