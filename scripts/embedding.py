import numpy as np
import torch as t
import torch.nn as nn
import dgl

class node_init_embedding(nn.Module) : 
    def __init__(self, batch_size, hidden_size, node_basic_size, node_contact_size) -> None:
        super(node_init_embedding, self).__init__()
        
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.node_basic_size = node_basic_size
        self.node_contact_size = node_contact_size
        
        self.node_basic_embedding = nn.Embedding(self.node_basic_size, self.hidden_size)
        self.node_contact_embedding = nn.Embedding(self.node_contact_size, self.hidden_size)
        
        self.node_basic_linear = nn.Linear(1, self.hidden_size, bias=False)
        self.node_contact_linear = nn.Linear(1, self.hidden_size, bias=False)
        
    def forward(self, x) : 
        graph = dgl.load_graphs(x)[0][0]
        node_feat = graph.ndata['f']
        basic_embed_feat = t.as_tensor(node_feat[:, :2], dtype=t.int64)
        basic_linear_feat = node_feat[:, 2:10].unsqueeze(-1).to(t.float32)
        contact_linear_feat = node_feat[:, 10].unsqueeze(-1).unsqueeze(-1).to(t.float32)
        contact_embed_feat = t.as_tensor(node_feat[:, 11:], dtype=t.int64)
        
        #print(self.node_basic_embedding(basic_embed_feat).device, self.node_basic_linear(basic_linear_feat).device, self.node_contact_linear(contact_linear_feat).device, self.node_contact_embedding(contact_embed_feat).device)
        x = t.cat((self.node_basic_embedding(basic_embed_feat), self.node_basic_linear(basic_linear_feat), self.node_contact_linear(contact_linear_feat), self.node_contact_embedding(contact_embed_feat)), 1)
        x = t.sum(x, dim=1)
        x = x.view(self.batch_size, x.size(0), self.hidden_size)
        
        return x
        
class edge_init_embedding(nn.Module) : 
    def __init__(self, batch_size, hidden_size, edge_contact_dim) -> None:
        super(edge_init_embedding, self).__init__()
        
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.edge_contact_dim = edge_contact_dim
        
        self.edge_embedding = nn.Embedding(self.edge_contact_dim, self.hidden_size)
        
        self.edge_linear = nn.Linear(1, self.hidden_size)
        
    def forward(self, x) : 
        graph = dgl.load_graphs(x)[0][0] 
        edge_feat = graph.edata['f']
        linear_feat = edge_feat[:, 2:].unsqueeze(-1).to(t.float32)
        embed_feat = t.as_tensor(edge_feat[:, :2], dtype=t.int64)
        
        x = t.cat((self.edge_linear(linear_feat), self.edge_embedding(embed_feat)), 1)
        x = t.sum(x, dim=1)
        x = x.view(self.batch_size, x.size(0), self.hidden_size)
        
        return x
# temp_node = node_init_embedding(1, 512, 168, 128)
# temp_edge = edge_init_embedding(1, 512, 128)
# print(temp_node('2BNQ.dgl').size(), temp_edge('2BNQ.dgl').size())
