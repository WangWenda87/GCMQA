import os
import torch as t
import torch.nn as nn
import torch.optim as optim
from loss import EvaluateMetrics
from scripts.clean_pipe import *
from scripts.build_graph import graph
from scripts.embedding import node_init_embedding, edge_init_embedding
from scripts.multihead_attention import *
from scripts.readout import NodeReadout, EdgeReadout
from scripts.utils import contact_mask, judge_contact

# class GCloss(nn.Module) : 
    
#     def __init__(self) -> None:
#         super().__init__()
        
#     def forward(n, e, tr) : 
        
#         l1 = nn.MSELoss(n['plddt'], tr['lddt'])
#         l2 = nn.MSELoss(n['score'][0][0], t.mean(tr['interface score']))
#         l3 = nn.MSELoss(n['score'][0][1], tr['mean DockQ'])
        
#         sum_mse = nn.MSELoss(size_average = False)
#         dim = tr['deviation map'].size(0)
#         N = dim * (dim - 1) / 2

#         l4 = sum_mse(t.triu(e), t.triu(tr['deviation map'])) / N

#         l = (l1 + l2 + l3 + l4) / 4
#         return l, [l1, l2, l3, l4]

def GCloss(n, e, tr, mse = nn.MSELoss(), smooth = nn.SmoothL1Loss(), eps=1e-5) :
    
    
    l1 = smooth(n['plddt'], tr['lddt']) * 100 + eps
    l2 = smooth(n['score'][0][0], t.mean(tr['interface score'])) * 100 + eps
    l3 = smooth(n['score'][0][1], tr['mean DockQ']) * 100 + eps
    
    sum_smooth = nn.SmoothL1Loss(size_average = False)
    dim = tr['deviation map'].size(0)
    N = dim * (dim - 1) / 2

    l4 = sum_smooth(t.triu(e), t.triu(tr['deviation map'])) / N

    if l4 > 50 : 
        l4 = t.tensor(50)

    l = (l1 + l2 + l3) / 3

    return l, [l1, l2, l3, l4]
    
class GCQA(nn.Module) : 
    
    def __init__(self, **kwargs) -> None:
        super(GCQA, self).__init__()
        
        self.build(**kwargs)
        
        #self.criterion      = GCloss()
        self.inputs         = None
        self.targets        = None
        self.outputs        = None
        self.loss           = 0
        self.accuracy       = 0
        self.optimizer      = None
        lr                  = kwargs.get('lr', 0.001)
        self.optimizer      = optim.SGD(self.parameters(), lr, momentum=0.9, weight_decay=0)
        
    def build(self, **kwargs) :
        
        self.input_list           = kwargs.get('input_list')
        self.dgl_folder           = kwargs.get('dgl_folder')
        self.n_layers             = kwargs.get('n_layers')
        self.batch_size           = kwargs.get('batch_size')
        self.hidden_size          = kwargs.get('h_size')
        self.node_basic_size      = kwargs.get('n_b_size')
        self.node_contact_size    = kwargs.get('n_c_size')
        self.edge_contact_size    = kwargs.get('e_c_size')
        self.node_bias_size       = kwargs.get('n_bias_size')
        self.edge_bias_size       = kwargs.get('e_bias_size')
        self.ffn_size             = kwargs.get('ffn_size')
        self.dropout              = kwargs.get('dropout')
        self.attention_dropout    = kwargs.get('a_dropout')
        self.num_heads            = kwargs.get('num_heads')
        self.plddt_out1           = kwargs.get('l_mlp1')
        self.plddt_out2           = kwargs.get('l_mlp2')
        self.score_out1           = kwargs.get('s_mlp1')
        self.score_out2           = kwargs.get('s_mlp2')
        self.edge_out_hidden_size = kwargs.get('e_out_hidden_size')
        
        self.node_embedding       = node_init_embedding(self.batch_size, self.hidden_size, self.node_basic_size, self.node_contact_size)
        self.edge_embedding       = edge_init_embedding(self.batch_size, self.hidden_size, self.edge_contact_size)
        
        self.node_mha_update      = NodeEncoderLayer(self.hidden_size, self.node_bias_size, self.ffn_size, self.dropout, self.attention_dropout, self.num_heads)
        self.edge_mha_update      = EdgeEncoderLayer(self.hidden_size, self.edge_bias_size, self.ffn_size, self.dropout, self.attention_dropout, self.num_heads)
        
        self.node_output          = NodeReadout(self.hidden_size, self.plddt_out1, self.plddt_out2, self.score_out1, self.score_out2)
        self.edge_output          = EdgeReadout(self.hidden_size, self.edge_out_hidden_size)
        
    def forward(self, inputs) : 

        pdb_name = inputs.split('/')[-1].split('.')[0]
        graph(inputs, pdb_name, self.dgl_folder)
      
        pdb_graph = self.dgl_folder + pdb_name + '.dgl'
        
        g = dgl.load_graphs(pdb_graph)[0][0]
        degrees_vec = g.out_degrees()
        dist = distance(inputs)
        int_chain_label = inter_chains_mask(chain_info(inputs)[0])
        c_vec = contact_mask(inputs)
        c_map = judge_contact(inputs)
        
        node = self.node_embedding(pdb_graph)
        edge = self.edge_embedding(pdb_graph)
        
        for i in range(self.n_layers) : 
            node = self.node_mha_update(node, int_chain_label, dist, c_map)
            edge = self.edge_mha_update(edge, c_vec, degrees_vec)
            
        scores = self.node_output(node, c_vec.squeeze(0))
        deviation_map = self.edge_output(edge, c_map.unsqueeze(0))
        
        return scores, deviation_map
    
    def fit(self, scores, deviation_map, _label, pred=False) : 
        
        self.loss = GCloss(scores, deviation_map, _label)[0]
        #print(self.loss)
        self.sub_loss = GCloss(scores, deviation_map, _label)[1]
        
        if not pred:
            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()
        
        
# _model = GCQA()
# scores, dev = _model('/extendplus/wander_W/QA/clean_pdb/pred/2WD5_6A_52_clean.pdb')
