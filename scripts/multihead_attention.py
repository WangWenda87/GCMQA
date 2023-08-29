import numpy as np
import dgl
import torch.nn as nn
import torch as t
from scripts.embedding import *
from scripts.utils import distance, chain_info, inter_chains_mask, judge_contact, contact_mask


class node_bias_encoding(nn.Module) :
    def __init__(self, bias_h_size, _type='contact') -> None:
        super(node_bias_encoding, self).__init__()
        
        self.hidden_linear = nn.Linear(2, bias_h_size)
        self.self_linear = nn.Linear(bias_h_size, 1)
        self._type = _type
        
    def forward(self, int_chain_label, dist_map, contact_map) : 
        int_chain_label = int_chain_label.to(t.float32)
        if self._type == 'distance' : 
            map = dist_map.to(t.float32)
        else : 
            map = contact_map.to(t.float32)
            
        x = t.stack((map, int_chain_label), dim=-1)
        x = self.hidden_linear(x)
        x = self.self_linear(x).squeeze(-1)
        x = x.unsqueeze(0).unsqueeze(0)
        return x
    
# temp = node_bias_encoding(64)
# print(temp('2BNQ_tidy.pdb').size())

class edge_bias_encoding(nn.Module) : 
    def __init__(self, h, edge_dict_dim, knn=20) -> None :
        super(edge_bias_encoding, self).__init__()
        
        self.knn = knn
        
        self.edge_embedding = nn.Embedding(edge_dict_dim, h)
        self.mask_embedding = nn.Embedding(2, h)
        
    def forward(self, c_mask, degrees_vec) : 
        
        c_mask = c_mask.repeat_interleave(self.knn).unsqueeze(-1).to(t.int32)
        degrees_vec = degrees_vec.repeat_interleave(self.knn).unsqueeze(-1).to(t.int32)
        d = self.edge_embedding(degrees_vec).transpose(0, 1)
        c = self.mask_embedding(c_mask).transpose(0, 1).transpose(-2, -1)
        
        x = d.matmul(c).unsqueeze(0)
        

        return x
    
# temp = edge_bias_encoding('2BNQ_tidy.pdb', 256, 64)
# print(temp('2BNQ.dgl').size())

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x

class NodeMultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, bias_h_size, attention_dropout_rate, num_heads):
        super(NodeMultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)
        
        self.attn_node_dis_bias = node_bias_encoding(bias_h_size, _type='distance')
        self.attn_node_con_bias = node_bias_encoding(bias_h_size, _type='contact')

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, int_chain_label, dist_map, contact_map, q, k, v):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = orig_q_size[0]
        
        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = t.matmul(q, k)  # [b, h, q_len, k_len]
        
        self.distance_encoding = self.attn_node_dis_bias(int_chain_label, dist_map, contact_map)
        self.contact_encoding = self.attn_node_con_bias(int_chain_label, dist_map, contact_map)
        attn_bias = self.distance_encoding + self.contact_encoding
        
        x = x + attn_bias
        x = t.softmax(x, dim=3)
        #x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x
    
class NodeEncoderLayer(nn.Module):
    def __init__(self, hidden_size, bias_h_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(NodeEncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = NodeMultiHeadAttention(hidden_size, bias_h_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, int_chain_label, dist_map, contact_map):
        y = self.self_attention_norm(x)
        y = self.self_attention(int_chain_label, dist_map, contact_map, y, y, y)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x

class EdgeMultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, edge_bias_dict_dim, attention_dropout_rate, num_heads):
        super(EdgeMultiHeadAttention, self).__init__()

        self.hidden_size = hidden_size
        self.edge_b_d_size = edge_bias_dict_dim
        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, c_mask, degrees_vec, q, k, v):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)
        

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = t.matmul(q, k)  # [b, h, q_len, k_len]
        self.attn_edge_bias = edge_bias_encoding(self.hidden_size, self.edge_b_d_size)
        attn_bias = self.attn_edge_bias(c_mask, degrees_vec)

        x = x + attn_bias

        x = t.softmax(x, dim=3)
        #x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x
    
class EdgeEncoderLayer(nn.Module):
    def __init__(self, hidden_size, edge_bias_dict_dim, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(EdgeEncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = EdgeMultiHeadAttention(hidden_size, edge_bias_dict_dim, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, c_mask, degrees_vec):
        y = self.self_attention_norm(x)
        y = self.self_attention(c_mask, degrees_vec, y, y, y)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x
    
