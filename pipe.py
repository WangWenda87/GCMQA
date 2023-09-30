from Bio.PDB import PDBParser
import os
import shutil
import torch as t
import torch.nn as nn
from scripts.utils import judge_contact, contact_mask, distance, inter_chains_mask
from scripts.tool.to_single_chain import single_chain
from scripts.embedding import edge_init_embedding, node_init_embedding
from scripts.multihead_attention import *
from scripts.readout import NodeReadout, EdgeReadout
from scripts.build_graph import graph
from loss import EvaluateMetrics
from model import GCloss

_name = '4YZE_clean'

pdb = '/home/u2021103648/workspace/dataQA/clean_pdb/pred/' + _name + '.pdb'
target = '/home/u2021103648/workspace/dataQA/clean_pdb/true/' + _name + '.pdb'
pdb_graph = '/home/u2021103648/workspace/dataQA/dgl/' + _name + '.dgl'

graph(pdb, _name, '/home/u2021103648/workspace/dataQA/dgl')

g = dgl.load_graphs(pdb_graph)[0][0]
degrees_vec = g.out_degrees()
dist = distance(pdb)
int_chain_label = inter_chains_mask(chain_info(pdb)[0])
c_vec = contact_mask(pdb)
c_map = judge_contact(pdb)

node_embedding = node_init_embedding(1, 256, 168, 128)
edge_embedding = edge_init_embedding(1, 256, 128)
node_update = NodeEncoderLayer(256, 64, 64, 0.2, 0.1, 8)
edge_update = EdgeEncoderLayer(256, 64, 64, 0.2, 0.1, 8)
node_readout = NodeReadout(256, 256, 128, 256, 128)
edge_readout = EdgeReadout(256, 512)

node = node_embedding(pdb_graph)
edge = edge_embedding(pdb_graph)

for i in range(12) :
	node = node_update(node, int_chain_label, dist, c_map)
	edge = edge_update(edge, c_vec, degrees_vec)
scores = node_readout(node, c_vec.squeeze(0))
deviation_map = edge_readout(edge, c_map.unsqueeze(0))

# print(scores)
# print(deviation_map)

label = EvaluateMetrics(target, pdb).metric()
l = GCloss(target, scores, deviation_map, label)
print(l[0])