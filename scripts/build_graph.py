# build a KNN graph for a complex structure

import os
import dgl
from sklearn.preprocessing import MinMaxScaler, MinMaxScaler
from scripts.utils import *
#from utils import *

def graph(pdb : str, graph_name : str, save_path : str) -> None : 

    # create graph
    full_atoms, complex_graph, edges, same_chain_info = pdb2graph(pdb)

    # node features # dim(41)
    sequence_one_hot = residue_type(pdb2fasta(pdb)) #21
    #print(len(pdb2fasta(pdb)))
    dssp_features = cal_dssp(pdb)

    dssp = second_structure(dssp_features) #8
    rasa = feat_rasa(dssp_features) #1
    phi = feat_angles(dssp_features, 'phi') #1
    psi = feat_angles(dssp_features, 'psi') #1
    #hse = feat_hse(pdb) #2
    chain_id = get_chain(pdb) #1
    #print(sequence_one_hot.size(), dssp.size(), rasa.size(), phi.size(), psi.size())

    intra_contact_counts = cal_counts(pdb) #1
    inter_contact_counts = cal_counts(pdb, intra_counts=False) #1
    contact_chains_counts = cal_contact_chains(pdb) #1

    relative_position = coordinate_transform(pdb) #3

    #print(relative_position.size(), chain_id.size(), intra_contact_counts.size(), inter_contact_counts.size(), contact_chains_counts.size())

    #edge features #dim(5)
    #same_chain_info #1
    ca_distance = edge_dist(pdb, edges, 'CA') #1
    cb_distance = edge_dist(pdb, edges, 'CB') #1
    no_distance = edge_dist(pdb, edges, 'NO') #1
    edge_whether_contact = edge_contact(pdb, edges, same_chain_info) #1

    #add features to graph
    update_node_feature(complex_graph, [sequence_one_hot, dssp, rasa, phi, psi, 
                                        relative_position, chain_id, intra_contact_counts, inter_contact_counts, 
                                        contact_chains_counts])
    
    update_edge_feature(complex_graph, [same_chain_info, edge_whether_contact, cb_distance, ca_distance, no_distance])

    dgl.save_graphs(filename=os.path.join(save_path, f'{graph_name}.dgl'), g_list=complex_graph)
    print(f'{graph_name} SUCCESS')
    print(complex_graph)

#graph('/home/u2021103648/workspace/dataQA/clean_pdb/pred/2bkr_82_clean.pdb', '2bkr', './')
