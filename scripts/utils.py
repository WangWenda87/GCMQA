"""
@ Description: Subfunctions used in the modelClean raw pdb
"""

import os
from copy import deepcopy
import pandas as pd
import torch as t
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from Bio.PDB import HSExposure
import dgl
import scipy.sparse as sp
import numpy as np
from biopandas.pdb import PandasPdb
from typing import List, Union
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import scripts.tool.geomeas as gm
#import tool.geomeas as gm
import warnings
warnings.filterwarnings("ignore")

#os.getcwd()

def remove_noca(pdb : str, out_pdb) : 
    
    pp = PandasPdb().read_pdb(pdb)
    ppdb = PandasPdb().read_pdb(pdb).df['ATOM']
    chain_list = list(set(ppdb['chain_id']))
    s = 0
    for c in chain_list : 
        res_num = max(ppdb['residue_number'][ppdb['chain_id'] == c])
        for i in range(res_num) :
            atom_list = ppdb['atom_name'][(ppdb['chain_id'] == c) & (ppdb['residue_number'] == i + 1)].tolist()
            if 'CA' not in atom_list : 
                ppdb.drop(ppdb[(ppdb['chain_id'] == c) & (ppdb['residue_number'] == i + 1)].index, inplace=True)
                s = s + 1
    if s != 0 :             
        pp.df['ATOM'] = ppdb
        pp.to_pdb(out_pdb)
    else : 
        pass
    
def fix_pdb(pdb : str, output : str) : 
    
    pdb_df = PandasPdb().read_pdb(pdb).df['ATOM']
    cb_df = pdb_df[((pdb_df.loc[:, 'residue_name'] == 'GLY') & (pdb_df.loc[:, 'atom_name'] == 'CA')) | (pdb_df.loc[:, 'atom_name'] == 'CB')]
    ca_df = pdb_df[pdb_df.loc[:, 'atom_name'] == 'CA']
    n_df = pdb_df[(pdb_df.loc[:, 'atom_name'] == 'N')]
    o_df = pdb_df[(pdb_df.loc[:, 'atom_name'] == 'O')]
    
    if len(cb_df) == len(ca_df) and len(ca_df) == len(n_df) and len(n_df) ==len(o_df) :
        pass
    else : 
        fix_cmd = "pdbfixer " + pdb + " --add-atoms=heavy --output=" + output
        clean_cmd = "sed -i '/^REMARK\|^CONECT/d' " + output
        os.system(fix_cmd)
        os.system(clean_cmd)

# basic property
# pdb structue --> sequence fasta
def pdb2fasta(pdb : str) -> str:

    ppdb = PandasPdb().read_pdb(pdb)
    pdb_df = ppdb.df['ATOM']
    residue_name = pdb_df[pdb_df.loc[:, 'atom_name'] == 'CA']['residue_name']

    aa = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N',
             'ASP': 'D', 'CYS': 'C', 'GLU': 'E',
             'GLN': 'Q', 'GLY': 'G', 'HIS': 'H',
             'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
             'MET': 'M', 'PHE': 'F', 'PRO': 'P',
             'SER': 'S', 'THR': 'T', 'TRP': 'W',
             'TYR': 'Y', 'VAL': 'V'}
    if not os.path.isfile(pdb):
        raise FileExistsError(f'PDB File does not exist {pdb}')

    seq = []
    for res in residue_name : 
        seq.append(aa[res])

    return "".join(seq)

#sequence --> one-hot
def residue_type(seque : str) -> t.Tensor:

    aa_dict =  {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
                'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
                'M': 10, 'N': 11, 'P': 12, 'Q': 13,
                'R': 14, 'S': 15, 'T': 16, 'V': 17,
                'W': 18, 'Y': 19, 'X': 20}

    seq = seque
    length = len(seq)
    aa_type = np.zeros([length])

    for idx, item in enumerate(seq.upper()):
        if item not in aa_dict.keys():
            item = 'X'
        col_idx = aa_dict[item]
        aa_type[idx] = col_idx

    return t.from_numpy(aa_type).reshape(-1, 1)


# pdb --> dssp_features (ss, RASA, phi, psi)
def cal_dssp(pdb : str) -> pd.DataFrame :
    pdb_name = pdb.split('/')[-1].split('.')[0]
    p = PDBParser()

    structure = p.get_structure(pdb_name, pdb)
    model = structure[0]
    dssp = DSSP(model, pdb)
    key_list = list(dssp.keys())

    ss_list = []
    rasa_list = []
    phi_list = []
    psi_list = []

    for key in key_list:
        ss, rasa, phi, psi = dssp[key][2:6]
        ss_list.append(ss)
        rasa_list.append(rasa)
        phi_list.append(phi)
        psi_list.append(psi)

    dssp_feature_df = pd.DataFrame(list(zip(ss_list, rasa_list, phi_list, psi_list)),
                              columns=['ss', 'rasa', 'phi', 'psi'])
    
    return dssp_feature_df

# dssp_feature --> second structure
def second_structure(df : pd.DataFrame) -> t.Tensor : 
    tokens_dict = {'H': 0, 'B': 1, 'E': 2, 'G': 3, 'I': 4, 'T': 5, 'S': 6, '-': 7}
    ss = np.zeros([df.shape[0]])
    ss_list = df.ss.to_list()

    for idx, item in enumerate(ss_list):
        if item not in tokens_dict:
            raise KeyError(f'This {item} is not secondary structure type.')
        col_idx = tokens_dict[item]
        ss[idx] = col_idx

    return t.from_numpy(ss).reshape(-1, 1)

# print(second_structure(cal_dssp('2BNQ_tidy.pdb')))

# dssp_feature --> RASA
def feat_rasa(df : pd.DataFrame) -> t.Tensor : 
    return t.tensor(df['rasa']).reshape(-1, 1)

#dssp_feature --> phi, psi
max_abs_scaler = MaxAbsScaler()

def feat_angles(df : pd.DataFrame, angle_type = 'phi') -> t.Tensor : 
    angle = max_abs_scaler.fit_transform(t.tensor(df[angle_type].values).reshape(-1, 1))
    return t.tensor(angle)


# pdb --> HSE
min_max_scaler = MinMaxScaler()

def feat_hse(pdb : str) -> t.Tensor : 
    pdb_name = pdb.split('/')[-1].split('.')[0]
    p = PDBParser()
    structure = p.get_structure(pdb_name, pdb)
    model = structure[0]
    hse = HSExposure.HSExposureCA

    exp_ca=hse(model)
    #print(exp_ca.keys())
    key_list = list(exp_ca.keys())

    au_list = []
    ad_list = []

    for key in key_list:
        rseau, rsead = exp_ca[key][0:2]
        au_list.append(rseau)
        ad_list.append(rsead)
    
    HSEau = min_max_scaler.fit_transform(t.tensor(np.asarray(au_list)).reshape(-1, 1))
    HSEad = min_max_scaler.fit_transform(t.tensor(np.asarray(ad_list)).reshape(-1, 1))
    return t.tensor([HSEau, HSEad]).reshape(-1, 2)

# Calculate CA-CA, or CB-CB or N-O distance for a pdb file
def distance(pdb_file: str, atom_type='CB') -> t.Tensor :
    
    ppdb = PandasPdb().read_pdb(pdb_file)
    pdb_df = ppdb.df['ATOM']

    if atom_type == 'CB':
        # GLY does not have CB, use CA to instead of.
        filtered_df = pdb_df[((pdb_df.loc[:, 'residue_name'] == 'GLY') & (pdb_df.loc[:, 'atom_name'] == 'CA')) | (pdb_df.loc[:, 'atom_name'] == 'CB')]
    elif atom_type == 'CA':
        filtered_df = pdb_df[pdb_df.loc[:, 'atom_name'] == 'CA']
    elif atom_type == 'NO':
        filtered_df = pdb_df[(pdb_df.loc[:, 'atom_name'] == 'N') | (pdb_df.loc[:, 'atom_name'] == 'O')]
    else:
        raise ValueError('Atom type should be CA, CB or NO.')

    if atom_type != 'NO':
        coord = filtered_df.loc[:, ['x_coord', 'y_coord', 'z_coord']].values.tolist()
        real_dist = euclidean_distances(coord)
    else:
        coord_N = filtered_df[filtered_df.loc[:, 'atom_name'] == 'N'].loc[:,
                  ['x_coord', 'y_coord', 'z_coord']].values.tolist()
        coord_O = filtered_df[filtered_df.loc[:, 'atom_name'] == 'O'].loc[:,
                  ['x_coord', 'y_coord', 'z_coord']].values.tolist()

        real_dist = euclidean_distances(coord_N, coord_O)  # up-triangle N-O, low-triangle O-N

    return t.tensor(real_dist)

# print(distance('2BNQ_tidy.pdb', 'NO'))
#get chain id
def get_chain(pdb : str) -> t.Tensor : 
    ppdb = PandasPdb().read_pdb(pdb)
    pdb_df = ppdb.df['ATOM']
    ca_df = pdb_df[pdb_df.loc[:, 'atom_name'] == 'CA']

    label_encoder = LabelEncoder()
    encoded_list = label_encoder.fit_transform(ca_df['chain_id'])
    chain_id = min_max_scaler.fit_transform(t.tensor(np.asarray(encoded_list)).reshape(-1, 1))
    return t.tensor(chain_id)

# contact property
# extract contact counts

def chain_info(pdb : str) :
    
    ppdb = PandasPdb().read_pdb(pdb)
    pdb_df = ppdb.df['ATOM']
    ca_df = pdb_df[pdb_df.loc[:, 'atom_name'] == 'CA']
    length_dict = ca_df['chain_id'].value_counts(sort = False).values
    all_length = len(ca_df)
    
    return length_dict, all_length

def inter_chains_mask(length):
    dim = sum(length)
    refer = [0]
    for l in range(len(length)):
        re = sum(length[:(l + 1)])
        refer.append(re)
    chain_mask = np.zeros(shape = (dim, dim))
    for i in range(dim):
        for j in range(dim):
            for r in range(len(refer) - 1):
                sub_re = [refer[r], refer[r + 1]]
                if (i >= sub_re[0] and j >= sub_re[0] and i < sub_re[1] and j < sub_re[1]):
                    chain_mask[i, j] = 1
    chain_mask = t.from_numpy(chain_mask)
    return chain_mask


def interface(d, cutoff):
    k = np.empty(len(d))
    k[d < cutoff] = 1
    k[d > cutoff] = 0
    return k

def judge_contact(pdb_path : str, intra_cutoff = 6, inter_cutoff = 8) -> t.Tensor:
    d_map = distance(pdb_path, atom_type='CA')
    c_length = chain_info(pdb_path)
    c_mask = inter_chains_mask(c_length[0])
    dim = d_map.shape[0]
    c = np.empty(shape = (dim, dim))

    c[c_mask == 1] = interface(d_map[c_mask == 1], cutoff=intra_cutoff)
    c[c_mask == 0] = interface(d_map[(c_mask == 0)], cutoff=inter_cutoff)
    c = t.from_numpy(c)
    c = c - t.eye(dim)
    return c

def contact_mask(pdb_path : str) -> t.Tensor : 
    contact_map = judge_contact(pdb_path)
    c_length = chain_info(pdb_path)
    c_mask = 1 - inter_chains_mask(c_length[0])
    c_map = t.mul(contact_map, c_mask)
    x = t.sum(c_map, dim=0)
    x[x != 0] = 1
    return x
    
# temp = contact_mask('2BNQ_tidy.pdb')
# print(temp)

# intra-chain and inter-chain contact counts
def cal_counts(pdb : str, intra_counts = True) -> t.Tensor :
    contact_map = judge_contact(pdb)
    chain_mask = inter_chains_mask(chain_info(pdb)[0])
    if intra_counts == True : 
        mask = chain_mask
    else : 
        mask = t.mul(chain_mask, -1).add(1)
    masked_map = t.mul(contact_map, mask)
    counts = t.sum(masked_map, 1)
    return counts.reshape(-1, 1)

# contact chains number of one residue
def cal_contact_chains(pdb : str) -> t.Tensor : 
    ppdb = PandasPdb().read_pdb(pdb)
    pdb_df = ppdb.df['ATOM']
    ca_df = pdb_df[pdb_df.loc[:, 'atom_name'] == 'CA']
    chains_count = np.zeros(len(ca_df))
    

    contact_map = judge_contact(pdb)
    chain_mask = t.mul(inter_chains_mask(chain_info(pdb)[0]), -1).add(1)
    inter_con_map = t.mul(contact_map, chain_mask)

    contact_sites = t.where(inter_con_map == 1)
    site_info = []
    for i in range(len(contact_sites[0])) :
        s1 = contact_sites[0][i].item()
        s2 = contact_sites[1][i].item()
        chain1 = ca_df['chain_id'].iloc[s1]
        chain2 = ca_df['chain_id'].iloc[s2]
        info = chain1 + '-' + chain2
        site_info.append([s1, info])
    
    dic = list(set([tuple(t) for t in site_info]))
    dic = sorted(dic,key=lambda t:t[0])
    for j in dic : 
        chains_count[j[0]] = chains_count[j[0]] + 1
    return t.tensor(chains_count).reshape(-1, 1)

# structure position
def cal_rep_res(pdb : str) : 
    ppdb = PandasPdb().read_pdb(pdb)
    pdb_df = ppdb.df['ATOM']
    filtered_df = pdb_df[((pdb_df.loc[:, 'residue_name'] == 'GLY') & (pdb_df.loc[:, 'atom_name'] == 'CA')) | (pdb_df.loc[:, 'atom_name'] == 'CB')]
    coord = np.array(filtered_df.loc[:, ['x_coord', 'y_coord', 'z_coord']].values.tolist())
    centroid = np.mean(coord, 0)
    distance_vector = np.linalg.norm(coord - centroid, ord=None, axis=1)
    f_residue = coord[np.where(distance_vector == max(distance_vector)), :]
    n_residue = coord[np.where(distance_vector == min(distance_vector)), :]
    return centroid.ravel(), f_residue.ravel(), n_residue.ravel()

def coordinate_transform(pdb : str) -> t.Tensor : 
    O = cal_rep_res(pdb)[0]
    O_x = cal_rep_res(pdb)[1]
    O_y = cal_rep_res(pdb)[2]
    rotation_matrix = gm.Pose().calPoseFrom3Points(O, O_x, O_y)[0]
    translation_matrix = gm.Pose().calPoseFrom3Points(O, O_x, O_y)[1]
    r_inv = np.linalg.inv(rotation_matrix)
    
    ppdb = PandasPdb().read_pdb(pdb)
    pdb_df = ppdb.df['ATOM']
    filtered_df = pdb_df[((pdb_df.loc[:, 'residue_name'] == 'GLY') & (pdb_df.loc[:, 'atom_name'] == 'CA')) | (pdb_df.loc[:, 'atom_name'] == 'CB')]
    coord = np.array(filtered_df.loc[:, ['x_coord', 'y_coord', 'z_coord']].values.tolist())
    
    transform_matrix = np.dot(r_inv, (coord - translation_matrix.reshape(1, -1)).T)
    return t.round(t.tensor(transform_matrix.T), decimals=3)

#build KNN graph
def pdb2graph(pdb_file: str, knn=20):
    atom_df = PandasPdb().read_pdb(pdb_file).df['ATOM']
    atom_df_full = deepcopy(atom_df)  
    atom_df = atom_df[atom_df.loc[:, 'atom_name'] == 'CA']
    node_coords = t.tensor(atom_df[['x_coord', 'y_coord', 'z_coord']].values, dtype=t.float32)
    protein_graph = dgl.knn_graph(node_coords, knn, exclude_self=True)
    protein_graph = protein_graph.remove_self_loop()  # remove self loop
    srcs = protein_graph.edges()[0]
    dsts = protein_graph.edges()[1]

    edges = list(zip(srcs, dsts))

    # CA-CA distance
    atom_df_ca = atom_df[atom_df.loc[:, 'atom_name'] == 'CA']
    chain_id_list = atom_df_ca.loc[:, 'chain_id'].tolist()
    chain_id_dict = dict(zip([i for i in range(len(chain_id_list))], chain_id_list))  

    same_chain_feature = []
    for i in edges:
        u, v = i
        u = u.item()
        v = v.item()
        if (chain_id_dict[u] == chain_id_list[v]):
            same_chain_feature.append(0)
        else:
            same_chain_feature.append(1)
    return atom_df_full, protein_graph, edges, t.tensor(same_chain_feature).reshape(-1, 1)

# edge distance features
def edge_dist(pdb : str, edges : List, atom_type='CB') -> t.Tensor : 
    distance_map = distance(pdb, atom_type)
    edge_feat = []

    for i in edges :
        if atom_type != 'NO' :
            edge_feat.append(distance_map[i].item())
        else :
            u = i[0]
            v = i[1]
            edge_feat.append((distance_map[u, v].item() + distance_map[v, u].item()) / 2)

    return t.tensor(edge_feat).reshape(-1, 1)

#edge contact feature
def edge_contact(pdb : str, edges : List, chain_info : t.Tensor) -> t.Tensor : 
    CB_dist = edge_dist(pdb, edges)
    contact_list = np.zeros(len(CB_dist))

    for i in range(len(chain_info)) :
        if chain_info[i] == 0 :
            if CB_dist[i] <= 6 : 
                contact_list[i] = 1
        else :
            if CB_dist[i] <= 8 : 
                contact_list[i] = 1
    
    return t.tensor(contact_list).reshape(-1, 1)

# add features
def update_node_feature(graph: dgl.DGLGraph, new_node_features: List) -> None:

    for node_feature in new_node_features:
        if not graph.ndata:
            graph.ndata['f'] = node_feature
        else:
            graph.ndata['f'] = t.cat((graph.ndata['f'], node_feature), dim=1)
    return None


def update_edge_feature(graph: dgl.DGLGraph, new_edge_features: List) -> None:

    for edge_feature in new_edge_features:
        if not graph.edata:
            graph.edata['f'] = edge_feature
        else:
            graph.edata['f'] = t.cat((graph.edata['f'], edge_feature), dim=1)
    return None

