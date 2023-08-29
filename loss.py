from Bio.PDB import PDBParser
import os
import shutil
import torch as t
import torch.nn as nn
from scripts.utils import judge_contact, contact_mask, distance, remove_noca
from scripts.tool.to_single_chain import single_chain
from scripts.embedding import edge_init_embedding, node_init_embedding
from scripts.multihead_attention import *
from scripts.readout import NodeReadout, EdgeReadout
from arguments import buildParser
from biopandas.pdb import PandasPdb
import subprocess

current_path = os.getcwd()
workspace = os.path.dirname(os.getcwd()) + "/spaceQA/"

class EvaluateMetrics(object) : 
    def __init__(self, gt, pred) -> None:
        
        self.gt = gt
        self.pred = pred
        
        self.gt_name = gt.split('/')[-1]
        self.pred_name = pred.split('/')[-1]
        self.pred_pdb = self.pred_name.split('.')[0]
        
        args = buildParser().parse_args()
        self.clean_path = args.clean_pdb_dir
        self.job_name = args.name
                
        pass
    
    def ICS(self) : 
        '''
        F1-score : ICS = 2 * (P * R) / (P + R)
        P = ( Cg ∩ Cp ) / Cp, R = ( Cg ∩ Cp ) / Cg
        '''
        
        gt_contact = judge_contact(self.gt, intra_cutoff=1e-8, inter_cutoff=5)
        pred_contact = judge_contact(self.pred, intra_cutoff=1e-8, inter_cutoff=5)
        
        common_contact = t.mul(gt_contact, pred_contact)
        
        gt_contact_num = t.sum(gt_contact)
        pred_contact_num = t.sum(pred_contact)
        common_contact_num = t.sum(common_contact)
        
        # print(gt_contact_num, pred_contact_num, common_contact_num)
        
        if gt_contact_num == 0 : 
            return 0.5
        
        elif gt_contact_num != 0 and pred_contact_num == 0 : 
            return 0
        
        else : 
            if common_contact_num == 0 : 
                return 0
            
            P = common_contact_num / pred_contact_num
            R = common_contact_num / gt_contact_num
            
            ics = 2 * P * R / (P + R)
            return ics.item()
    
    def IPS(self) : 
        '''
        Jaccard coef : IPS = |Ig and Ip| / |Ig || Ip|
                       Ig : interface residues in native, Ip : interface residues in predicted model
        '''
        
        gt_interface = contact_mask(self.gt)
        pred_interface = contact_mask(self.pred)
        
        if gt_interface.sum() == 0 and pred_interface.sum() == 0 : 
            return 0.5
        
        else : 
            common_num = t.mul(gt_interface, pred_interface).sum()
            u = gt_interface + pred_interface
            u[u != 0] = 1
            union_num = t.sum(u)
            return (common_num / union_num).item()
    
    def cal_lddt(self, target_folder='single_chain_pdb') : 
        
        if not os.path.exists(workspace + self.job_name + '/' + target_folder) : 
            os.mkdir(workspace + self.job_name + '/' + target_folder)
        
        os.chdir(workspace + self.job_name)
        single_chain(self.gt, _type='true_')
        single_chain(self.pred)
        
        run_command = "bash " + current_path + "/scripts/tool/lDDT.sh pred_" + self.pred_name + " true_" + self.gt_name
        os.system(run_command)
        
        f=open('pred_' + self.pred_pdb + '_score.txt')
        score=[]
        for line in f:
            score.append(float(line.strip()))
        
        os.remove('pred_' + self.pred_pdb + '_score.txt')    
        shutil.rmtree(target_folder)
        os.chdir(current_path)
        
        return t.tensor(score).to(t.float32)
    
    def deviation_map(self) : 
        
        gt_dm = distance(self.gt, atom_type='CA')
        pred_dm = distance(self.pred, atom_type='CA')
        
        return abs(gt_dm - pred_dm).to(t.float32)
    
    def meanDockQ(self) : 
        
        C = PandasPdb().read_pdb(self.pred).df['ATOM']['chain_id']
        chains = list(set(C))
        if len(chains) == 1 : 
            lddt_cmd = 'lddt ' + self.pred + ' ' + self.gt + ' > ' + workspace + self.job_name + '/' + self.pred_pdb + '_1lddt.txt'
            get_cmd = "awk '$1==\"Global\"{print $4}' "  + workspace + self.job_name + '/' + self.pred_pdb + "_1lddt.txt"
            rm_cmd = 'rm ' + workspace + self.job_name + '/' + self.pred_pdb + "_1lddt.txt"
            os.system(lddt_cmd)
            global_lddt = subprocess.getoutput(get_cmd)
            os.system(rm_cmd)
            return t.tensor(float(global_lddt)).to(t.float32)
            
        
        os.chdir(workspace + self.job_name)
        if os.path.exists(self.pred_pdb + '_value.txt') : 
            os.remove(self.pred_pdb + '_value.txt')
            
        run_cmd = "bash " + current_path + "/scripts/tool/DockQ.sh " + self.gt + " " + self.pred
        os.system(run_cmd)
        
        f=open(self.pred_pdb + '_value.txt')
        dq=[]
        for line in f:
            dq.append(float(line.strip()))
        
        os.remove(self.pred_pdb + '_value.txt')
        dq = t.tensor(dq)
        
        os.chdir(current_path)
        
        return t.mean(dq).to(t.float32)
    
    def metric(self) : 
        
        m = {}
        m['lddt'] = self.cal_lddt()
        m['interface score'] = t.tensor([self.ICS(), self.IPS()]).to(t.float32)
        m['mean DockQ'] = self.meanDockQ()
        m['deviation map'] = self.deviation_map()
        
        return m
            
# label = EvaluateMetrics('/extendplus/wander_W/QA/clean_pdb/true/1EKU_clean.pdb', '/extendplus/wander_W/QA/clean_pdb/pred/1EKU_clean.pdb').metric()
# print(label)

# print(cgat_loss(n, e, label))
