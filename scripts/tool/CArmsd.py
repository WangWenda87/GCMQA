from Bio.PDB import PDBParser
from Bio.SVDSuperimposer import SVDSuperimposer
from biopandas.pdb import PandasPdb
import pandas as pd
import csv

class ca_rmsd(object) : 
    def __init__(self, gt, pred) -> None:
        
        self.gt = gt
        self.pred = pred
        self.supp = SVDSuperimposer()
        
        pass
    
    def coords(self, pdb) : 
        
        ppdb = PandasPdb().read_pdb(pdb).df['ATOM']
        ca_df = ppdb[ppdb.loc[:, 'atom_name'] == 'CA'].loc[:, ['x_coord', 'y_coord', 'z_coord']].values
        
        return ca_df
    
    def rmsd(self) : 
        
        ca_gt = self.coords(self.gt)
        ca_pred = self.coords(self.pred)
        self.supp.set(ca_gt, ca_pred)
        self.supp.run()
        rms = self.supp.get_rms()
        
        return '{:.3f}'.format(rms)
    
pdb_path = '/home/u2021103648/workspace/dataQA/clean_pdb/'
gt_list = '/home/u2021103648/workspace/dataQA/gt_950.csv'
pred_list = '/home/u2021103648/workspace/dataQA/pred_950.csv'


with open(gt_list) as csvfile:
    csv_reader = csv.reader(csvfile)
    gt = []
    for row in csv_reader:
        gt.append(row[0].split('/')[-1].split('.')[0])       
with open(pred_list) as csvfile:
    csv_reader = csv.reader(csvfile)
    pred = []
    for row in csv_reader:
        pred.append(row[0].split('/')[-1].split('.')[0])

RMSD = []
for i in range(len(pred)) : 
    g = pdb_path + 'true/' + gt[i] + '_clean.pdb'
    p = pdb_path + 'pred/' + pred[i]+ '_clean.pdb'
    rms = ca_rmsd(g, p).rmsd()
    RMSD.append(rms)
    
data = pd.DataFrame(data = RMSD,index = None,columns = 'RMSD')
data.to_csv('/home/u2021103648/workspace/dataQA/rmsd_ca.csv')
    

#temp = ca_rmsd('/home/u2021103648/workspace/dataQA/clean_pdb/true/3QB4_4A_16_clean.pdb', '/home/u2021103648/workspace/dataQA/clean_pdb/pred/3QB4_4A_16_clean.pdb')
# print(temp.rmsd())