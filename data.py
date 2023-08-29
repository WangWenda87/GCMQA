import os, functools, math, csv, random, pickle, json
import numpy as np

import torch as t
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from collections import defaultdict as ddict
from scripts.clean_pipe import clean_pipe
from loss import EvaluateMetrics

def dataset_split(dataset, train_ratio, val_ratio, test_ratio, random_seed=0) : 
    
    data_size = len(dataset)
    val_size = int(data_size * val_ratio)
    test_size = int(data_size * test_ratio)
    train_size = int(data_size - val_size - test_size)
    
    train_set, sub_set = random_split(dataset, lengths=[train_size, val_size + test_size], generator=t.Generator().manual_seed(random_seed))
    val_set, test_set  = random_split(sub_set, lengths=[val_size, test_size], generator=t.Generator().manual_seed(random_seed))
    
    return train_set, val_set, test_set

def get_train_val_test_loader(dataset, train_ratio, val_ratio, test_ratio, collate_fn=default_collate, batch_size=1, num_workers=2, pin_memory=False, predict=False):

    train_set, val_set, test_set = dataset_split(dataset, train_ratio, val_ratio, test_ratio)

    
    if not predict:
        train_indices   = [i for i, row in enumerate(train_set) if os.path.exists(row[0])]
        val_indices     = [i for i, row in enumerate(val_set) if os.path.exists(row[0])]
        test_indices    = [i for i, row in enumerate(test_set) if os.path.exists(row[0])]

        random.shuffle(train_indices)
        random.shuffle(val_indices)
        random.shuffle(test_indices)

        # Sample elements randomly from a given list of indices, without replacement.
        train_sampler   = SubsetRandomSampler(train_indices)
        val_sampler     = SubsetRandomSampler(val_indices)
        test_sampler    = SubsetRandomSampler(test_indices)

        train_loader    = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory)
        val_loader      = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory)
        test_loader     = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory)
        
        train_features, train_labels = next(iter(train_loader))
        
        return train_loader, val_loader, test_loader
    
    else : 
        test_indices    = [i for i, row in enumerate(test_set) if os.path.exists(row[0])]
        random.shuffle(test_indices)
        test_sampler    = SubsetRandomSampler(test_indices)
        test_loader     = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory)
        return test_loader

class ComplexDataset(Dataset) : 
    def __init__(self, gt_filename, pred_filename, list_dir, clean_pdb_dir, random_seed=123) -> None:
        
        self.list_dir = list_dir
        self.clean_pdb_dir = clean_pdb_dir
        gt_list = os.path.join(self.list_dir, gt_filename)
        pred_list = os.path.join(self.list_dir, pred_filename)
        
        assert os.path.exists(gt_list) and os.path.exists(pred_list), '{} or {} does not exist!'.format(gt_list, pred_list)
        
        with open(gt_list) as f_gt:
            reader = csv.reader(f_gt)
            self.gt_data = [row[0] for row in reader]
        random.seed(random_seed)
        random.shuffle(self.gt_data)

        with open(pred_list) as f_pred:
            reader = csv.reader(f_pred)
            self.pred_data = [row[0] for row in reader]
        random.seed(random_seed)
        random.shuffle(self.pred_data)
  
        assert len(self.gt_data) == len(self.pred_data), "the number of native and decoys in the dataset varies"
        
    def __len__(self):
        return len(self.gt_data)
        
    def __getitem__(self, idx) : 
        gt = self.gt_data[idx]
        pred = self.pred_data[idx]
        gt_name = gt.split('/')[-1].split('.')[0] + '_clean.pdb'
        pred_name = pred.split('/')[-1].split('.')[0] + '_clean.pdb'
        
        gt_full_path = self.clean_pdb_dir + 'true/' + gt_name
        pred_full_path = self.clean_pdb_dir + 'pred/' + pred_name
        
        # print(pred_full_path, gt_full_path)
        
        # if os.path.exists(gt_full_path) : 
        #     clean_pipe(pred, self.clean_pdb_dir + 'pred/')
            
        # else : 
        #     clean_pipe(gt, self.clean_pdb_dir + 'true/')
        #     clean_pipe(pred, self.clean_pdb_dir + 'pred/')
            
        assert os.path.exists(gt_full_path) and os.path.exists(pred_full_path), "clean pdb file do not create successfully"
        
        #metrics = EvaluateMetrics(gt_full_path, pred_full_path).metric()
        
        return pred_full_path, gt_full_path

# dataset = ComplexDataset('gt_1.csv', 'pred_1.csv', '/extendplus/wander_W/QA/', '/extendplus/wander_W/QA/clean_pdb/')
# _train, _test, _val = get_train_val_test_loader(dataset, 0.7, 0.1, 0.2, batch_size=1)
# print(len(_train), len(_test), len(_val))
# for i, (pdb_name, metrics) in enumerate(_val):
#     print('name : ', pdb_name)
#     print('metrics : ', metrics)
#     print(len(pdb_name))
#     break
