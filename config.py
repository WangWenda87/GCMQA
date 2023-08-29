import torch as t
  
device      = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
FloatTensor = t.cuda.FloatTensor if t.cuda.is_available() else t.FloatTensor
LongTensor  = t.cuda.LongTensor if t.cuda.is_available() else t.LongTensor