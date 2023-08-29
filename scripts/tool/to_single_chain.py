import numpy as np
import os
import shutil

def chains(pdb : str) : 
    with open(pdb, 'r') as file:
        content = file.readlines()
    chain = []
    for line in content:
        if line[:4] == 'ATOM' : 
            chain_id = line[21]
            if chain_id not in chain and chain_id != " ": 
                chain.append(chain_id)
    return chain, len(chain)

def split_chains(pdb : str, chain_id : str) : 
    with open(pdb, 'r') as file:
        content = file.readlines()
    for line in content : 
        fb = open(chain_id + ".pdb", 'a')
        if line[:4] == 'ATOM' and line[21] == chain_id : 
            fb.write(line)
        fb.close()
    return None

def update(pdb : str, base_chain : str, add_chain : str) : 
    split_chains(pdb, add_chain)
    f_base = open(base_chain + '.pdb', 'rb')
    base_content = f_base.readlines()

    last_res = int(base_content[-1][22:26])
    f_add = open(add_chain + '.pdb', 'rb')
    add_content = f_add.readlines()
    for line in add_content : 
        new_res = str(int(line[22:26]) + last_res).rjust(4)
        new_line = line[:21] + base_chain.encode() + new_res.encode() + line[26:]
        f_new = open(add_chain + '_new.pdb', 'a+')
        f_new.write(new_line.decode())
    f = open('combine.pdb', 'a+')
    for line in open(base_chain + '.pdb') : 
        f.writelines(line)
    for line in open(add_chain + '_new.pdb') : 
        f.writelines(line)
    os.rename('combine.pdb', base_chain + '.pdb')    
    f_base.close()
    f_add.close()
    f_new.close()
    
    os.remove(add_chain + '_new.pdb')
    os.remove(add_chain + '.pdb')
    return None

def single_chain(pdb: str, single_chain_folder='scripts/tool/single_chain_pdb/', _type='pred_') : 
    name = pdb.split('/')[-1].split('.')[0]
    chain_list = chains(pdb)[0]
    chain_num = chains(pdb)[1]
    split_chains(pdb, chain_list[0])
    for i in range(1, chain_num) : 
        update(pdb, chain_list[0], chain_list[i])
    
    os.rename(chain_list[0] + '.pdb', name + '_single_chain.pdb')
    shutil.move(name + '_single_chain.pdb', single_chain_folder)
    os.rename(single_chain_folder + name + '_single_chain.pdb', single_chain_folder + _type + name + '.pdb')
    
    return None
