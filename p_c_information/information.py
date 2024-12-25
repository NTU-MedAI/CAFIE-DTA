import pubchempy as pcp
import csv
import torch
from pubchempy import get_compounds
import pandas as pd
import math
import json
import pickle
import numpy as np
from collections import OrderedDict

path='/home/ntu/Documents/feiailu/MY_Module/p_c_information/Davis/'
ligands = json.load(open(path + "ligands_can.txt"), object_pairs_hook=OrderedDict)

XD = []
XT = []
for d in ligands.keys():
    XD.append(ligands[d])
#print(XD)


#file_path = r'/home/ntu/Documents/feiailu/MY_Module/DATA/DATAdrug1.csv'  # r对路径进行转义，windows需要
#raw_data = pd.read_csv(file_path, header=0)  # header=0表示第一行是表头，就自动去除了
#raw_data = pd.read_csv(file_path, header=None)  # 不设置表头
#print(raw_data)



#csv_path = "/home/ntu/Documents/feiailu/MY_Module/p_c_information/info.csv"

def write_csv(csv_path, M):
    with open(csv_path, 'w', encoding='utf-8', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(M)

def ite_write_csv(csv_path, row):
    '''
    追加写入，每次只写入一行
    '''
    with open(csv_path, 'a+', newline='', encoding='utf-8') as csvfile:
        csv_write = csv.writer(csvfile)
        csv_write.writerow(row)

def count_atoms(chformula):
    atoms = ['C', 'H', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I']
    atoms_count = {atom: 0 for atom in atoms}
    for atom in atoms_count.keys():
        count = chformula.count(atom)
        atoms_count[atom] = count
    return list(atoms_count.values())

save_lsit=[]
'''a_lane1_list = []  # 车道1

#for i in range(0, 115184):
for i in range(0, 20):
    a_lane1_list.append(raw_data.values[i, 6])  # 读取excel第一列的值，并添加到列表
print(a_lane1_list)
#print("a_lane1_list = " + str(a_lane1_list))'''



#dict_compound_information={'CC1=CC2=CC3=C(OC(=O)C=C3C)C(C)=C2O1': [3, 39.4, 374, 0, 17, 0, 3, 14, 0, 0, 3, 0, 0, 0, 0, 0], 'NCC(=O)CCC(O)=O': [-3.8, 80.4, 121, 2, 9, 0, 4, 5, 0, 1, 3, 0, 0, 0, 0, 0], 'O=[Ti]=O': [None, 34.1, 18.3, 0, 3, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0], 'O=C(C1=CC=CC=C1)C1=CC=CC=C1': [3.4, 17.1, 165, 0, 14, 0, 1, 13, 0, 0, 1, 0, 0, 0, 0, 0], 'CC1=C(C)C=C2N(C[C@H](O)[C@H](O)[C@H](O)CO)C3=NC(=O)NC(=O)C3=NC2=C1': [-1.5, 155, 680, 5, 27, 0, 7, 17, 3, 4, 6, 0, 0, 0, 0, 0], 'CC(C(O)=O)C1=CC2=C(C=C1)C1=C(N2)C=CC(Cl)=C1': [4, 53.1, 362, 2, 19, 0, 2, 16, 0, 1, 2, 0, 0, 1, 0, 0], '[H][C@]12CC[C@]3([H])[C@]([H])(C[C@@H](O)[C@]4(C)[C@H](CC[C@]34O)C3=CC(=O)OC3)[C@@]1(C)CC[C@@H](C2)O[C@H]1C[C@H](O)[C@H](O[C@H]2C[C@H](O)[C@H](O[C@H]3C[C@H](O)[C@H](O)[C@@H](C)O3)[C@@H](C)O2)[C@@H](C)O1': [1.3, 203, 1450, 6, 55, 0, 14, 41, 18, 0, 14, 0, 0, 0, 0, 0], '[H][C@]12[C@H](OC(=O)C3=CC=CC=C3)[C@]3(O)C[C@H](OC(=O)[C@H](O)[C@@H](NC(=O)C4=CC=CC=C4)C4=CC=CC=C4)C(C)=C([C@@H](OC(C)=O)C(=O)[C@]1(C)[C@@H](O)C[C@H]1OC[C@@]21OC(C)=O)C3(C)C': [2.5, 221, 1790, 4, 62, 0, 14, 47, 8, 1, 14, 0, 0, 0, 0, 0], '[H][C@@]12C[C@H](O)[C@@]3(C)C(=O)[C@H](O)C4=C(C)[C@H](C[C@@](O)([C@@H](OC(=O)C5=CC=CC=C5)[C@]3([H])[C@@]1(CO2)OC(C)=O)C4(C)C)OC(=O)[C@H](O)[C@@H](NC(=O)OC(C)(C)C)C1=CC=CC=C1': [1.6, 224, 1660, 5, 58, 0, 14, 43, 8, 1, 14, 0, 0, 0, 0, 0], '[H][C@@](O)([C@@H](NC(=O)OC(C)(C)C)C1=CC=CC=C1)C(=O)O[C@@]1([H])C[C@@]2(O)[C@@]([H])(OC(=O)C3=CC=CC=C3)[C@]3([H])[C@@]4(CO[C@]4([H])C[C@]([H])(OC)[C@@]3(C)C(=O)[C@]([H])(OC)C(=C1C)C2(C)C)OC(C)=O': [2.7, 202, 1690, 3, 60, 0, 14, 45, 8, 1, 14, 0, 0, 0, 0, 0], '[H][C@]12CC[C@]3([H])[C@]([H])(CC[C@]4(C)[C@H](CC[C@]34O)C3=CC(=O)OC3)[C@@]1(C)CC[C@@H](C2)O[C@H]1C[C@H](O)[C@H](O[C@H]2C[C@H](O)[C@H](O[C@H]3C[C@H](O)[C@H](O)[C@@H](C)O3)[C@@H](C)O2)[C@@H](C)O1': [2.3, 183, 1410, 5, 54, 0, 13, 41, 17, 0, 13, 0, 0, 0, 0, 0], '[H][C@@]12CC[C@]3(O)C[C@H](C[C@@H](O)[C@]3(CO)[C@@]1([H])[C@H](O)C[C@]1(C)[C@H](CC[C@]21O)C1=CC(=O)OC1)O[C@@H]1O[C@@H](C)[C@H](O)[C@@H](O)[C@H]1O': [-1.7, 207, 1080, 8, 41, 0, 12, 29, 11, 0, 12, 0, 0, 0, 0, 0], 'CC(C(O)=O)C1=CC=C(S1)C(=O)C1=CC=CC=C1': [3.3, 82.6, 323, 1, 18, 0, 4, 14, 0, 0, 3, 0, 1, 0, 0, 0], 'CO[C@H]1\\C=C\\O[C@@]2(C)OC3=C(C2=O)C2=C(C(O)=C3C)C(=O)C(NC(=O)\\C(C)=C/C=C/[C@H](C)[C@H](O)[C@@H](C)[C@@H](O)[C@@H](C)[C@H](OC(C)=O)[C@@H]1C)=C1NC3(CCN(CC3)CC(C)C)N=C21': [5.6, 209, 1880, 5, 61, 0, 14, 46, 8, 4, 11, 0, 0, 0, 0, 0], 'O=C1NC(=O)C(N1)(C1=CC=CC=C1)C1=CC=CC=C1': [2.5, 58.2, 350, 2, 19, 0, 2, 15, 0, 2, 2, 0, 0, 0, 0, 0], 'CO[C@H]1\\C=C\\O[C@@]2(C)OC3=C(C2=O)C2=C(O)C(\\C=N\\N4CCN(C)CC4)=C(NC(=O)\\C(C)=C/C=C/[C@H](C)[C@H](O)[C@@H](C)[C@@H](O)[C@@H](C)[C@H](OC(C)=O)[C@@H]1C)C(O)=C2C(O)=C3C': [4.9, 220, 1620, 6, 59, 0, 15, 43, 8, 4, 12, 0, 0, 0, 0, 0], 'OP(O)(=O)OCN1C(=O)NC(C1=O)(C1=CC=CC=C1)C1=CC=CC=C1': [0.6, 116, 547, 3, 25, 0, 6, 16, 0, 2, 6, 0, 0, 0, 0, 0], 'CCC1(C(=O)NCNC1=O)C1=CC=CC=C1': [0.9, 58.2, 279, 2, 16, 0, 2, 12, 0, 2, 2, 0, 0, 0, 0, 0], 'CCCC(C)C1(CC)C(=O)NC(=O)NC1=O': [2.1, 75.3, 305, 2, 16, 0, 3, 11, 0, 2, 3, 0, 0, 0, 0, 0], 'CC1=C2NC(=O)C3=C(N=CC=C3)N(C3CC3)C2=NC=C1': [2, 58.1, 397, 1, 20, 0, 4, 15, 0, 4, 1, 0, 0, 0, 0, 0]}

dict_compound_information={}

smile_list=['CC1=CC2=CC3=C(OC(=O)C=C3C)C(C)=C2O1', 'NCC(=O)CCC(O)=O', 'O=[Ti]=O', 'O=C(C1=CC=CC=C1)C1=CC=CC=C1', 'CC1=C(C)C=C2N(C[C@H](O)[C@H](O)[C@H](O)CO)C3=NC(=O)NC(=O)C3=NC2=C1', 'CC(C(O)=O)C1=CC2=C(C=C1)C1=C(N2)C=CC(Cl)=C1', '[H][C@]12CC[C@]3([H])[C@]([H])(C[C@@H](O)[C@]4(C)[C@H](CC[C@]34O)C3=CC(=O)OC3)[C@@]1(C)CC[C@@H](C2)O[C@H]1C[C@H](O)[C@H](O[C@H]2C[C@H](O)[C@H](O[C@H]3C[C@H](O)[C@H](O)[C@@H](C)O3)[C@@H](C)O2)[C@@H](C)O1', '[H][C@]12[C@H](OC(=O)C3=CC=CC=C3)[C@]3(O)C[C@H](OC(=O)[C@H](O)[C@@H](NC(=O)C4=CC=CC=C4)C4=CC=CC=C4)C(C)=C([C@@H](OC(C)=O)C(=O)[C@]1(C)[C@@H](O)C[C@H]1OC[C@@]21OC(C)=O)C3(C)C', '[H][C@@]12C[C@H](O)[C@@]3(C)C(=O)[C@H](O)C4=C(C)[C@H](C[C@@](O)([C@@H](OC(=O)C5=CC=CC=C5)[C@]3([H])[C@@]1(CO2)OC(C)=O)C4(C)C)OC(=O)[C@H](O)[C@@H](NC(=O)OC(C)(C)C)C1=CC=CC=C1', '[H][C@@](O)([C@@H](NC(=O)OC(C)(C)C)C1=CC=CC=C1)C(=O)O[C@@]1([H])C[C@@]2(O)[C@@]([H])(OC(=O)C3=CC=CC=C3)[C@]3([H])[C@@]4(CO[C@]4([H])C[C@]([H])(OC)[C@@]3(C)C(=O)[C@]([H])(OC)C(=C1C)C2(C)C)OC(C)=O', '[H][C@]12CC[C@]3([H])[C@]([H])(CC[C@]4(C)[C@H](CC[C@]34O)C3=CC(=O)OC3)[C@@]1(C)CC[C@@H](C2)O[C@H]1C[C@H](O)[C@H](O[C@H]2C[C@H](O)[C@H](O[C@H]3C[C@H](O)[C@H](O)[C@@H](C)O3)[C@@H](C)O2)[C@@H](C)O1', '[H][C@@]12CC[C@]3(O)C[C@H](C[C@@H](O)[C@]3(CO)[C@@]1([H])[C@H](O)C[C@]1(C)[C@H](CC[C@]21O)C1=CC(=O)OC1)O[C@@H]1O[C@@H](C)[C@H](O)[C@@H](O)[C@H]1O', 'CC(C(O)=O)C1=CC=C(S1)C(=O)C1=CC=CC=C1', 'CO[C@H]1\\C=C\\O[C@@]2(C)OC3=C(C2=O)C2=C(C(O)=C3C)C(=O)C(NC(=O)\\C(C)=C/C=C/[C@H](C)[C@H](O)[C@@H](C)[C@@H](O)[C@@H](C)[C@H](OC(C)=O)[C@@H]1C)=C1NC3(CCN(CC3)CC(C)C)N=C21', 'O=C1NC(=O)C(N1)(C1=CC=CC=C1)C1=CC=CC=C1', 'CO[C@H]1\\C=C\\O[C@@]2(C)OC3=C(C2=O)C2=C(O)C(\\C=N\\N4CCN(C)CC4)=C(NC(=O)\\C(C)=C/C=C/[C@H](C)[C@H](O)[C@@H](C)[C@@H](O)[C@@H](C)[C@H](OC(C)=O)[C@@H]1C)C(O)=C2C(O)=C3C', 'OP(O)(=O)OCN1C(=O)NC(C1=O)(C1=CC=CC=C1)C1=CC=CC=C1', 'CCC1(C(=O)NCNC1=O)C1=CC=CC=C1', 'CCCC(C)C1(CC)C(=O)NC(=O)NC1=O', 'CC1=C2NC(=O)C3=C(N=CC=C3)N(C3CC3)C2=NC=C1']
smile_list1=['CC(C(=O)NC(CO)C(=O)NC(CCCN=C(N)N)C(=O)NC(CCCN=C(N)N)C(=O)NC(CCCN=C(N)N)C(=O)NC(CCCN=C(N)N)C(=O)NC(CCCN=C(N)N)C(=O)NC(CCCN=C(N)N)C(=O)O)NC(=O)C(C)NS(=O)(=O)C1=CC=CC2=C1C=CN=C2']
def get_infomatrion(smile_list):
    #for i in range(0, 115184):
    #for i in range(0, 20):
    features=[]
    for smile in smile_list:
        #for compound in get_compounds(input[i], 'smiles'):
        for compound in get_compounds(smile, 'smiles'):
            if smile=="CCOC1=CC(=C(C=C1)C=C2C(=O)N=C(S2)N)O":
                b1='4589745'
            elif smile=="C1CN(CC1O)CC2=NC(=O)C3=C(N2)C4=C(S3)C=CC(=C4)C5=CC=C(C=C5)O":
                b1='135927974'
            elif smile=='C1C2C(C3C(C(O2)O)OC(=O)C4=CC(=C(C(=C4C5=C(C(=C(C=C5C(=O)O3)O)O)O)O)O)O)OC(=O)C6=CC(=C(C(=C6C7=C(C(=C8C9=C7C(=O)OC2=C(C(=C(C3=C(C(=C(C=C3C1=O)O)O)O)C(=C92)C(=O)O8)O)O)O)O)O)O)O':
                b1='44460933'
            elif smile =='C1C2C(C3C4C(C5=C(C(=C(C(=C5C(=O)O4)C6=C(C(=C(C(=C6C(=O)O3)C7=C(C(=C(C=C7C(=O)O2)O)O)O)O)O)O)O)O)O)O)OC(=O)C8=CC(=C(C(=C8C9=C(C(=C2C3=C9C(=O)OC4=C(C(=C(C5=C(C(=C(C=C5C(=O)O1)O)O)O)C(=C34)C(=O)O2)O)O)O)O)O)O)O':
                b1='44460932'
            elif smile=='CC12C(C(CC(O1)N3C4=C(C=C(C=C4)O)C5=C6C(=C7C8=C(N2C7=C53)C=CC(=C8)O)COC6=O)NC)OC':
                b1='10074667 '
            elif smile=='C1C2C(C(C(C(O2)O)O)O)OC(=O)C3=CC(=C(C(=C3C4=C(C(=C5C6=C4C(=O)OC7=C(C(=C(C8=C(C(=C(C=C8C1=O)O)O)O)C(=C67)C(=O)O5)O)O)O)O)O)O)O':
                b1='44460577 '
            else:
                b1 = compound.cid
            #print("compound:",compound)
            c1 = compound.isomeric_smiles
            automs=count_atoms(c1)
            #print(automs)


        # 获取物CID

        c = pcp.get_compounds(b1, 'cid')[0]
        cid = c.cid

        # 获取物理化学信息
        props = ['xlogp', 'tpsa', 'complexity', 'h_bond_donor_count',
                 'heavy_atom_count', 'charge',
                 'h_bond_acceptor_count']  # molecular_weight,xlogp:亲脂性    TPSA:拓扑分子极性表面积 是常用于药物化学的一个参数，其定义为化合物内极性分子的总表面积 complexity:复杂性

        #props = ['isomeric_smiles', 'xlogp', 'tpsa','complexity','molecular_weight','h_bond_donor_count','heavy_atom_count','charge','h_bond_acceptor_count']     #xlogp:亲脂性    TPSA:拓扑分子极性表面积 是常用于药物化学的一个参数，其定义为化合物内极性分子的总表面积 complexity:复杂性
        #MolecularWeight,charge,HBondDonorCount，HeavyAtomCount，IsotopeAtomCount，
        compound = pcp.Compound.from_cid(cid)
        #print(dir(compound))
        info = [compound.__getattribute__(prop)  for prop in props ]
        #for i,a in enumerate(info):
        #  if a==None
        # 将信息转化为矩阵形式
        middle_c=info+automs
        #print("middle_c:",middle_c)
        dict_compound_information[smile]=middle_c
        print(dict_compound_information)
        #print("dict_compound_information:",dict_compound_information)

    #     #features.append(dict_compound_information[smile])
    #     features.append(middle_c)
    #     df_features = pd.DataFrame(features)
    #     df_features = df_features.fillna(0)
    # #print('df_features:',df_features)
    #
    # np_features = df_features.to_numpy()
    # tensor_feature=torch.tensor(np_features)
    # float_feature = torch.tensor(tensor_feature, dtype=torch.float32)
    # #print(a.size())
    # return float_feature


get_infomatrion(XD)



path_in='/home/ntu/Documents/feiailu/MY_Module/p_c_information/Davis/all_compound_information.txt'
with open(path_in,"w") as f:
     f.write(json.dumps(dict_compound_information))







