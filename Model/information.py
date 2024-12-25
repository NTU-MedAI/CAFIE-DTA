
#import pubchempy as pcp
import csv
import torch
#from pubchempy import get_compounds
import pandas as pd
import json
import time

path_infomation_KIBA='/home/ntu/fal_CPA/CPA1/p_c_information/KIBA/'
with open(path_infomation_KIBA+"all_compound_information1.txt") as f:
    com_info_KIBA=json.load(f)
#com_info_KIBA=json.load(open(path_infomation_KIBA+"all_compound_information1.txt"))

path_infomation_DAVIS='/home/ntu/fal_CPA/CPA1/p_c_information/Davis/'
com_info_DAVIS=json.load(open(path_infomation_DAVIS+"all_compound_information.txt"))




def write_csv(csv_path, M):
    with open(csv_path, 'w', encoding='utf-8', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(M)

def ite_write_csv(csv_path, row):

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
'''a_lane1_list = []  

#for i in range(0, 115184):
for i in range(0, 20):
    a_lane1_list.append(raw_data.values[i, 6])  
print(a_lane1_list)
#print("a_lane1_list = " + str(a_lane1_list))'''


'''
def get_infomatrion(smile_list):
    #for i in range(0, 115184):
    #for i in range(0, 20):
    features=[]
    for smile in smile_list:
        #for compound in get_compounds(input[i], 'smiles'):
        for compound in get_compounds(smile, 'smiles'):
            b1 = compound.cid
            c1 = compound.isomeric_smiles
            automs=count_atoms(c1)
            #print(automs)




        c = pcp.get_compounds(b1, 'cid')[0]
        cid = c.cid


        props = ['xlogp', 'tpsa', 'complexity', 'h_bond_donor_count',
                 'heavy_atom_count', 'charge',
                 'h_bond_acceptor_count']  

        #props = ['isomeric_smiles', 'xlogp', 'tpsa','complexity','molecular_weight','h_bond_donor_count','heavy_atom_count','charge','h_bond_acceptor_count']             #MolecularWeight,charge,HBondDonorCount，HeavyAtomCount，IsotopeAtomCount，
        compound = pcp.Compound.from_cid(cid)
        #print(dir(compound))
        info = [compound.__getattribute__(prop)  for prop in props ]
        #for i,a in enumerate(info):
        #  if a==None
        middle_c=info+automs
        features.append(middle_c)
        #print('middle_c:',middle_c)
    df_features = pd.DataFrame(features)
    df_features = df_features.fillna(0)
    #print('df_features:',df_features)

    np_features = df_features.to_numpy()
    tensor_feature=torch.tensor(np_features)
    float_feature = torch.tensor(tensor_feature, dtype=torch.float32)
    #print(a.size())
    return float_feature'''



def get_infomatrion_KIBA(smiles_list):
    #start_time = time.time()
    features = []
    for smiles in smiles_list:
        #print(protein_list)
        features.append(com_info_KIBA[smiles])

            #print('middle_c:',middle_c)
        df_features = pd.DataFrame(features)
        df_features = df_features.fillna(0)
        #print('df_features:',df_features)

        np_features = df_features.to_numpy()
        #tensor_feature=torch.tensor(np_features)
        float_feature = torch.as_tensor(np_features, dtype=torch.float32)
        #end_time = time.time()
        #times = end_time - start_time
        #print('times:', times)
        #print(float_feature)
    return   float_feature
'''


def get_infomatrion_KIBA(smiles_list):
    start_time=time.time()
    features = []
    for smiles in smiles_list:
        features.append(com_info_KIBA.get(smiles,{}))
        df_features=pd.DataFrame(features).fillna(0)
        np_features=df_features.to_numpy(dtype='float32')
        float_features=torch.tensor(np_features)
    end_time = time.time()
    times=end_time-start_time
    print('times:',times)
    return float_features

'''
def get_infomatrion_DAVIS(smiles_list):

    features = []
    for smiles in smiles_list:
        #print(smiles)
        features.append(com_info_DAVIS[smiles])

            #print('middle_c:',middle_c)
        df_features = pd.DataFrame(features)
        df_features = df_features.fillna(0)
        #print('df_features:',df_features)

        np_features = df_features.to_numpy()
        #tensor_feature=torch.tensor(np_features)
        float_feature = torch.as_tensor(np_features, dtype=torch.float32)
        #print(float_feature.size())
    return   float_feature










