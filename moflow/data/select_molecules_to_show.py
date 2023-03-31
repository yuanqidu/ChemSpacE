import pandas as pd
import numpy as np
import rdkit 
from rdkit import Chem
from rdkit.Chem import Draw
import json

def find_pair():
    file_name = 'zinc250k_property.csv'
    dataset = pd.read_csv(file_name)

    smiles = dataset['smile'].to_numpy()

    file_name = 'new_zinc250k_property.csv'
    dataset = pd.read_csv(file_name)
    dataset = dataset.drop(columns=['Unnamed: 0'])
    names = []
    for name in dataset:
        names.append(name)

    smiles_pair = []
    smiles_flat = []
    for name in names:
        prop = dataset[name].to_numpy()
        idx_max = np.argmax(prop)
        idx_min = np.argmin(prop)
        smiles_pair.append([smiles[idx_min],smiles[idx_max]])
        smiles_flat.append(int(idx_min))
        smiles_flat.append(int(idx_max))
    smiles_pair = np.array(smiles_pair)
    print (smiles_pair.shape)
    np.save('zinc250k_dir_pair.npy', smiles_pair)
    json.dump(smiles_flat,open('zinc250k_selected_pair.json','w'))

    file_name = 'qm9_property.csv'
    dataset = pd.read_csv(file_name)

    smiles = dataset['smile'].to_numpy()

    file_name = 'new_qm9_property.csv'
    dataset = pd.read_csv(file_name)
    dataset = dataset.drop(columns=['Unnamed: 0'])
    names = []
    for name in dataset:
        names.append(name)

    smiles_pair = []
    smiles_flat = []
    for name in names:
        prop = dataset[name].to_numpy()
        idx_max = np.argmax(prop)
        idx_min = np.argmin(prop)
        smiles_pair.append([smiles[idx_min],smiles[idx_max]])
        smiles_flat.append(int(idx_min))
        smiles_flat.append(int(idx_max))
        print (idx_max, idx_min)
        print (int(idx_max), int(idx_min))
    smiles_pair = np.array(smiles_pair)
    print (smiles_pair.shape)
    np.save('qm9_dir_pair.npy', smiles_pair)
    json.dump(smiles_flat,open('qm9_selected_pair.json','w'))

# file_name = 'chembl.txt'
# fread = open(file_name, 'r')
# smiles = []
# for line in fread:
#     smiles.append(line.replace('\n',''))
# smiles = np.array(smiles)

# file_name = 'new_chembl_property.csv'
# dataset = pd.read_csv(file_name)
# dataset = dataset.drop(columns=['Unnamed: 0'])
# names = []
# for name in dataset:
#     names.append(name)

# smiles_pair = []
# for name in names:
#     prop = dataset[name].to_numpy()
#     idx_max = np.argmax(prop)
#     idx_min = np.argmin(prop)
#     smiles_pair.append([smiles[idx_min],smiles[idx_max]])
# smiles_pair = np.array(smiles_pair)
# print (smiles_pair.shape)
# np.save('chembl_dir_pair.npy', smiles_pair)

file_name = 'zinc250k_property.csv'
dataset = pd.read_csv(file_name)

smiles = dataset['smile']
qed = dataset['qed'].to_numpy()
smiles = smiles.to_numpy()
idx_sort = np.argsort(qed)

Draw.MolToFile(Chem.MolFromSmiles(smiles[idx_sort[0]]), 'zinc_qed_min_1.png')
Draw.MolToFile(Chem.MolFromSmiles(smiles[idx_sort[-1]]), 'zinc_qed_max_1.png')

Draw.MolToFile(Chem.MolFromSmiles(smiles[idx_sort[1]]), 'zinc_qed_min_2.png')
Draw.MolToFile(Chem.MolFromSmiles(smiles[idx_sort[-2]]), 'zinc_qed_max_2.png')

Draw.MolToFile(Chem.MolFromSmiles(smiles[idx_sort[2]]), 'zinc_qed_min_3.png')
Draw.MolToFile(Chem.MolFromSmiles(smiles[idx_sort[-3]]), 'zinc_qed_max_3.png')

Draw.MolToFile(Chem.MolFromSmiles(smiles[idx_sort[3]]), 'zinc_qed_min_4.png')
Draw.MolToFile(Chem.MolFromSmiles(smiles[idx_sort[-4]]), 'zinc_qed_max_4.png')

Draw.MolToFile(Chem.MolFromSmiles(smiles[idx_sort[4]]), 'zinc_qed_min_5.png')
Draw.MolToFile(Chem.MolFromSmiles(smiles[idx_sort[-5]]), 'zinc_qed_max_5.png')

# print (sm_max, sm_min)
print (qed[idx_sort[0]], qed[idx_sort[-1]])