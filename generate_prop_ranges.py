import numpy as np 
import pandas as pd
import os 
import math
import pickle
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors
from tdc import Oracle

import mflow.utils.environment as env

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') 

GSK3B_scorer = Oracle(name = 'GSK3B')
SA_scorer = Oracle(name = 'SA')
DRD2_scorer = Oracle(name = 'DRD2')
JNK3_scorer = Oracle(name = 'JNK3')

def check_SA(gen_smiles):
    score = SA_scorer(Chem.MolToSmiles(gen_smiles))
    return score 

def check_DRD2(gen_smiles):
    score = DRD2_scorer(Chem.MolToSmiles(gen_smiles))
    return score

def check_JNK3(gen_smiles):
    score = JNK3_scorer(Chem.MolToSmiles(gen_smiles))
    return score

def check_GSK3B(gen_smiles):
    score = GSK3B_scorer(Chem.MolToSmiles(gen_smiles))
    return score

def check_plogp(mol):
    plogp = env.penalized_logp(mol)
    return plogp

def cache_prop_pred():
    prop_pred = {}
    for prop_name, function in Descriptors.descList:
        prop_pred[prop_name] = function
    prop_pred['sa'] = check_SA
    prop_pred['drd2'] = check_DRD2
    prop_pred['jnk3'] = check_JNK3 
    prop_pred['gsk3b'] = check_GSK3B
    prop_pred['plogp'] = check_plogp
    return prop_pred

def main():
    with open('data/zinc250k.csv') as f:
        train_smiles = [line.split(',', 2)[1] for line in f]
    train_smiles = train_smiles[1:]
    print(len(train_smiles))
    prop_pred = cache_prop_pred()
    # print('Prop pred fxns: ', prop_pred)
    props = ['qed', 'plogp', 'MolLogP']
    prop_ranges = []
    for prop in props:
        prop_values = []
        for smi in tqdm(train_smiles):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                prop_values.append(prop_pred[prop](mol))
        prop_ranges.append([min(prop_values), max(prop_values)])

    range_df = pd.DataFrame(prop_ranges, props)
    print(range_df)
    range_df.to_pickle('data/zinc250k_range.pkl')

if __name__ == '__main__':
    main()