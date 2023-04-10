import argparse
import os
import sys
# for linux env.
sys.path.insert(0,'..')
from distutils.util import strtobool

import torch

import numpy as np
import pandas as pd
import torch

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw
from tdc import Oracle

from sklearn.preprocessing import scale, StandardScaler

from data.data_loader import NumpyTupleDataset
from data import transform_qm9, transform_zinc250k
from data.transform_zinc250k import zinc250_atomic_num_list, transform_fn_zinc250k
# from mflow.generate import generate_mols_along_axis
from mflow.models.hyperparams import Hyperparameters
from mflow.models.utils import check_validity, adj_to_smiles, _to_numpy_array
from mflow.utils.model_utils import load_model, smiles_to_adj
from mflow.models.model import rescale_adj
import mflow.utils.environment as env
import time
import functools
print = functools.partial(print, flush=True)

GSK3B_scorer = Oracle(name = 'GSK3B')
DRD2_scorer = Oracle(name = 'DRD2')
JNK3_scorer = Oracle(name = 'JNK3')

# Define helper oracle functions that take in a RDKit Mol object and return the specfied molecular property
def check_DRD2(gen_smiles):
    score = DRD2_scorer(Chem.MolToSmiles(gen_smiles))
    return score

def check_JNK3(gen_smiles):
    score = JNK3_scorer(Chem.MolToSmiles(gen_smiles))
    return score

def check_GSK3B(gen_smiles):
    score = GSK3B_scorer(Chem.MolToSmiles(gen_smiles))
    return score

def get_z(model, mol, device):
    """
    Convert a given SMILES string into its corresponding molecular graph representation and use a trained model to 
    extract a latent representation vector from the molecular graph. The returned representation vector is in the form 
    of a NumPy array.

    Args:
        model: An instance of a trained PyTorch model
        mol (str): A SMILES string
        device (torch.device, optional): A PyTorch device instance to be used for running the computation. Default is None, 
                                        indicating that the computation will be performed on the CPU.

    Returns:
        z (ndarray): A NumPy array representing the molecular latent representation extracted from the given molecular 
                   structure.
    """
    adj_idx, x_idx = smiles_to_adj(mol, data_name='zinc250k')
    if device:
        adj_idx = adj_idx.to(device)
        x_idx = x_idx.to(device)
        adj_normalized = rescale_adj(adj_idx).to(device)
    z_idx, _ = model(adj_idx, x_idx, adj_normalized)
    z_idx[0] = z_idx[0].reshape(z_idx[0].shape[0], -1)
    z_idx[1] = z_idx[1].reshape(z_idx[1].shape[0], -1)
    z_idx = torch.cat((z_idx[0], z_idx[1]), dim=1).squeeze(dim=0)  # h:(1,45), adj:(1,324) -> (1, 369) -> (369,)
    z = np.expand_dims(_to_numpy_array(z_idx), axis=0)
    return z 


def optimize_mol(model, smiles, direction, num_range, path_len=21, sim_cutoff=0,
             data_name='zinc250k', atomic_num_list=zinc250_atomic_num_list, property_name='plogp', device=None):
    """
    Optimize a given molecular structure for a single property.

    Args:
        model: An instance of a trained PyTorch model to be used for molecular optimization in latent space.
        smiles (str): A SMILES string representing the molecular structure to be optimized.
        direction (ndarray): A NumPy array specifying the direction in the latent space to optimize the molecule towards. 
                            The array has the same shape as the latent representation vector.
        num_range (float): A scalar value specifying the range of the optimization path in the given direction.
        path_len (int, optional): An integer value specifying the number of points on the optimization path. Default is 21.
        sim_cutoff (float, optional): A scalar value specifying the Tanimoto similarity threshold for accepting a new 
                                        molecular structure. Default is 0.
        data_name (str, optional): The name of the dataset used to train the model. Defaults to 'zinc250k'.
        atomic_num_list (list, optional): A list of atomic numbers to be used for molecular graph construction. Default is 
                                            'zinc250_atomic_num_list'.
        property_name (str, optional): A string specifying the molecular property name to be optimized. Default is 'plogp'.
        device (torch.device, optional): The device to use for processing. Default is None, indicating that the computation 
                                            will be performed on the CPU.

    Returns:
        smiles_path (list): A list of SMILES strings representing the molecular structures along the optimization path.
        results (list): A list of tuples containing the SMILES string, the corresponding property value, the Tanimoto 
                        similarity to the input molecule, and the input molecule SMILES string for each valid molecule 
                        generated during the optimization process.
        start (tuple): A tuple containing the input molecule SMILES string, its corresponding property value, and None.
    """
    if property_name == 'qed':
        propf = env.qed  # [0,1]
    elif property_name == 'plogp':
        propf = env.penalized_logp  # unbounded, normalized later???
    elif property_name == 'drd2':
        propf = check_DRD2 
    elif property_name == 'jnk3':
        propf = check_JNK3 
    elif property_name == 'gsk3b':
        propf = check_GSK3B 
    else:
        raise ValueError("Wrong property_name{}".format(property_name))
    model.eval()
    
    mol = Chem.MolFromSmiles(smiles)
    fp1 = AllChem.GetMorganFingerprint(mol, 2)
    start = (smiles, propf(mol), None) # , mol)
    z = get_z(model, smiles, device)
    with torch.no_grad():
        distances = np.linspace(-num_range,num_range,path_len)
        distances = distances.tolist()
        print ('Z Shape: ', z.shape, 'Direction Shape: ', direction.shape)
        z_to_decode = []
        for j in range(len(distances)):
            z_to_decode.append(z + distances[j]*direction)
        z_to_decode = torch.from_numpy(np.array(z_to_decode)).squeeze().float().to(device)
        adj, x = model.reverse(z_to_decode)
        smiles_path = adj_to_smiles(adj.cpu(), x.cpu(), atomic_num_list)
    val_res = check_validity(adj, x, atomic_num_list)
    valid_mols = val_res['valid_mols']
    valid_smiles = val_res['valid_smiles']

    results = []
    sm_set = set()
    sm_set.add(smiles)
    for m, s in zip(valid_mols, valid_smiles):
        if s in sm_set:
            continue
        sm_set.add(s)
        p = propf(m)
        fp2 = AllChem.GetMorganFingerprint(m, 2)
        sim = DataStructs.TanimotoSimilarity(fp1, fp2)
        if sim >= sim_cutoff:
            results.append((s, p, sim, smiles))
    # smile, property, similarity, mol
    results.sort(key=lambda tup: tup[1], reverse=True)
    return smiles_path, results, start

def optimize_mol_multi_prop(model, smiles, directions, num_range, path_len=21, sim_cutoff=0,
             data_name='zinc250k', atomic_num_list=zinc250_atomic_num_list, device=None):
    """
    Optimize a given molecular structure for multiple properties.

    Args:
        model (nn.Module): An instance of a trained PyTorch model to be used for molecular optimization in latent space.
        smiles (str): A SMILES string representing the molecular structure to be optimized.
        directions (list of numpy.ndarray): A list of direction vectors corresponding to the properties to be optimized.
        num_range (float): A scalar value specifying the range of the optimization path in the given directions.
        path_len (int, optional): The number of intermediate points on the optimization path. Defaults to 21.
        sim_cutoff (float, optional):  A scalar value specifying the Tanimoto similarity threshold for accepting a new 
                                    molecular structure. Defaults to 0.
        data_name (str, optional): The name of the dataset used to train the model. Defaults to 'zinc250k'.
        atomic_num_list (list of int, optional): A list of atomic numbers for the atoms present in the molecular structure. Defaults to zinc250_atomic_num_list.
        device (torch.device, optional): The device to use for processing. Default is None, indicating that the computation will be performed on the CPU.

    Returns:
        smiles_path (list of str): A list of SMILES strings representing the optimized molecular structures at each intermediate point.
        results_1 (list of tuple): A list of tuples, where each tuple contains the SMILES string, the property value and the Tanimoto similarity of the optimized molecular structures with respect to the first property.
        results_2 (list of tuple): A list of tuples, where each tuple contains the SMILES string, the property value and the Tanimoto similarity of the optimized molecular structures with respect to the second property.
        start_1 (tuple): A tuple containing the SMILES string, the property value and None for the starting molecular structure with respect to the first property.
        start_2 (tuple): A tuple containing the SMILES string, the property value and None for the starting molecular structure with respect to the second property.
    """
    propf_1 = env.qed  # [0,1]
    propf_2 = env.penalized_logp  # unbounded, normalized later???
    if len(directions) > 1:
        combined_directions = directions[0] * directions[1]
        pos_attributes = (combined_directions >= 0) * directions[1]
        direction = directions[0] + pos_attributes
    else:
        direction = directions[0]
    model.eval()
    
    mol = Chem.MolFromSmiles(smiles)
    fp1 = AllChem.GetMorganFingerprint(mol, 2)

    start_1 = (smiles, propf_1(mol), None) # , mol)
    start_2 = (smiles, propf_2(mol), None) # , mol)
    z = get_z(model, smiles, device)
    with torch.no_grad():
        distances = np.linspace(-num_range,num_range,path_len)
        distances = distances.tolist()
        print ('Z Shape: ', z.shape, 'Direction Shape: ', direction.shape)
        z_to_decode = []
        for j in range(len(distances)):
            z_to_decode.append(z + distances[j]*direction)
        z_to_decode = torch.from_numpy(np.array(z_to_decode)).squeeze().float().to(device)
        adj, x = model.reverse(z_to_decode)
        smiles_path = adj_to_smiles(adj.cpu(), x.cpu(), atomic_num_list)
    val_res = check_validity(adj, x, atomic_num_list)
    valid_mols = val_res['valid_mols']
    valid_smiles = val_res['valid_smiles']

    results_1 = []
    results_2 = []
    sm_set = set()
    sm_set.add(smiles)
    for m, s in zip(valid_mols, valid_smiles):
        if s in sm_set:
            continue
        sm_set.add(s)
        p1 = propf_1(m)
        p2 = propf_2(m)
        fp2 = AllChem.GetMorganFingerprint(m, 2)
        sim = DataStructs.TanimotoSimilarity(fp1, fp2)
        if sim >= sim_cutoff:
            results_1.append((s, p1, sim, smiles))
            results_2.append((s, p2, sim, smiles))

    return smiles_path, results_1, results_2, start_1, start_2

def load_property_csv(data_name):
    """
    Load property values from a CSV file for a given dataset.

    Args:
        data_name (str): Name of the dataset, either 'qm9' or 'zinc250k'.

    Returns:
        tuples (list): A list of tuples, each containing property values for a single molecule.
        prop_to_idx (dict): A dictionary that maps property names to their corresponding index in the tuple.
    """
    if data_name == 'qm9':
        # Total: 133885	
        filename = 'data/qm9_properties.csv'
    elif data_name == 'zinc250k':
        # Total: 249455	
        filename = 'data/zinc250k_properties.csv'

    df = pd.read_csv(filename)  # smile,qed,plogp,MolLogP,MolWt,sa,drd2,jnk3,gsk3b 

    tuples = [tuple(x) for x in df.values]
    props = ['smile','qed','plogp','MolLogP','MolWt','sa','drd2','jnk3','gsk3b']
    values = np.linspace(0,8,9, dtype=int)
    prop_to_idx = dict(zip(props, values))
    print('Load {} done, length: {}'.format(filename, len(tuples)))
    return tuples, prop_to_idx

def find_top_score_smiles(model, device, data_name, property_name, prop_to_idx, train_prop, topk, atomic_num_list, path):
    """
    Find top k optimized molecules for a given property.

    Args:
        model (object): A PyTorch model for optimizing molecules.
        device (str): The device to use for optimization.
        data_name (str): The name of the dataset.
        property_name (str): The name of the property score to optimize for.
        prop_to_idx (dict): A dictionary that maps property names to their corresponding column index in the dataset.
        train_prop (list): A list of tuples containing the molecular properties of the training dataset.
        topk (int): The number of optimized molecules to return.
        atomic_num_list (list): A list of atomic numbers to consider for the optimized molecules.
        path (str): The path to save the optimized molecules.

    Returns:
        Saves a list of tuples containing the optimized molecules, their corresponding property score,
        their similarity to the reference molecule and the reference molecule itself to a csv at the given path

    """
    start_time = time.time()
    idx = prop_to_idx[property_name] # smile,qed,plogp,MolLogP,MolWt,sa,drd2,jnk3,gsk3b 
    print('Finding top {} score'.format(property_name))
    train_prop_sorted = sorted(train_prop, key=lambda tup: tup[idx], reverse=True)  
    if not os.path.exists('./'+data_name+'_chemspace_opt/'+f'{path}'):
        os.makedirs('./'+data_name+'_chemspace_opt/'+f'{path}')
    direction = np.load('./boundaries_'+args.data_name+'/boundary_'+property_name+'.npy')
    result_list = []
    for i, r in enumerate(train_prop_sorted):
        if i >= topk:
            break
        if i % 50 == 0:
            print('Optimization {}/{}, time: {:.2f} seconds'.format(i, topk, time.time() - start_time))
        smile = r[prop_to_idx['smile']]
        _ , results, _ = optimize_mol(model, smile, direction, num_range=100, path_len=21, sim_cutoff=0.0,
                                    data_name=data_name, atomic_num_list=atomic_num_list,
                                    property_name=property_name, device=device)
        result_list.extend(results)  # results: [(smile2, property, sim, smile1), ...]

    result_list.sort(key=lambda tup: tup[1], reverse=True)

    # check novelty
    train_smile = set()
    for i, r in enumerate(train_prop_sorted):
        smile = r[prop_to_idx['smile']]
        train_smile.add(smile)
        mol = Chem.MolFromSmiles(smile)
        smile2 = Chem.MolToSmiles(mol, isomericSmiles=True)
        train_smile.add(smile2)

    result_list_novel = []

    start_time = time.time()
    idx = prop_to_idx[property_name] # smile,qed,plogp,MolLogP,MolWt,sa,drd2,jnk3,gsk3b 
    print('Finding top {} score'.format(property_name))
    train_prop_sorted = sorted(train_prop, key=lambda tup: tup[idx], reverse=True)  
    if not os.path.exists('./'+data_name+'_chemspace_opt/'+f'{path}'):
        os.makedirs('./'+data_name+'_chemspace_opt/'+f'{path}')
    direction = np.load('./boundaries_'+args.data_name+'/boundary_'+property_name+'.npy')
    result_list = []
    for i, r in enumerate(train_prop_sorted):
        if i >= topk:
            break
        if i % 50 == 0:
            print('Optimization {}/{}, time: {:.2f} seconds'.format(i, topk, time.time() - start_time))
        smile = r[prop_to_idx['smile']]
        _ , results, _ = optimize_mol(model, smile, direction, num_range=100, path_len=21, sim_cutoff=0.0,
                                    data_name=data_name, atomic_num_list=atomic_num_list,
                                    property_name=property_name, device=device)
        result_list.extend(results)  # results: [(smile2, property, sim, smile1), ...]

    result_list.sort(key=lambda tup: tup[1], reverse=True)

    # check novelty
    train_smile = set()
    for i, r in enumerate(train_prop_sorted):
        smile = r[prop_to_idx['smile']]
        train_smile.add(smile)
        mol = Chem.MolFromSmiles(smile)
        smile2 = Chem.MolToSmiles(mol, isomericSmiles=True)
        train_smile.add(smile2)

    result_list_novel = []
    for i, r in enumerate(result_list):
        smile, score, sim, smile_original = r
        if smile not in train_smile:
            result_list_novel.append(r)

    # dump results
    f = open('./'+data_name+'_chemspace_opt/'+f'{path}' + '_discovered_sorted.csv', "w")
    for r in result_list_novel:
        smile, score, sim, smile_original = r
        f.write('{},{},{},{}\n'.format(score, smile, sim, smile_original))
        f.flush()
    f.close()
    print('Dump done!')

def find_top_score_smiles_multi_prop(model, device, data_name, property_name, prop_to_idx, train_prop, topk, atomic_num_list, path):
    """
    Find top k optimized molecules for multiple properties.

    Args:
        model (object): A PyTorch model for optimizing molecules.
        device (str): The device to use for optimization.
        data_name (str): The name of the dataset.
        property_name (str): The name of the property score to optimize for.
        prop_to_idx (dict): A dictionary that maps property names to their corresponding column index in the dataset.
        train_prop (list): A list of tuples containing the molecular properties of the training dataset.
        topk (int): The number of optimized molecules to return.
        atomic_num_list (list): A list of atomic numbers to consider for the optimized molecules.
        path (str): The path to save the optimized molecules.

    Returns:
        Saves a list of tuples containing the optimized molecules, their corresponding property scores,
        their similarity to the reference molecule, the reference molecule itself, the change in each property,
        and the combined improvement to a csv at the given path

    """
    start_time = time.time()
    print('Finding top {} score'.format(property_name))
    train_list = list(zip(*train_prop))
    qed_scaler = StandardScaler()
    plogp_scaler = StandardScaler()
    qed_list = np.array(train_list[prop_to_idx['qed']]).reshape(-1,1)
    plogp_list = np.array(train_list[prop_to_idx['plogp']]).reshape(-1,1)
    qed_scaler.fit(qed_list)
    plogp_scaler.fit(plogp_list)
    print(qed_scaler.mean_)
    print(plogp_scaler.mean_)
    qed_scaled = qed_scaler.transform(qed_list).reshape(-1)
    plogp_scaled = plogp_scaler.transform(plogp_list).reshape(-1)
    train_prop_extended = []
    for i, smi in enumerate(train_prop):
        scaled_prop = (qed_scaled[i]+plogp_scaled[i],)
        new_tup = smi + scaled_prop
        train_prop_extended.append(new_tup)
    print('Finding top qed_plogp score')
    train_prop_sorted = sorted(train_prop_extended, key=lambda  tup: tup[-1], reverse=True)  

    if not os.path.exists('./'+data_name+'_chemspace_opt/'+f'{path}'):	
        os.makedirs('./'+data_name+'_chemspace_opt/'+f'{path}')

    directions = []
    props = ['qed', 'plogp']
    for prop_name in props:
        direction = np.load('./boundaries_'+args.data_name+'/boundary_'+prop_name+'.npy')
        directions.append(direction)

    result_list = []
    for i, r in enumerate(train_prop_sorted):
        if i >= topk:
            break
        if i % 50 == 0:
            print('Optimization {}/{}, time: {:.2f} seconds'.format(i, topk, time.time() - start_time))
        smile = r[prop_to_idx['smile']]
        qed = r[prop_to_idx['qed']]
        plogp = r[prop_to_idx['plogp']]
        _ , results1, results2, _ , _ = optimize_mol_multi_prop(model, smile, directions, num_range=100, path_len=21, sim_cutoff=0.0,
                                    data_name=data_name, atomic_num_list=atomic_num_list, device=device)

        qed_results = np.array(list(zip(*results1))[1]).reshape(-1,1)
        plogp_results = np.array(list(zip(*results2))[1]).reshape(-1,1)
        improvement_qed = qed_scaler.transform(qed_results).reshape(-1)
        improvement_plogp = plogp_scaler.transform(plogp_results).reshape(-1)
        improvement_combined = improvement_qed + improvement_plogp
        results_1_sorted = [r for _,r in sorted(zip(improvement_combined, results1), reverse=True)]
        results_2_sorted = [r for _,r in sorted(zip(improvement_combined, results2), reverse=True)]
    
        improvement_combined = np.sort(improvement_combined)[::-1]
        for r1, r2, imp in zip(results_1_sorted, results_2_sorted, improvement_combined):
            smile1, property1, sim, _ = r1
            smile2, property2, _ , _ = r2
            qed_delta = property1 - qed
            plogp_delta = property2 - plogp
            result_list.append((smile1, smile, sim, property1, property2, qed, plogp, qed_delta, plogp_delta, imp))
      
    result_list.sort(key=lambda tup: tup[-1], reverse=True)

    # check novelty
    train_smile = set()
    for i, r in enumerate(train_prop_sorted):
        qed, plogp, smile, _ = r
        train_smile.add(smile)
        mol = Chem.MolFromSmiles(smile)
        smile2 = Chem.MolToSmiles(mol, isomericSmiles=True)
        train_smile.add(smile2)

    result_list_novel = []
    for i, r in enumerate(result_list):
        smile = r[0]
        if smile not in train_smile:
            result_list_novel.append(r)

    # dump results
    f = open('./'+data_name+'_chemspace_opt/'+f'{path}' + '_discovered_sorted.csv', "w")
    for r in result_list_novel:
        smile, smile_original, sim, qed, plogp, qed_original, plogp_original, qed_delta, plogp_delta, imp = r
        f.write('{},{},{},{},{},{},{},{},{},{}\n'.format(imp, qed, plogp, smile, smile_original, sim, qed_original, plogp_original, qed_delta, plogp_delta))
        f.flush()
    f.close()
    print('Dump done!')

def constrain_optimization_smiles(model, device, data_name, property_name, prop_to_idx, train_prop, topk,
                                  atomic_num_list, path, path_range=100, sim_cutoff=0.0):
    """
    Optimize molecules for a given property, while maintaining the desired similarity (Tanimoto) to 
    the given reference molecule.

    Args:
        model (object): A PyTorch model for optimizing molecules.
        device (str): The device to use for optimization.
        data_name (str): The name of the dataset.
        property_name (str): The name of the property score to optimize for.
        prop_to_idx (dict): A dictionary that maps property names to their corresponding column index in the dataset.
        train_prop (list): A list of tuples containing the molecular properties of the training dataset.
        topk (int): The number of optimized molecules to return.
        atomic_num_list (list): A list of atomic numbers to consider for the optimized molecules.
        path (str): The path to save the optimized molecules.
        path_range (int, optional): The maximum number of steps that can be taken in each direction during optimization.
            Defaults to 100.
        sim_cutoff (float, optional): The similarity cutoff used to filter out similar molecules during optimization.
            Defaults to 0.0.

    Returns:
        Saves a list of tuples containing the optimized molecules, their corresponding property score,
        their similarity to the reference molecule, the reference molecule, the original property score, 
        and the change in property to a csv at the given path

    """
    start_time = time.time()
    idx = prop_to_idx[property_name] # smile,qed,plogp,MolLogP,MolWt,sa,drd2,jnk3,gsk3b 

    print('Constrained optimization of {} score'.format(property_name))
    train_prop_sorted = sorted(train_prop, key=lambda tup: tup[idx]) #, reverse=True)  # qed, plogp, smile
    result_list = []
    nfail = 0
    if not os.path.exists('./'+data_name+'_chemspace_consopt/'+f'{path}'):
        os.makedirs('./'+data_name+'_chemspace_consopt/'+f'{path}')
    direction = np.load('./boundaries_'+data_name+'/boundary_'+property_name+'.npy')
    for i, r in enumerate(train_prop_sorted):
        if i >= topk:
            break
        if i % 50 == 0:
            print('Optimization {}/{}, time: {:.2f} seconds'.format(i, topk, time.time() - start_time))
        smile = r[prop_to_idx['smile']]
        prop = r[idx]

        _ , results, _ = optimize_mol(model, smile, direction, num_range=path_range, path_len=21, sim_cutoff=sim_cutoff,
                                    data_name=data_name, atomic_num_list=atomic_num_list,
                                    property_name=property_name, device=device)
        # for idx, smi_save in enumerate(smiles_path):
        #     np.save(open(filepath+'_'+str(i)+'_'+str(idx)+'.npy','wb'),smi_save)
        if len(results) > 0:
            smile2, property2, sim, _ = results[0]
            prop_delta = property2 - prop
            if prop_delta >= 0:
                result_list.append((smile2, property2, sim, smile, prop, prop_delta))
            else:
                nfail += 1
                print('Failure:{}:{}'.format(i, smile))
        else:
            nfail += 1
            print('Failure:{}:{}'.format(i, smile))

    print(result_list)
    df = pd.DataFrame(result_list,
                      columns=['smile_new', 'prop_new', 'sim', 'smile_old', 'prop_old', 'prop_delta'])

    print(df.describe())
    df.to_csv('./'+data_name+'_chemspace_consopt/'+f'{path}'+'_constrain_optimization.csv', index=False)
    print('Dump done!')
    print('nfail:{} in total:{}'.format(nfail, topk))
    print('success rate: {}'.format((topk-nfail)*1.0/topk))

def constrain_optimization_smiles_multi_prop(model, device, data_name, property_name, prop_to_idx, train_prop, topk,
                                  atomic_num_list, path, path_range=100, sim_cutoff=0.0):
    """
    Optimize molecules for multiple properties, while maintaining the desired similarity (Tanimoto) to 
    the given reference molecule.

    Args:
        model (object): A PyTorch model for optimizing molecules.
        device (str): The device to use for optimization.
        data_name (str): The name of the dataset.
        property_name (str): The name of the property score to optimize for.
        prop_to_idx (dict): A dictionary that maps property names to their corresponding column index in the dataset.
        train_prop (list): A list of tuples containing the molecular properties of the training dataset.
        topk (int): The number of optimized molecules to return.
        atomic_num_list (list): A list of atomic numbers to consider for the optimized molecules.
        path (str): The path to save the optimized molecules.
        path_range (int, optional): The maximum number of steps that can be taken in each direction during optimization.
            Defaults to 100.
        sim_cutoff (float, optional): The similarity cutoff used to filter out similar molecules during optimization.
            Defaults to 0.0.

    Returns:
        Saves a list of tuples containing the optimized molecules, their corresponding property scores,
        their similarity to the reference molecule, the reference molecule, the original property scores, 
        and the change in properties to a csv at the given path

    """
    start_time = time.time()
    
    train_list = list(zip(*train_prop))
    qed_scaled = scale(train_list[prop_to_idx['qed']])
    plogp_scaled = scale(train_list[prop_to_idx['plogp']])
    train_prop_extended = []
    for i, smi in enumerate(train_prop):
        scaled_prop = (qed_scaled[i]+plogp_scaled[i],)
        new_tup = smi + scaled_prop
        train_prop_extended.append(new_tup)

    print('Constrained optimization of qed_plogp score'.format(property_name))

    train_prop_sorted = sorted(train_prop_extended, key=lambda  tup: tup[-1]) #, reverse=True)  # qed, plogp, smile
    result_list = []
    nfail = 0
    count = 0
    filepath = './'+data_name+'_chemspace_consopt/'+f'{path}'+'/smiles'
    if not os.path.exists('./'+data_name+'_chemspace_consopt/'+f'{path}'):
        os.makedirs('./'+data_name+'_chemspace_consopt/'+f'{path}')

    directions = []
    props = ['qed', 'plogp']
    for prop_name in props:
        direction = np.load('./boundaries_'+args.data_name+'/boundary_'+prop_name+'.npy')
        directions.append(direction)
   
    for i, r in enumerate(train_prop_sorted):
        if i >= topk:
            break
        if i % 50 == 0:
            print('Optimization {}/{}, time: {:.2f} seconds'.format(i, topk, time.time() - start_time))
        smile = r[prop_to_idx['smile']]
        qed = r[prop_to_idx['qed']]
        plogp = r[prop_to_idx['plogp']]
        smiles_path, results_1, results_2, _, _ = optimize_mol_multi_prop(model, smile, directions, num_range=path_range, path_len=21, sim_cutoff=sim_cutoff,
                                    data_name=data_name, atomic_num_list=atomic_num_list, device=device)

        for idx, smi_save in enumerate(smiles_path):
            np.save(open(filepath+'_'+str(i)+'_'+str(idx)+'.npy','wb'),smi_save)

        if len(results_1) > 0 and len(results_2) > 0:
            improvement_1 = [(result[1] - qed)/qed for result in results_1]
            improvement_2 = [(result[1] - plogp)/plogp for result in results_2]
            improvement_combined = improvement_1 + improvement_2
            results_1_sorted = [r for _,r in sorted(zip(improvement_combined, results_1), reverse=True)]
            results_2_sorted = [r for _,r in sorted(zip(improvement_combined, results_2), reverse=True)]
            smile1, property1, sim, _ = results_1_sorted[0]
            smile2, property2, _ , _ = results_2_sorted[0]
            qed_delta = property1 - qed
            plogp_delta = property2 - plogp
            if qed_delta >=0 and plogp_delta >= 0:
                result_list.append((smile1, smile, sim, property1, property2, qed, plogp, qed_delta, plogp_delta))
            else:
                nfail += 1
                print('Failure:{}:{}'.format(i, smile))
        else:
            count += 1
            nfail += 1
            print('Failure:{}:{}'.format(i, smile))

    df = pd.DataFrame(result_list,
                      columns=['smile_new', 'smile_old', 'sim', 'qed_new', 'plogp_new', 'qed_old', 'plogp_old', 'qed_delta', 'plogp_delta'])

    print(df.describe())
    df.to_csv('./'+data_name+'_chemspace_consopt/'+f'{path}'+'_constrain_optimization_multi.csv', index=False)
    print('Dump done!')
    print('nfail:{} in total:{}'.format(nfail, topk))
    print('success rate: {}'.format((topk-nfail)*1.0/topk))
    print(count)

def plot_top_qed_mol():
    """ 
        Plot the top 25 molecules sorted by property score.
    """
    import cairosvg
    filename = 'qed_discovered_sorted.csv'
    df = pd.read_csv(filename, header=None, names=['qed', 'Smile', 'Similarity', 'Smile Original'])
    vmol = []
    vlabel = []
    for index, row in df.head(n=25).iterrows():
        score, smile, sim, smile_old = row
        print(score)
        vmol.append(Chem.MolFromSmiles(smile))
        vlabel.append('{:.3f}'.format(score))

    svg = Draw.MolsToGridImage(vmol, legends=vlabel, molsPerRow=5, #5,
                               subImgSize=(120, 120), useSVG=True)  # , useSVG=True

    cairosvg.svg2pdf(bytestring=svg.encode('utf-8'), write_to="top_qed_moflow.pdf")
    cairosvg.svg2png(bytestring=svg.encode('utf-8'), write_to="top_qed_moflow.png")
    # print('Dump {}.png/pdf done'.format(filepath))

    img = Draw.MolsToGridImage(vmol, legends=vlabel, molsPerRow=5,
                               subImgSize=(300, 300), useSVG=True)
    # print(img)


def plot_mol_constraint_opt():
    """
        Plot the molecular structures of two given SMILES strings along with their corresponding property values.
    """
    import cairosvg
    vsmiles = ['O=C(NCc1ccc2c3c(cccc13)C(=O)N2)c1ccc(F)cc1',
               'O=C(NCC1=Cc2c[nH]c(=O)c3cccc1c23)c1ccc(F)cc1']
    vmol = [Chem.MolFromSmiles(s) for s in vsmiles]
    vplogp = ['{:.2f}'.format(env.penalized_logp(mol)) for mol in vmol]

    # vhighlight = [vmol[0].GetSubstructMatch(Chem.MolFromSmiles('C2=C1C=CC=C3C1=C(C=C2)NC3')),
    #               vmol[1].GetSubstructMatch(Chem.MolFromSmiles('C4=CC6=C5C4=CC=CC5=C[N](=C6)[H]'))]
    svg = Draw.MolsToGridImage(vmol, legends=vplogp, molsPerRow=2,
                               subImgSize=(250, 100), useSVG=True)
                               #highlightAtoms=vhighlight)  # , useSVG=True

    cairosvg.svg2pdf(bytestring=svg.encode('utf-8'), write_to="copt2.pdf")
    cairosvg.svg2png(bytestring=svg.encode('utf-8'), write_to="copt2.png")


def plot_mol_matrix():
    """
        Plot a matrix representation of the molecular structure and save the images as PDF and PNG files.
    """
    import cairosvg
    import seaborn as sns
    import matplotlib.pyplot as plt
    smiles = 'CN(C)C(=N)NC(=N)N'  #'CC(C)NC1=CC=CO1'  #'CC1=C(SC(=C1)C(=O)NCC2=NOC=C2)Br'
    bond, atoms = smiles_to_adj(smiles, 'qm9')
    bond = bond[0]
    atoms = atoms[0]

    # def save_mol_png(mol, filepath, size=(100, 100)):
    #     Draw.MolToFile(mol, filepath, size=size)

    Draw.MolToImageFile(Chem.MolFromSmiles(smiles), 'mol.pdf')
    # save_mol_png(Chem.MolFromSmiles(smiles), 'mol.png')
    svg = Draw.MolsToGridImage([Chem.MolFromSmiles(smiles)], legends=[], molsPerRow=1,
                               subImgSize=(250, 250), useSVG=True)
    # highlightAtoms=vhighlight)  # , useSVG=True

    cairosvg.svg2pdf(bytestring=svg.encode('utf-8'), write_to="mol.pdf")
    cairosvg.svg2png(bytestring=svg.encode('utf-8'), write_to="mol.png")

    # sns.set()
    # ax = sns.heatmap(1-atoms)
    # with sns.axes_style("white"):
    fig, ax = plt.subplots(figsize=(2, 3.4))
    # sns.palplot(sns.diverging_palette(240, 10, n=9))
    ax = sns.heatmap(atoms, linewidths=.5, ax=ax, annot_kws={"size": 18}, cbar=False,
                     xticklabels=False, yticklabels=False, square=True, cmap="vlag", vmin=-1, vmax=1, linecolor='black')
    # ,cmap=sns.diverging_palette(240, 10, n=9)) #"YlGnBu"  , square=True

    plt.show()
    fig.savefig('atom.pdf')
    fig.savefig('atom.png')


    for i, x in enumerate(bond):
        fig, ax = plt.subplots(figsize=(5, 5))
        # sns.palplot(sns.diverging_palette(240, 10, n=9))
        ax = sns.heatmap(x, linewidths=.5,  ax=ax,   annot_kws={"size": 18}, cbar=False,
                         xticklabels=False, yticklabels=False, square=True, cmap="vlag", vmin=-1, vmax=1, linecolor='black')
                              # ,cmap=sns.diverging_palette(240, 10, n=9)) #"YlGnBu"  , square=True

        plt.show()
        fig.savefig('bond{}.pdf'.format(i))
        fig.savefig('bond{}.png'.format(i))


if __name__ == '__main__':
    # plot_mol()
    # plot_mol_constraint_opt()
    # plot_mol_matrix()
    # plot_top_qed_mol()
    # exit(-1)
    start = time.time()
    print("Start at Time: {}".format(time.ctime()))

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='data')
    parser.add_argument('--data_name', type=str, choices=['qm9', 'zinc250k'], required=True,
                        help='dataset name')
    parser.add_argument("--snapshot_path", "-snapshot", type=str, default='model_snapshot_epoch_200')
    parser.add_argument("--hyperparams_path", type=str, default='moflow-params.json')
    parser.add_argument("--property_model_path", type=str, default=None)
    parser.add_argument("--save_path", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument('-l', '--learning_rate', type=float, default=0.001, help='Base learning rate')
    parser.add_argument('-e', '--lr_decay', type=float, default=0.999995,
                        help='Learning rate decay, applied every step of the optimization')
    parser.add_argument('-w', '--weight_decay', type=float, default=1e-5,
                        help='L2 norm for the parameters')
    parser.add_argument('--hidden', type=str, default="",
                        help='Hidden dimension list for output regression')
    parser.add_argument('-x', '--max_epochs', type=int, default=5, help='How many epochs to run in total?')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='GPU Id to use')

    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--img_format", type=str, default='svg')
    parser.add_argument('--multi_property', action='store_true', default=False, help='To run optimization/constrained optimization with multiple properties')
    parser.add_argument('--property_name', type=str, default='plogp', choices=['qed', 'plogp', 'drd2', 'gsk3b', 'jnk3', 'qed_plogp'])
    parser.add_argument('--additive_transformations', type=strtobool, default=False,
                        help='apply only additive coupling layers')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='temperature of the gaussian distributions')

    parser.add_argument('--topk', type=int, default=800, help='Top k smiles as seeds')
    parser.add_argument('--path_range', type=int, default=100, help='Range of manipulation')
    parser.add_argument('--debug', type=strtobool, default='true', help='To run optimization with more information')

    parser.add_argument("--sim_cutoff", type=float, default=0.00)
    parser.add_argument('--topscore', action='store_true', default=False, help='To find top score')
    parser.add_argument('--consopt', action='store_true', default=False, help='To do constrained optimization')

    args = parser.parse_args()

    # Device configuration
    device = -1
    if args.gpu >= 0:
        # device = args.gpu
        device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    property_name = args.property_name.lower()

    if args.data_name == 'qm9':
        qm9_model = 'models/results/qm9_64gnn_128-64lin_1-1mask_0d6noise_convlu1'
        model_dir = os.path.join(os.getcwd(), qm9_model)
    elif args.data_name == 'zinc250k':
       zinc_model = 'models/results/zinc250k_512t2cnn_256gnn_512-64lin_10flow_19fold_convlu2_38af-1-1mask'
       model_dir = os.path.join(os.getcwd(), zinc_model)
    
    snapshot_path = os.path.join(model_dir, args.snapshot_path)
    hyperparams_path = os.path.join(model_dir, args.hyperparams_path)
    model_params = Hyperparameters(path=hyperparams_path)
    model = load_model(snapshot_path, model_params, debug=True)  # Load moflow model

    if args.hidden in ('', ','):
        hidden = []
    else:
        hidden = [int(d) for d in args.hidden.strip(',').split(',')]
    print('Hidden dim for output regression: ', hidden)

    if args.data_name == 'qm9':
        atomic_num_list = [6, 7, 8, 9, 0]
        transform_fn = transform_qm9.transform_fn
        valid_idx = transform_qm9.get_val_ids()
        molecule_file = 'qm9_relgcn_kekulized_ggnp.npz'
    elif args.data_name == 'zinc250k':
        atomic_num_list = zinc250_atomic_num_list
        transform_fn = transform_zinc250k.transform_fn_zinc250k
        valid_idx = transform_zinc250k.get_val_ids()
        molecule_file = 'zinc250k_relgcn_kekulized_ggnp.npz'
    else:
        raise ValueError("Wrong data_name{}".format(args.data_name))

    # dataset = NumpyTupleDataset(os.path.join(args.data_dir, molecule_file), transform=transform_fn)  # 133885
    dataset = NumpyTupleDataset.load(os.path.join(args.data_dir, molecule_file), transform=transform_fn)

    print('Load {} done, length: {}'.format(os.path.join(args.data_dir, molecule_file), len(dataset)))
    assert len(valid_idx) > 0
    train_idx = [t for t in range(len(dataset)) if t not in valid_idx]  # 224568 = 249455 - 24887
    n_train = len(train_idx)  # 120803 zinc: 224568
    train = torch.utils.data.Subset(dataset, train_idx)  # 120803
    test = torch.utils.data.Subset(dataset, valid_idx)  # 13082  not used for generation

    train_dataloader = torch.utils.data.DataLoader(train, batch_size=args.batch_size)

    # print("loading hyperparamaters from {}".format(hyperparams_path))
    prop_list, prop_to_idx = load_property_csv(args.data_name, normalize=False)
    train_prop = [prop_list[i] for i in train_idx]
    test_prop = [prop_list[i] for i in valid_idx]
    print('Prepare data done! Time {:.2f} seconds'.format(time.time() - start))

    model.to(device)
    model.eval()

    if args.topscore:
        print('Finding top score:')
        if not args.multi_property:
            find_top_score_smiles(model, device, args.data_name, property_name, prop_to_idx, train_prop, args.topk, atomic_num_list, args.save_path)
        else:
            find_top_score_smiles_multi_prop(model, device, args.data_name, property_name, prop_to_idx, train_prop, args.topk, atomic_num_list, args.save_path)

    if args.consopt:
        print('Constrained optimization:')
        if not args.multi_property:
            constrain_optimization_smiles(model, device, args.data_name, property_name, prop_to_idx, train_prop, args.topk,   
                                    atomic_num_list, args.save_path, path_range=args.path_range, sim_cutoff=args.sim_cutoff)
        else:
            constrain_optimization_smiles_multi_prop(model, device, args.data_name, property_name, prop_to_idx, train_prop, args.topk,   
                                    atomic_num_list, args.save_path, path_range=args.path_range, sim_cutoff=args.sim_cutoff)

    print('Total Time {:.2f} seconds'.format(time.time() - start))