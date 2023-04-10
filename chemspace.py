import argparse
import os
import sys
sys.path.insert(0,'..')
from distutils.util import strtobool
import torch
import numpy as np
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger

import matplotlib.pyplot as plt 

plt.switch_backend('agg')
plt.rcParams.update({'font.size': 30})

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

from tqdm import tqdm

from tdc import Oracle

import matplotlib.pyplot as plt

from rdkit.Chem import Descriptors

from data import transform_qm9, transform_zinc250k
from data.transform_zinc250k import zinc250_atomic_num_list
from mflow.models.hyperparams import Hyperparameters
from mflow.models.utils import check_validity, adj_to_smiles, check_novelty, _to_numpy_array
from mflow.utils.model_utils import load_model
from mflow.models.model import rescale_adj
import mflow.utils.environment as env

# from IPython.display import SVG, display
from data.data_loader import NumpyTupleDataset

from mflow.utils.model_utils import smiles_to_adj 

import time
import functools
print = functools.partial(print, flush=True)

# Define helper oracle functions that take in a RDKit Mol object and return the specfied molecular property
def check_SA(mol):
    scorer = Oracle(name = 'SA')
    score = scorer(Chem.MolToSmiles(mol))
    return score 

def check_DRD2(mol):
    scorer = Oracle(name = 'DRD2')
    score = scorer(Chem.MolToSmiles(mol))
    return score

def check_JNK3(mol):
    scorer = Oracle(name = 'JNK3')
    score = scorer(Chem.MolToSmiles(mol))
    return score

def check_GSK3B(mol):
    scorer = Oracle(name = 'GSK3B')
    score = scorer(Chem.MolToSmiles(mol))
    return score

def check_plogp(mol):
    plogp = env.penalized_logp(mol)
    return plogp

def cache_prop_pred():
    """
    Return dictionary of oracle functions for molecular properties

    Returns:
        dict: A dictionary containing the property name as keys and their respective oracle functions as values
    """
    prop_pred = {}
    for prop_name, function in Descriptors.descList:
        prop_pred[prop_name] = function
    prop_pred['sa'] = check_SA
    prop_pred['drd2'] = check_DRD2
    prop_pred['jnk3'] = check_JNK3 
    prop_pred['gsk3b'] = check_GSK3B
    prop_pred['plogp'] = check_plogp
    return prop_pred

def get_z(model, mols, device):
    """
    Get latent vectors for molecules
    
    Args:
        model (Moflow model): The Moflow model
        mols (list of str): The list of SMILES strings of molecules 
        device (torch.device): The torch device

    Returns:
        numpy.ndarray: The latent vectors for molecules
    """
    z = []
    for mol in mols:
        adj_idx, x_idx = smiles_to_adj(mol, data_name=args.data_name)
        if device:
            adj_idx = adj_idx.to(device)
            x_idx = x_idx.to(device)
            adj_normalized = rescale_adj(adj_idx).to(device)
        z_idx, _ = model(adj_idx, x_idx, adj_normalized)
        z_idx[0] = z_idx[0].reshape(z_idx[0].shape[0], -1)
        z_idx[1] = z_idx[1].reshape(z_idx[1].shape[0], -1)
        z_idx = torch.cat((z_idx[0], z_idx[1]), dim=1).squeeze(dim=0)  # h:(1,45), adj:(1,324) -> (1, 369) -> (369,)
        z_idx = np.expand_dims(_to_numpy_array(z_idx), axis=0)
        z.append(z_idx)
    z = np.concatenate(z, axis=0)
    return z 

def generate_mols(model, temp=0.7, z_mu=None, batch_size=20, true_adj=None, device=-1):  #  gpu=-1):
    """
    Generates molecules using a trained Moflow model.

    Args:
        model (Moflow): The Moflow model used for generating molecules.
        temp (float): Temperature for sampling.
        z_mu (numpy.ndarray): Latent vector of a molecule.
        batch_size (int): Batch size for generating molecules.
        true_adj (numpy.ndarray): True adjacency matrix.
        device (torch.device or int): The device to use for generating molecules.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: The adjacency matrix,
        feature matrix, and latent vector of the generated molecule.

    Raises:
        ValueError: If device is not a valid type.

    """
    if isinstance(device, torch.device):
        pass
    elif isinstance(device, int):
        if device >= 0:
            # device = args.gpu
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu', int(device))
        else:
            device = torch.device('cpu')
    else:
        raise ValueError("only 'torch.device' or 'int' are valid for 'device', but '%s' is "'given' % str(device))

    z_dim = model.b_size + model.a_size  # 324 + 45 = 369   9*9*4 + 9 * 5
    mu = np.zeros(z_dim)  # (369,) default , dtype=np.float64
    sigma_diag = np.ones(z_dim)  # (369,)

    if model.hyper_params.learn_dist:
        if len(model.ln_var) == 1:
            sigma_diag = np.sqrt(np.exp(model.ln_var.item())) * sigma_diag
        elif len(model.ln_var) == 2:
            sigma_diag[:model.b_size] = np.sqrt(np.exp(model.ln_var[0].item())) * sigma_diag[:model.b_size]
            sigma_diag[model.b_size+1:] = np.sqrt(np.exp(model.ln_var[1].item())) * sigma_diag[model.b_size+1:]

    sigma = temp * sigma_diag

    with torch.no_grad():
        if z_mu is not None:
            mu = z_mu
            sigma = 0.01 * np.eye(z_dim)
        # mu: (369,), sigma: (369,), batch_size: 100, z_dim: 369
        z = np.random.normal(mu, sigma, (batch_size, z_dim))  # .astype(np.float32)
        z = torch.from_numpy(z).float().to(device)
        adj, x = model.reverse(z, true_adj=true_adj)

    return adj, x, z  # (bs, 4, 9, 9), (bs, 9, 5), (bs, 369)

def generate_mols_dis(model, z, temp=0.7, z_mu=None, batch_size=20, true_adj=None, device=-1):  #  gpu=-1):
    """
    Generates molecules using a trained Moflow model and a given latent vector.

    Args:
        model (Moflow): The Moflow model used for generating molecules.
        z (numpy.ndarray): The latent vector used for generating molecules.
        temp (float): Temperature for sampling.
        z_mu (numpy.ndarray): Latent vector of a molecule.
        batch_size (int): Batch size for generating molecules.
        true_adj (numpy.ndarray): True adjacency matrix.
        device (torch.device or int): The device to use for generating molecules.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: The adjacency matrix,
        feature matrix, and latent vector of the generated molecule.

    Raises:
        ValueError: If device is not a valid type.

    """
    if isinstance(device, torch.device):
        pass
    elif isinstance(device, int):
        if device >= 0:
            # device = args.gpu
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu', int(device))
        else:
            device = torch.device('cpu')
    else:
        raise ValueError("only 'torch.device' or 'int' are valid for 'device', but '%s' is "'given' % str(device))

    z_dim = model.b_size + model.a_size  # 324 + 45 = 369   9*9*4 + 9 * 5
    mu = np.zeros(z_dim)  # (369,) default , dtype=np.float64
    sigma_diag = np.ones(z_dim)  # (369,)

    if model.hyper_params.learn_dist:
        if len(model.ln_var) == 1:
            sigma_diag = np.sqrt(np.exp(model.ln_var.item())) * sigma_diag
        elif len(model.ln_var) == 2:
            sigma_diag[:model.b_size] = np.sqrt(np.exp(model.ln_var[0].item())) * sigma_diag[:model.b_size]
            sigma_diag[model.b_size+1:] = np.sqrt(np.exp(model.ln_var[1].item())) * sigma_diag[model.b_size+1:]

    sigma = temp * sigma_diag

    with torch.no_grad():
        if z_mu is not None:
            mu = z_mu
            sigma = 0.01 * np.eye(z_dim)
        # mu: (369,), sigma: (369,), batch_size: 100, z_dim: 369
        z = torch.from_numpy(z).float().to(device)
        adj, x = model.reverse(z, true_adj=true_adj)

    return adj, x, z  # (bs, 4, 9, 9), (bs, 9, 5), (bs, 369)

def traverse(filepath, model, data, direction, num_range, path_len=11,
                             atomic_num_list=[6, 7, 8, 9, 0],
                            device=None):
    """
    Traverse a given latent space direction and save molecules.

    Args:
        filepath (str): Path to the output file.
        model (Moflow): The Moflow model used for generating molecules.
        data (numpy.ndarray): The latent vector used for generating molecules.
        direction (numpy.ndarray): The direction to traverse the latent space.
        num_range (Tuple[float, float]): The range of values to traverse.
        path_len (int): The number of points to traverse along the direction.
        atomic_num_list (List[int]): List of atomic numbers to use for the generated molecules.
        device (torch.device or int): The device to use for generating molecules.

    Returns:
        None

    """
    with torch.no_grad():
        z = data

        distances = np.linspace(-num_range,num_range,path_len)
        distances = distances.tolist()

        for i in tqdm(range(z.shape[0])):
            with torch.no_grad():
                z0 = z[i]
                z_to_decode = []
                for j in range(len(distances)):
                    z_to_decode.append(z0+distances[j]*direction)
                z_to_decode = torch.from_numpy(np.array(z_to_decode)).squeeze().float().to(device)
                adj, x = model.reverse(z_to_decode)
                smile0 = adj_to_smiles(adj.cpu(), x.cpu(), atomic_num_list)
                for idx, smi_save in enumerate(smile0):
                    np.save(open(filepath+str(i)+'_'+str(idx)+'.npy','wb'),smi_save)

        return  

def traverse_multi_prop(filepath, model, data, directions, num_range, path_len=11,
                             atomic_num_list=[6, 7, 8, 9, 0],
                            device=None):
    """
    Traverse a latent space in the two given directions and save molecules.

    Args:
        filepath (str): Path to the output file.
        model (Moflow): The Moflow model used for generating molecules.
        data (numpy.ndarray): The latent vector used for generating molecules.
        directions (List[List[float]]): A list of two 1D float lists of length d, representing
            the two property directions to traverse.
        num_range (Tuple[float, float]): The range of values to traverse.
        path_len (int): The number of points to traverse along the direction.
        atomic_num_list (List[int]): List of atomic numbers to use for the generated molecules.
        device (torch.device or int): The device to use for generating molecules.

    Returns:
        None

    """

    with torch.no_grad():
        z = data

        distances = np.linspace(-num_range,num_range,path_len)
        distances = distances.tolist()

        if len(directions) > 1:
            combined_directions = directions[0] * directions[1]
            pos_attributes = (combined_directions >= 0) * directions[1]
            direction = directions[0] + pos_attributes
        else:
            direction = directions[0]
        for i in tqdm(range(z.shape[0])):
            with torch.no_grad():
                z0 = z[i]
                z_to_decode = []
                for j in range(len(distances)):
                    z_to_decode.append(z0+distances[j]*direction)
                z_to_decode = torch.from_numpy(np.array(z_to_decode)).squeeze().float().to(device)
                adj, x = model.reverse(z_to_decode)
                smile0 = adj_to_smiles(adj.cpu(), x.cpu(), atomic_num_list)
                for idx, smi_save in enumerate(smile0):
                    np.save(open(filepath+str(i)+'_'+str(idx)+'.npy','wb'),smi_save)

        return  

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='data')
    parser.add_argument('--data_name', type=str, choices=['qm9', 'zinc250k'], required=True,
                        help='dataset name')
    parser.add_argument("--snapshot_path", "-snapshot", type=str, default='model_snapshot_epoch_200')
    parser.add_argument("--hyperparams_path", type=str, default='moflow-params.json')
    
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument('--additive_transformations', type=strtobool, default='false',
                        help='apply only additive coupling layers')
    parser.add_argument('--delta', type=float, default=0.1)
    parser.add_argument('--n_experiments', type=int, default=1, help='number of times generation to be run')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='temperature of the gaussian distribution')
    parser.add_argument('--path_len', type=int, default=21)
    parser.add_argument('--baseline', type=str, default='chemspace')
    parser.add_argument('--multi_property', action='store_true', default=False)

    parser.add_argument('--save_fig', type=strtobool, default='true')
    parser.add_argument('--save_score', type=strtobool, default='true')
    parser.add_argument('--random', action='store_true', default=False)
    parser.add_argument('--traverse', action='store_true', default=False)
    parser.add_argument('--disent', action='store_true', default=False)
    parser.add_argument('--largest', action='store_true', default=False)
    parser.add_argument('--num_range', type=int, default=1)
    parser.add_argument('-b', '--boundary_path', type=str, required=False,
                      help='Path to the semantic boundary. (required)')

    parser.add_argument('--inter_times', type=int, default=5)

    parser.add_argument('--correct_validity', type=strtobool, default='true',
                        help='if apply validity correction after the generation')
    args = parser.parse_args()

    start = time.time()
    print("Start at Time: {}".format(time.ctime()))

    if args.data_name == 'qm9':
        qm9_model = 'models/results/qm9_64gnn_128-64lin_1-1mask_0d6noise_convlu1'
        model_dir = os.path.join(os.getcwd(), qm9_model)
    elif args.data_name == 'zinc250k':
        zinc_model = 'models/results/zinc250k_512t2cnn_256gnn_512-64lin_10flow_19fold_convlu2_38af-1-1mask'
        model_dir = os.path.join(os.getcwd(), zinc_model)
    
    snapshot_path = os.path.join(model_dir, args.snapshot_path)
    hyperparams_path = os.path.join(model_dir, args.hyperparams_path)
    print("loading hyperparamaters from {}".format(hyperparams_path))
    model_params = Hyperparameters(path=hyperparams_path)
    model = load_model(snapshot_path, model_params, debug=True)
    if len(model.ln_var) == 1:
        print('model.ln_var: {:.2f}'.format(model.ln_var.item()))
    elif len(model.ln_var) == 2:
        print('model.ln_var[0]: {:.2f}, model.ln_var[1]: {:.2f}'.format(model.ln_var[0].item(), model.ln_var[1].item()))

    if args.gpu >= 0:
        # device = args.gpu
        device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    model.to(device)
    model.eval()  # Set model for evaluation

    prop_pred = cache_prop_pred()

    if args.data_name == 'qm9':
        atomic_num_list = [6, 7, 8, 9, 0]
        transform_fn = transform_qm9.transform_fn
        # true_data = TransformDataset(true_data, transform_qm9.transform_fn)
        valid_idx = transform_qm9.get_val_ids()
        pair_idx = transform_qm9.get_pair_ids()
        molecule_file = 'qm9_relgcn_kekulized_ggnp.npz'
    elif args.data_name == 'zinc250k':
        atomic_num_list = zinc250_atomic_num_list
        # transform_fn = transform_qm9.transform_fn
        transform_fn = transform_zinc250k.transform_fn_zinc250k
        # true_data = TransformDataset(true_data, transform_fn_zinc250k)
        valid_idx = transform_zinc250k.get_val_ids()
        pair_idx = transform_zinc250k.get_pair_ids()
        molecule_file = 'zinc250k_relgcn_kekulized_ggnp.npz'

    batch_size = args.batch_size
    dataset = NumpyTupleDataset.load(os.path.join(args.data_dir, molecule_file), transform=transform_fn)

    assert len(valid_idx) > 0
    train_idx = [t for t in range(len(dataset)) if t not in valid_idx]  # 120803 = 133885-13082
    n_train = len(train_idx)  # 120803
    train = torch.utils.data.Subset(dataset, train_idx)  # 120803
    test = torch.utils.data.Subset(dataset, valid_idx)  # 13082  not used for generation

    print('{} in total, {}  training data, {}  testing data, {} batchsize, train/batchsize {}'.format(
        len(dataset),
        len(train),
        len(test),
        batch_size,
        len(train)/batch_size)
    )

    train_x = [a[0] for a in train]
    train_adj = [a[1] for a in train]
    train_smiles = adj_to_smiles(train_adj, train_x, atomic_num_list)
    test_x = [a[0] for a in test]
    test_adj = [a[1] for a in test]
    test_smiles = adj_to_smiles(test_adj, test_x, atomic_num_list)

    # 1. traverse given directions
    if args.traverse:
        mol_smiles = None
        gen_dir = os.path.join(model_dir, 'generated')
        print('Dump figure in {}'.format(gen_dir))
        if not os.path.exists(gen_dir):
            os.makedirs(gen_dir)
        valid, novel, unique, success = [], [], [], []
        z_dim = model.b_size + model.a_size  # 324 + 45 = 369   9*9*4 + 9 * 5
        mu = np.zeros(z_dim)  # (369,) default , dtype=np.float64
        sigma_diag = np.ones(z_dim)  # (369,)

        if model.hyper_params.learn_dist:
            if len(model.ln_var) == 1:
                sigma_diag = np.sqrt(np.exp(model.ln_var.item())) * sigma_diag
            elif len(model.ln_var) == 2:
                sigma_diag[:model.b_size] = np.sqrt(np.exp(model.ln_var[0].item())) * sigma_diag[:model.b_size]
                sigma_diag[model.b_size+1:] = np.sqrt(np.exp(model.ln_var[1].item())) * sigma_diag[model.b_size+1:]

        sigma = args.temperature * sigma_diag

        z = np.random.normal(mu, sigma, (args.batch_size, z_dim))

        if args.multi_property:
            prop_pred = ['qed', 'plogp']
            directions = []

        # define properties of interest
        property_of_interest = ['qed','plogp','sa','MolWt', 'MolLogP', 'drd2', 'jnk3', 'gsk3b']
        for prop_name in tqdm(prop_pred):
            if prop_name not in property_of_interest:
                continue
            if args.baseline == 'chemspace':
                direction = np.load('./boundaries_'+args.data_name+'/boundary_'+prop_name+'.npy')
                if args.multi_property:
                    directions.append(direction)
            elif args.baseline == 'random':
                if not os.path.exists('./boundaries_'+args.baseline+'_'+args.data_name+'/boundary_'+prop_name+'.npy'):
                    direction = np.random.normal(0, 1, (1, z_dim))
                    np.save('./boundaries_'+args.baseline+'_'+args.data_name+'/boundary_'+prop_name+'.npy', direction)
                else:
                    direction = np.load('./boundaries_'+args.baseline+'_'+args.data_name+'/boundary_'+prop_name+'.npy')
            elif args.baseline == 'largest':
                direction = np.load('./boundaries_'+args.baseline+'_'+args.data_name+'/boundary_'+prop_name+'.npy')
            if not args.multi_property:
                filepath = './'+args.data_name+'_'+args.baseline+'_manipulation_'+str(args.num_range)+'/'+prop_name+'/smiles_'
                if not os.path.exists('./'+args.data_name+'_'+args.baseline+'_manipulation_'+str(args.num_range)+'/'+prop_name):
                    os.makedirs('./'+args.data_name+'_'+args.baseline+'_manipulation_'+str(args.num_range)+'/'+prop_name)
                _ = traverse(filepath, model, data=z, direction=direction, num_range=args.num_range, path_len=args.path_len,
                                                        atomic_num_list=atomic_num_list, device=device)
            else:
                print('Number of directions: ', len(directions))
                filepath = './'+args.data_name+'_'+args.baseline+'_manipulation_'+str(args.num_range)+'/'+'qed_plogp'+'/smiles_'
                if not os.path.exists('./'+args.data_name+'_'+args.baseline+'_manipulation_'+str(args.num_range)+'/'+'qed_plogp'):
                    os.makedirs('./'+args.data_name+'_'+args.baseline+'_manipulation_'+str(args.num_range)+'/'+'qed_plogp')
                _ = traverse_multi_prop(filepath, model, data=z, directions=directions, num_range=args.num_range, path_len=args.path_len,
                                                        atomic_num_list=atomic_num_list, device=device)


    # 2. Random generation
    if args.random:
        print('Load trained model and data done! Time {:.2f} seconds'.format(time.time() - start))

        save_fig = args.save_fig
        valid_ratio = []
        unique_ratio = []
        novel_ratio = []
        abs_unique_ratio = []
        abs_novel_ratio = []
        generated_latent = []
        props = [[] for i in range(len(prop_pred))]
        for i in range(args.n_experiments):
            adj, x, z = generate_mols(model, batch_size=batch_size, true_adj=None, temp=args.temperature,
                                device=device)
        
            val_res = check_validity(adj, x, atomic_num_list, correct_validity=args.correct_validity)
            novel_r, abs_novel_r = check_novelty(val_res['valid_smiles'], train_smiles, x.shape[0])
            novel_ratio.append(novel_r)
            abs_novel_ratio.append(abs_novel_r)

            unique_ratio.append(val_res['unique_ratio'])
            abs_unique_ratio.append(val_res['abs_unique_ratio'])
            valid_ratio.append(val_res['valid_ratio'])
            n_valid = len(val_res['valid_mols'])
            adj, x, z = _to_numpy_array(adj), _to_numpy_array(x), _to_numpy_array(z)
            generated_mols = val_res['valid_mols']
            
            for i in tqdm(range(len(generated_mols))):
                if generated_mols[i] is not None:
                    for idx, descriptor in enumerate(prop_pred):
                        props[idx].append(prop_pred[descriptor](generated_mols[i]))
                    generated_latent.append(z[i])
            
 
        np.save('./saved_latent/'+args.data_name+'_z.npy',np.array(generated_latent))
        np.save('./saved_latent/'+args.data_name+'_props.npy',np.array(props))

        print("validity: mean={:.2f}%, sd={:.2f}%, vals={}".format(np.mean(valid_ratio), np.std(valid_ratio), valid_ratio))
        print("novelty: mean={:.2f}%, sd={:.2f}%, vals={}".format(np.mean(novel_ratio), np.std(novel_ratio), novel_ratio))
        print("uniqueness: mean={:.2f}%, sd={:.2f}%, vals={}".format(np.mean(unique_ratio), np.std(unique_ratio),
                                                                    unique_ratio))
        print("abs_novelty: mean={:.2f}%, sd={:.2f}%, vals={}".
            format(np.mean(abs_novel_ratio), np.std(abs_novel_ratio), abs_novel_ratio))
        print("abs_uniqueness: mean={:.2f}%, sd={:.2f}%, vals={}".
            format(np.mean(abs_unique_ratio), np.std(abs_unique_ratio),
                                                                    abs_unique_ratio))
        print('Task random generation done! Time {:.2f} seconds, Data: {}'.format(time.time() - start, time.ctime()))

    # 3. prepare latent direction for largest manipulation
    if args.largest:
        args.baseline = 'largest'
        props = np.load(open(f'./saved_latent/{args.data_name}_props.npy','rb'))
        z = np.load(open(f'./saved_latent/{args.data_name}_z.npy','rb'))
        for idx, prop_name in tqdm(enumerate(prop_pred)):
            small = np.argsort(props[idx])[:3]
            large = np.argsort(props[idx])[-3:]
            # small_z = get_z(model, z[small], device)
            # large_z = get_z(model, z[large], device)
            direction = z[large].mean(axis=0)-z[small].mean(axis=0)
            np.save('./boundaries_'+args.baseline+'_'+args.data_name+'/boundary_'+prop_name+'.npy', direction)
    
    if args.disent:
        z_dim = model.b_size + model.a_size
        alpha_list = np.linspace(-3., 3., 5)
        dim_loss_adj = [[] for _ in range(z_dim)]
        dim_loss_x = [[] for _ in range(z_dim)]
        zs = np.random.normal(0, 1, (5, z_dim))
        for z in tqdm(zs):
            z = np.expand_dims(z, axis=0)
            adj_0, x_0, _ = generate_mols_dis(model, z, batch_size=1, true_adj=None, temp=args.temperature, device=device)
            for i in range(z_dim):
                z_0 = np.copy(z)
                for j in range(len(alpha_list)):
                    z_0[:,i] = alpha_list[j]
                    adj, x, _ = generate_mols_dis(model, z_0, batch_size=1, true_adj=None, temp=args.temperature,
                                            device=device)
                    dim_loss_adj[i].append(torch.mean((adj-adj_0)**2).detach().cpu().numpy())
                    dim_loss_x[i].append(torch.mean((x-x_0)**2).detach().cpu().numpy())

        x_axis = [f'z_{k}' for k in range(len(dim_loss_adj))]
        dim_loss_adj = [np.mean(ls) for ls in dim_loss_adj]
        dim_loss_x = [np.mean(ls) for ls in dim_loss_x]
        plt.xticks([])
        plt.yticks(fontsize=16)
        plt.xlabel('z')
        plt.ylabel('variance')
        plt.bar(x_axis, dim_loss_adj, color='blue', alpha=0.93)
        plt.savefig(f'{args.data_name}_moflow_dis_adj.png', bbox_inches='tight', dpi=200)
        plt.clf()
        plt.xticks([])
        plt.yticks(fontsize=16)
        plt.xlabel('z')
        plt.ylabel('variance')
        plt.bar(x_axis, dim_loss_x, color='blue', alpha=0.93)
        plt.savefig(f'{args.data_name}_moflow_dis_x.png', bbox_inches='tight', dpi=200)
        plt.clf()
        print (dim_loss_x, dim_loss_adj)
        dim_loss = [dim_loss_x[k] + dim_loss_adj[k] for k in range(len(dim_loss_x))]
        plt.xticks([])
        plt.yticks(fontsize=16)
        plt.xlabel('z')
        plt.ylabel('variance')
        plt.bar(x_axis, dim_loss, color='blue', alpha=0.93)
        print (dim_loss)
        plt.savefig(f'{args.data_name}_moflow_dis.png', bbox_inches='tight', dpi=200)