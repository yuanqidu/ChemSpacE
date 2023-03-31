# python 3.7
"""Demo."""
import os 
import sys
import argparse
sys.path.insert(0,'..')

import numpy as np
import torch
import streamlit as st
import SessionState

import rdkit 
from rdkit import Chem 
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors

from mflow.utils.model_utils import load_model
from data.transform_zinc250k import zinc250_atomic_num_list
from mflow.models.hyperparams import Hyperparameters
from mflow.models.utils import construct_mol

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default='./results/qm9_64gnn_128-64lin_1-1mask_0d6noise_convlu1')
parser.add_argument("--snapshot-path", "-snapshot", type=str, default='model_snapshot_epoch_200')
parser.add_argument("--hyperparams-path", type=str, default='moflow-params.json')
parser.add_argument("--data_name", type=str, default='qm9')
parser.add_argument("--gpu", type=int, default=-1)
args = parser.parse_args()

if args.gpu >= 0:
    device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')

if args.data_name == 'qm9':
    atomic_num_list = [6, 7, 8, 9, 0]
elif args.data_name == 'zinc250k':
    atomic_num_list = zinc250_atomic_num_list

@st.cache(allow_output_mutation=True, show_spinner=False)
def get_model(model_name):
    """Gets model by name."""
    snapshot_path = os.path.join(args.model_dir, args.snapshot_path)
    hyperparams_path = os.path.join(args.model_dir, args.hyperparams_path)
    print("loading hyperparamaters from {}".format(hyperparams_path))
    model_params = Hyperparameters(path=hyperparams_path)
    model = load_model(snapshot_path, model_params, debug=True).to(device)
    return model


@st.cache(allow_output_mutation=True, show_spinner=False)
def factorize_model(model, layer_idx):
    """Factorizes semantics from target layers of the given model."""
    return factorize_weight(model, layer_idx)


def sample(model, num=1):
    """Samples latent codes."""
    z_dim = model.a_size + model.b_size
    codes = torch.randn(num, z_dim).to(device)
    codes = codes.detach().cpu().numpy()
    return codes


@st.cache(allow_output_mutation=True, show_spinner=False)
def synthesize(model, code, prop, prop_function):
    """Synthesizes an image with the give code."""
    z = torch.from_numpy(code).float().to(device)
    adj, x = model.reverse(z)
    molecule = [construct_mol(x_elem, adj_elem, atomic_num_list) for x_elem, adj_elem in zip(x, adj)]
    label = f'{Chem.MolToSmiles(molecule[0])} \n {prop} \t {prop_function(molecule[0]):.3f}'
    mol = Draw.MolToImage(molecule[0], legends=label)
    return mol, label


def main():
    """Main function (loop for StreamLit)."""
    st.title('ChemSpacE Explorer')
    st.sidebar.title('Options')
    reset = st.sidebar.button('Reset')

    model_name = st.sidebar.selectbox(
        'Model to Interpret',
        ['MoFlow'])

    model = get_model(model_name)
    properties = ['qed','MolWt','MolLogP']
    prop_idx = st.sidebar.selectbox(
        'Property to Interpret',
        properties)
    prop_function = None
    for i, (descriptor, function) in enumerate(Descriptors.descList):
        if prop_idx in descriptor:
            prop_function = function

    directions = []
    for prop in properties:
        directions.append(np.load('boundaries_'+args.data_name+'/boundary_'+prop+'.npy'))

    num_semantics = st.sidebar.number_input(
        'Number of semantics', value=3, min_value=None, max_value=None, step=1)
    steps = {sem_idx: 0 for sem_idx in range(num_semantics)}
    max_step = 1.0
    for sem_idx in steps:
        steps[sem_idx] = st.sidebar.slider(
            f'Semantic {properties[sem_idx]}',
            value=0.0,
            min_value=-max_step,
            max_value=max_step,
            step=0.04 * max_step if not reset else 0.0)

    image_original_placeholder = st.empty()
    image_placeholder = st.empty()
    button_placeholder = st.empty()

    try:
        base_codes = np.load(f'latent_codes/{model_name}_latents.npy')
    except FileNotFoundError:
        base_codes = sample(model)

    state = SessionState.get(model_name=model_name,
                             code_idx=0,
                             codes=base_codes[0:1])
    if state.model_name != model_name:
        state.model_name = model_name
        state.code_idx = 0
        state.codes = base_codes[0:1]

    if button_placeholder.button('Random', key=bytes([0])):
        state.code_idx += 1
        if state.code_idx < base_codes.shape[0]:
            state.codes = base_codes[state.code_idx][np.newaxis]
        else:
            state.codes = sample(model)

    code = state.codes.copy()
    original_image, original_label = synthesize(model, code, prop_idx, prop_function)
    for sem_idx, step in steps.items():
        code += directions[sem_idx] * step
    image, label = synthesize(model, code, prop_idx, prop_function)
    col1, col2 = st.beta_columns(2)
    col1.image(original_image)
    col2.image(image)
    col1.text(original_label)
    col2.text(label)


if __name__ == '__main__':
    main()
