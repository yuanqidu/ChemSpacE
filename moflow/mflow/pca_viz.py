import numpy as np 
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt
import plotly.express as px
from tdc import Oracle
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw

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

def cache_prop_pred():
    prop_pred = {}
    for prop_name, function in Descriptors.descList:
        prop_pred[prop_name] = function
    prop_pred['sa'] = check_SA
    prop_pred['drd2'] = check_DRD2
    prop_pred['jnk3'] = check_JNK3 
    prop_pred['gsk3b'] = check_GSK3B
    return prop_pred

prop_pred = cache_prop_pred()
prop_name = np.array(list(prop_pred.keys()))

dataset = 'zinc250k'
latent = np.load(f'./saved_latent/{dataset}_z.npy')
prop = np.load(f'./saved_latent/{dataset}_props.npy')
# prop_name = np.load(f'./saved_latent/{dataset}_prop_name.npy')
print (latent.shape, prop.shape, prop_name.shape)

n_components = 2
pca = PCA(n_components = n_components)
components = pca.fit_transform(latent)
print (components.shape)

for i in range(prop.shape[0]):
    fig = px.scatter(
        components,
        color=prop[i],
        x=0, y=1,
        title=f'{dataset} {prop_name[i]}',
    )
    fig.write_image(f'./{dataset}_pca_prop/{dataset}_{prop_name[i]}_pca.png')