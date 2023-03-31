import numpy as np 
import os 
import math
import pickle
import argparse
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw
from tdc import Oracle
from rdkit import DataStructs
from rdkit import RDLogger
from scipy import stats
RDLogger.DisableLog('rdApp.*') 

import mflow.utils.environment as env


def check_validity(generated_all_smiles):
    count = 0
    valid_mols = []
    for sm in generated_all_smiles:
        mol = Chem.MolFromSmiles(sm)
        if mol is not None:
            valid_mols.append(sm)
            count += 1
    return count, valid_mols

def check_sim(train_smiles, gen_smiles):
    train_fps = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(train_smiles), 4, nBits=2048)
    gen_fps = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(gen_smiles), 4, nBits=2048)
    dist = DataStructs.TanimotoSimilarity(train_fps, gen_fps, returnDistance=True)
    score = np.mean(dist)
    return score

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

def check_unique(generated_all_smiles):
    return len(set(generated_all_smiles))

def check_novelty(generated_all_smiles, train_smiles):
    new_molecules = 0
    for sm in generated_all_smiles:
        if sm not in train_smiles:
            new_molecules += 1
    return new_molecules

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='zinc250k')
parser.add_argument("--baseline", type=str, default='chemspace')
parser.add_argument("--mani_range", type=int, default=5)
parser.add_argument("--save_dir", type=str, default='./')
parser.add_argument("--num_samples", type=str, default=200)
parser.add_argument("--path_len", type=int, default=21)
parser.add_argument("--epsilon", type=str, default=0.05)
parser.add_argument("--gamma", type=str, default=0.05)
args = parser.parse_args()

# props = ['MaxEStateIndex', 'MinEStateIndex', 'MaxAbsEStateIndex', 'MinAbsEStateIndex', 'qed', 'MolWt', 'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons', 'NumRadicalElectrons', 'MaxPartialCharge', 'MinPartialCharge', 'MaxAbsPartialCharge', 'MinAbsPartialCharge', 'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3', 'MWHI', 'MWLOW', 'CHGHI', 'CHGLO', 'LOGPHI', 'LOGPLOW', 'MRHI', 'MRLOW', 'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'HallKierAlpha', 'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 'VSA13', 'VSA14', 'TPSA', 'EState1', 'EState10', 'EState2', 'EState3', 'EState4', 'EState5', 'EState6', 'EState7', 'EState8', 'EState9', 'FractionCSP3', 'HeavyAtomCount', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds', 'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings', 'RingCount', 'MolLogP', 'MolMR', 'noTert', 'ArN', 'N', 'NH', 'COO2', 'noCOO', 'S', 'HOCCN', 'Imine', 'NH0', 'NH1', 'NH2', 'Ndealkylation1', 'Ndealkylation2', 'Nhpyrrole', 'SH', 'aldehyde', 'carbamate', 'halide', 'oxid', 'amide', 'amidine', 'aniline', 'methyl', 'azide', 'azo', 'barbitur', 'benzene', 'benzodiazepine', 'bicyclic', 'diazo', 'dihydropyridine', 'epoxide', 'ether', 'furan', 'guanido', 'halogen', 'hdrzine', 'hdrzone', 'imidazole', 'imide', 'isocyan', 'isothiocyan', 'ketone', 'Topliss', 'lactam', 'lactone', 'methoxy', 'morpholine', 'nitrile', 'nitro', 'arom', 'nonortho', 'nitroso', 'oxazole', 'oxime', 'hydroxylation', 'phenol', 'noOrthoHbond', 'acid', 'piperdine', 'piperzine', 'priamide', 'prisulfonamd', 'pyridine', 'quatN', 'sulfide', 'sulfonamd', 'sulfone', 'acetylene', 'tetrazole', 'thiazole', 'thiocyan', 'thiophene', 'alkane', 'urea']
# props = ['qed', 'SA', 'DRD2', 'JNK3', 'GSK3B', 'MolWt', 'MolLogP', 'BalabanJ', 'BertzCT', 'CHGHI', 'CHGLO', 'acetylene', 'tetrazole', 'thiazole', 'thiocyan']
with open('../data/'+args.dataset+'.txt') as f:
    train_smiles = [line.strip("\r\n ") for line in f]


prop_pred = cache_prop_pred()
prop_range_pd = pickle.load(open('../data/zinc250k_range.pkl','rb'))
prop_range_pd = prop_range_pd.T
print(prop_range_pd.keys())
# prop_range_pd = pickle.load(open(f'{args.dataset}_range.pkl','rb'))
# prop_names = list(prop_pred.keys())
prop_names = ['qed']
print(prop_names)
prop_range = {}
for prop_name in prop_names:
    prop_range[prop_name] = prop_range_pd[prop_name][1] - prop_range_pd[prop_name][0]

# base_dir = './'+args.dataset+'_'+args.baseline+'_manipulation_'+str(args.mani_range)
# files = os.listdir(base_dir)
base_dir = './'+args.dataset+'_'+args.baseline+'_manipulation_random_'
files = ['qed_plogp']

success_rate_strict = []
success_rate_soft = []
success_rate_soft_local = []
generated_all_smiles = []
record_message_strict = []
record_message_soft = []
record_message_soft_local = []
corr_coef_all = []
record_message_corr = []
for idx, fi in tqdm(enumerate(files)):
    prop_name = fi
    if prop_name in prop_pred:
        prop_dir = os.path.join(base_dir,fi)
        smiles = os.listdir(prop_dir)
        success_strict = 0
        success_soft = 0
        success_soft_local = 0
        corr_coef_result = []
        for idx1 in range(args.num_samples):
            one_smile = []
            one_prop = []
            for idx2 in range(args.path_len):
                smiles_dir = prop_dir+'/smiles_'+str(idx1)+'_'+str(idx2)+'.npy'
                try:
                    one_smile.append(np.load(smiles_dir).tolist())
                except:
                    continue
            generated_all_smiles.extend(one_smile)
            one_prop = []
            new_smiles = [] 
            for smi in one_smile:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    one_prop.append(prop_pred[prop_name](mol))
                    new_smiles.append(smi)
            one_smile = new_smiles
            one_prop = [v for v in one_prop if not (math.isinf(v) or math.isnan(v))]
            if len(one_prop) < 2:
                corr_coef = [0]
            else:
                corr_coef = stats.pearsonr(np.arange(len(one_prop)), one_prop)
            corr_coef_result.append(np.abs(corr_coef[0]))
            # strict test
            if ((all(one_prop[idx] <= one_prop[idx+1] for idx in range(len(one_prop)-1)) or all(one_prop[idx] >= one_prop[idx+1] for idx in range(len(one_prop)-1))) and all(check_sim(new_smiles[idx],new_smiles[0]) <= check_sim(new_smiles[idx+1],new_smiles[0]) for idx in range(len(new_smiles)-1))) and len(set(one_smile)) != 1:
                success_strict += 1
                labels_for_success = ['{:.2f}'.format(label_score) for mol, label_score in zip(one_smile, one_prop)]
                record_message_strict.append(str(success_strict)+' success '+str(success_strict)+'/'+str(idx1))
            if len(one_prop) > 0:
                prop_range_ = np.max(one_prop) - np.min(one_prop)
            if (all(one_prop[idx] <= one_prop[idx+1]+args.epsilon*prop_range_ for idx in range(len(one_prop)-1)) or all(one_prop[idx]+args.epsilon*prop_range_ >= one_prop[idx+1] for idx in range(len(one_prop)-1))) and all(check_sim(new_smiles[idx],new_smiles[0]) <= check_sim(new_smiles[idx+1],new_smiles[0])+args.gamma for idx in range(len(new_smiles)-1)) and len(set(one_smile)) != 1:
                success_soft_local += 1
                labels_for_success = ['{:.2f}'.format(label_score) for mol, label_score in zip(one_smile, one_prop)]
                record_message_soft_local.append(str(success_soft_local)+' success '+str(success_soft_local)+'/'+str(idx1))
                # smile_slide = [Chem.MolFromSmiles(sms) for sms in one_smile]
                # img = Draw.MolsToGridImage(smile_slide, legends=labels_for_success, molsPerRow=7,
                #                     subImgSize=(200,200))
                # if not os.path.exists(os.path.join(args.save_dir, args.dataset+'_boundaries_soft_'+str(args.mani_range)+'/'+fi)):
                #     os.makedirs(os.path.join(args.save_dir, args.dataset+'_boundaries_soft_'+str(args.mani_range)+'/'+fi))
                # img.save(os.path.join(args.save_dir, args.dataset+'_boundaries_soft_'+str(args.mani_range)+'/'+fi+'/'+str(success_soft)+'_'+fi+'.png'))
            # soft test
            if (all(one_prop[idx] <= one_prop[idx+1]+args.epsilon*prop_range[prop_name] for idx in range(len(one_prop)-1)) or all(one_prop[idx]+args.epsilon*prop_range[prop_name] >= one_prop[idx+1] for idx in range(len(one_prop)-1))) and all(check_sim(new_smiles[idx],new_smiles[0]) <= check_sim(new_smiles[idx+1],new_smiles[0])+args.gamma for idx in range(len(new_smiles)-1)) and len(set(one_smile)) != 1:
                success_soft += 1
                labels_for_success = ['{:.2f}'.format(label_score) for mol, label_score in zip(one_smile, one_prop)]
                record_message_soft.append(str(success_soft)+' success '+str(success_soft)+'/'+str(idx1))
                # smile_slide = [Chem.MolFromSmiles(sms) for sms in one_smile]
                # img = Draw.MolsToGridImage(smile_slide, legends=labels_for_success, molsPerRow=7,
                #                     subImgSize=(200,200))
                # if not os.path.exists(os.path.join(args.save_dir, args.dataset+'_boundaries_strict_'+str(args.mani_range)+'/'+fi)):
                #     os.makedirs(os.path.join(args.save_dir, args.dataset+'_boundaries_strict_'+str(args.mani_range)+'/'+fi))
                # img.save(os.path.join(args.save_dir, args.dataset+'_boundaries_strict_'+str(args.mani_range)+'/'+fi+'/'+str(success_strict)+'_'+fi+'.png'))

        success_rate_strict.append(success_strict*100/args.num_samples)
        success_rate_soft.append(success_soft*100/args.num_samples)
        success_rate_soft_local.append(success_soft_local*100/args.num_samples)
        corr_coef_result_temp = np.array(corr_coef_result)
        corr_coef_result_temp = np.nan_to_num(corr_coef_result_temp)
        corr_coef_all.append(np.mean(corr_coef_result_temp))
        record_message_corr.append(f'{fi} corr {corr_coef_all[-1]}')
        record_message_strict.append(fi+' success '+str(success_rate_strict[-1]))
        record_message_soft.append(fi+' success '+str(success_rate_soft[-1]))
        record_message_soft_local.append(fi+' success '+str(success_rate_soft_local[-1]))

f = open(os.path.join(args.save_dir,args.dataset+'_'+args.baseline+'_'+str(args.mani_range)+'_qed_combined_corr.txt'),'w+')
for record in record_message_corr:
    f.write(record+'\n')

f = open(os.path.join(args.save_dir,args.dataset+'_'+args.baseline+'_'+str(args.mani_range)+'_qed_combined_soft_global.txt'),'w+')
for record in record_message_soft:
    f.write(record+'\n')

f = open(os.path.join(args.save_dir,args.dataset+'_'+args.baseline+'_'+str(args.mani_range)+'_qed_combined_strict.txt'),'w+')
for record in record_message_strict:
    f.write(record+'\n')

f = open(os.path.join(args.save_dir,args.dataset+'_'+args.baseline+'_'+str(args.mani_range)+'_qed_combined_soft_local.txt'),'w+')
for record in record_message_soft_local:
    f.write(record+'\n')

final_record = []
final_record.append('total soft global success rate ' + str(np.mean(success_rate_soft)))
final_record.append('total strict success rate ' + str(np.mean(success_rate_strict)))
final_record.append('total soft local success rate ' + str(np.mean(success_rate_soft_local)))
final_record.append('total corr ' + str(np.mean(corr_coef_all)))
validity, valid_mols = check_validity(generated_all_smiles)
final_record.append('validity ' + str(validity)+ '/'+ str(len(generated_all_smiles)))
novelty = check_novelty(valid_mols,train_smiles)
final_record.append('novelty ' + str(novelty) + '/' + str(len(generated_all_smiles)))
uniqueness = check_unique(valid_mols)
final_record.append('uniqueness ' + str(uniqueness) + '/' + str(len(generated_all_smiles)))
f = open(os.path.join(args.save_dir,args.dataset+'_'+args.baseline+'_'+str(args.mani_range)+'_qed_combined_final_result.txt'),'w+')
for final_r in final_record:
    f.write(final_r+'\n')