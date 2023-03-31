import os 
import numpy as np
from rdkit.Chem import Descriptors
from tdc import Oracle
from tqdm import tqdm 
import pandas as pd 
import matplotlib.pyplot as plt
import rdkit 
import rdkit.Chem as Chem
import pickle

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

def read_manipulation_results():
    model_names = ['moflow']
    data_names = ['qm9','zinc250k']
    mode_types = ['random','large','molspace']
    mani_steps = ['1', '5', '10', '20']

    prop_name = [prop for prop, _ in Descriptors.descList]

    prop_name.extend(['sa','drd2','jnk3','gsk3b'])

    base_dir = './'
    for m in model_names:
        for d in data_names:
            if d == 'chembl':
                if m != 'hvae':
                    continue
            for s in mani_steps:
                soft_success = []
                strict_success = []
                valid, novel, unique = [], [], []
                props_result_strict = []
                props_result_soft = []
                final = os.path.join(base_dir, f'{m}_{d}_{s}_largest_final_result.txt')
                f = open(final, 'r')
                for line in f:
                    if line.startswith('total strict success'):
                        strict_success.append(float(line.split(" ")[-1]))
                    elif line.startswith('total soft success'):
                        soft_success.append(float(line.split(" ")[-1]))
                    elif 'validity' in line:
                        valid.append(100*float(line.split(" ")[-1].split("/")[0])/float(line.split(" ")[-1].split("/")[1]))
                    elif 'novelty' in line:
                        novel.append(100*float(line.split(" ")[-1].split("/")[0])/float(line.split(" ")[-1].split("/")[1]))
                    elif 'uniqueness' in line:
                        unique.append(100*float(line.split(" ")[-1].split("/")[0])/float(line.split(" ")[-1].split("/")[1]))
                strict = os.path.join(base_dir, f'{m}_{d}_{s}_largest_strict.txt')
                soft = os.path.join(base_dir, f'{m}_{d}_{s}_largest_soft.txt')
                for prop in props:
                    f = open(strict, 'r')
                    for line in f:
                        if prop in line and prop == line.split(" ")[0]:
                            props_result_strict.append(float(line.split(" ")[-1]))
                for prop in props:
                    f = open(soft, 'r')
                    for line in f:
                        if prop in line and prop == line.split(" ")[0]:
                            props_result_soft.append(float(line.split(" ")[-1]))
                
                print (f'{m}_{d}_{s} result report')
                table_full = ""
                strict = np.mean(strict_success)
                soft = np.mean(soft_success)
                valid = np.mean(valid)
                novel = np.mean(novel)
                unique = np.mean(unique)
                table_full += f"{strict:.2f} & {soft:.2f} & {valid:.2f} & {novel:.2f} & {unique:.2f}"
                print (table_full)

                table_ind_strict = f"strict {strict:.2f} & "
                for i, prop in enumerate(props_result_strict):
                    table_ind_strict += f" {prop:.2f} &"
                print (table_ind_strict)

                table_ind_soft = f"soft {soft:.2f} & "
                for i, prop in enumerate(props_result_soft):
                    table_ind_soft += f" {prop:.2f} &"
                print (table_ind_soft)

                print ('\n\n')

def read_svm_results(file, prop_pred, save_dir):
    f = open(file, 'r')
    train_acc, val_acc, test_acc = {}, {}, {}
    cur_prop = ''
    for line in f:
        if line.split(' ')[-1].replace('\n','') in prop_pred:
            cur_prop = line.split(' ')[-1].replace('\n','')
            print (cur_prop)
        if 'Finish training' in line:
            train_acc[cur_prop] = float(line.split(' ')[-1])
        elif 'Accuracy for validation set' in line:
            val_acc[cur_prop] = float(line.split(' ')[-1])
        elif 'Accuracy for remaining set' in line:
            test_acc[cur_prop] = float(line.split(' ')[-1])
    prop_name = list(prop_pred.keys())
    for i in range(len(prop_name)):
        bar1 = plt.bar(prop_name[i],train_acc[prop_name[i]],color='g')
        bar2 = plt.bar(prop_name[i],val_acc[prop_name[i]],color='b')
        bar3 = plt.bar(prop_name[i],test_acc[prop_name[i]],color='r')
        if i % 5 == 0 and i != 0:
            plt.legend((bar1,bar2,bar3), ('train','valid','test'))
            plt.ylabel('accuracy')
            plt.xlabel('properties')
            plt.savefig(save_dir+'/'+str(i//5)+'_svm_acc.png')
            plt.clf()
    for i in range(len(prop_name)):
        bar1 = plt.bar(prop_name[i],train_acc[prop_name[i]],color='g')
        bar2 = plt.bar(prop_name[i],val_acc[prop_name[i]],color='b')
        bar3 = plt.bar(prop_name[i],test_acc[prop_name[i]],color='r')
    plt.legend((bar1,bar2,bar3), ('train','valid','test'))
    plt.tick_params(labelbottom=False)
    plt.ylabel('accuracy')
    plt.xlabel('properties')
    plt.savefig(save_dir+'/all_svm_acc.png')
    plt.clf()

def cal_prop_range(dataset_name, prop_pred):
    dataset = pd.read_csv(f'../data/{dataset_name}.csv')
    dataset = pd.concat([dataset['smiles']],axis=1)
    smiles = dataset['smiles']
    smiles = smiles.to_numpy()

    prop_names = list(prop_pred.keys())
    # prop_names = [prop_names[0]]
    props = [[] for i in range(len(prop_names))]
    for smile in tqdm(smiles):
        mol = Chem.MolFromSmiles(smile)
        for i, prop_name in enumerate(prop_names):
            props[i].append(prop_pred[prop_name](mol))

    frames = []
    for i, prop_name in enumerate(prop_names):
        frames.append(pd.DataFrame(props[i],columns=[prop_name]))
    new_frames = [dataset]
    new_frames.extend(frames)
    new_dataset = pd.concat(new_frames,axis=1)
    new_dataset.to_csv(f'../data/{dataset_name}_props.csv')
    extract_names = ['smiles']
    extract_names.extend(prop_names)
    min_max_stats = new_dataset[extract_names].agg(['min','max'])
    largest_data, smallest_data = {}, {}
    for prop_name in prop_names:
        print (prop_name)
        smallest_data[prop_name] = new_dataset[new_dataset[prop_name]==min_max_stats[prop_name][0]]['smiles'].to_numpy()
        largest_data[prop_name] = new_dataset[new_dataset[prop_name]==min_max_stats[prop_name][1]]['smiles'].to_numpy()
    min_max_stats.to_pickle(f'{dataset_name}_range.pkl')
    pickle.dump(smallest_data,open(f'{dataset_name}_smallest.pkl','wb'))
    pickle.dump(largest_data,open(f'{dataset_name}_largest.pkl','wb'))


if __name__ == '__main__':
    prop_pred = cache_prop_pred()
    cal_prop_range('zinc250k', prop_pred)
    # read_svm_results('boundaries_zinc250k/log.txt', prop_pred, './svm_saved_zinc250k')