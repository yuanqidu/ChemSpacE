import os 
import numpy as np
from rdkit.Chem import Descriptors

# model_names = ['moflow']
data_names = ['qm9','zinc250k']
mode_types = ['random','largest','chemspace']
mani_steps = ['1']

prop_name = ['qed','MolLogP','sa','drd2','jnk3','gsk3b','MolWt']

base_dir = './'
for m in mode_types:
    for d in data_names:
        for s in mani_steps:
            if m == 'random' or m == 'largest':
                if s == '5' or s == '10':
                    continue
            soft_success = []
            strict_success = []
            valid, novel, unique, corr = [], [], [], []
            props_result_strict = []
            props_result_soft = []
            final = os.path.join(base_dir, f'{d}_{m}_{s}_final_result.txt')
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
                elif 'corr' in line:
                    corr.append(float(line.split(" ")[-1]))
            strict = os.path.join(base_dir, f'{d}_{m}_{s}_strict.txt')
            soft = os.path.join(base_dir, f'{d}_{m}_{s}_soft.txt')
            for prop in prop_name:
                f = open(strict, 'r')
                for line in f:
                    if prop in line and prop == line.split(" ")[0]:
                        props_result_strict.append(float(line.split(" ")[-1]))
            for prop in prop_name:
                f = open(soft, 'r')
                for line in f:
                    if prop in line and prop == line.split(" ")[0]:
                        props_result_soft.append(float(line.split(" ")[-1]))
            
            print (f'{d}_{m}_{s} result report')
            table_full = ""
            strict = np.mean(strict_success)
            soft = np.mean(soft_success)
            valid = np.mean(valid)
            novel = np.mean(novel)
            unique = np.mean(unique)
            corr = np.mean(corr)
            table_full += f"{strict:.2f} & {soft:.2f} & {valid:.2f} & {novel:.2f} & {unique:.2f} & {corr:.2f}"
            print (table_full)

            table_ind_strict = f"soft/strict {soft:.2f} / {strict:.2f} & "
            for i, prop in enumerate(props_result_strict):
                table_ind_strict += f" {props_result_soft[i]:.2f} / {prop:.2f} &"
            print (table_ind_strict)

            # table_ind_soft = f"soft {soft:.2f} & "
            # for i, prop in enumerate(props_result_soft):
            #     table_ind_soft += f" {prop:.2f} &"
            # print (table_ind_soft)

            print ('\n\n')