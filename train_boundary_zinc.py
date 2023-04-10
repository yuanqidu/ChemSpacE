# python3.7
"""Trains semantic boundary from latent space.

Basically, this file takes a collection of `latent code - attribute score`
pairs, and find the separation boundary by treating it as a bi-classification
problem and training a linear SVM classifier. The well-trained decision boundary
of the SVM classifier will be saved as the boundary corresponding to a
particular semantic from the latent space. The normal direction of the boundary
can be used to manipulate the correpsonding attribute of the synthesis.
"""

import os.path
import argparse
import numpy as np

from logger import setup_logger
from manipulator_margin import train_boundary
import utils.environment as env


from rdkit.Chem import Descriptors

from tdc import Oracle

import time

def check_SA(gen_smiles):
    scorer = Oracle(name = 'SA')
    score = scorer(gen_smiles)
    return score 

def check_DRD2(gen_smiles):
    scorer = Oracle(name = 'DRD2')
    score = scorer(gen_smiles)
    return score

def check_JNK3(gen_smiles):
    scorer = Oracle(name = 'JNK3')
    score = scorer(gen_smiles)
    return score

def check_GSK3B(gen_smiles):
    scorer = Oracle(name = 'GSK3B')
    score = scorer(gen_smiles)
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

def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser(
      description='Train semantic boundary with given latent codes and '
                  'attribute scores.')
  parser.add_argument('-o', '--output_dir', type=str, required=False,
                      help='Directory to save the output results. (required)')
  parser.add_argument('-c', '--latent_codes_path', type=str, required=False,
                      help='Path to the input latent codes. (required)')
  parser.add_argument('-s', '--scores_path', type=str, required=False,
                      help='Path to the input attribute scores. (required)')
  parser.add_argument('-n', '--chosen_num_or_ratio', type=float, default=0.1,
                      help='How many samples to choose for training. '
                           '(default: 0.05)')
  parser.add_argument('-r', '--split_ratio', type=float, default=0.7,
                      help='Ratio with which to split training and validation '
                           'sets. (default: 0.7)')
  parser.add_argument('-V', '--invalid_value', type=float, default=None,
                      help='Sample whose attribute score is equal to this '
                           'field will be ignored. (default: None)')

  return parser.parse_args()


def main():
  """Main function."""
  args = parse_args()
  
  logger = setup_logger('./boundaries_zinc250k', logger_name='generate_data')

  prop_pred = cache_prop_pred()
  latent_codes = np.load('./saved_latent/zinc250k_z.npy')

  scores = np.load('./saved_latent/zinc250k_props.npy')

  # TODO: Check why DRD2 raises a "ValueError: Found array with 0 sample(s) (shape=(0, 6156)) while a minimum of 1 is required."
  property_of_interest = ['qed','plogp','sa','MolWt', 'MolLogP', 'jnk3', 'gsk3b']
  time_elapse = 0
  train_nums = []
  for i, prop_name in enumerate(prop_pred):
    if prop_name not in property_of_interest:
        continue
    logger.info(prop_name)
    score = scores[i]

    boundary, train_num, train_time = train_boundary(latent_codes=latent_codes,
                              scores=score,
                              chosen_num_or_ratio=args.chosen_num_or_ratio,
                              split_ratio=args.split_ratio,
                              invalid_value=args.invalid_value,
                              logger=logger)

    end_time = time.time()
    time_elapse += train_time
    train_nums.append(train_num)
    print ('per',time_elapse)
  print ('final', time_elapse / len(property_of_interest)) 
  print ('train_num final', np.mean(train_nums))
  print (train_nums)
  np.save(os.path.join('./boundaries_zinc250k', 'boundary_'+prop_name+'.npy'), boundary)


if __name__ == '__main__':
  main()
