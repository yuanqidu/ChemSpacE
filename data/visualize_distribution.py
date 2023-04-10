import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns
from tqdm import tqdm

dataset = 'chembl'
prop = pd.read_csv(f'new_{dataset}_property.csv')
prop_names = list(prop.columns)[1:]
for prop_name in tqdm(prop_names):
    if prop_name != "smile":
        p = prop[prop_name]
        p = p.to_numpy()
        p = p[np.isfinite(p)]
        print (prop_name, np.min(p),np.max(p))
        sns.distplot(p)
        plt.xlabel(prop_name, fontsize=20)
        plt.ylabel('Density', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        plt.savefig(f'./{dataset}_prop_distributions/{dataset}_'+prop_name+'.png')
        plt.clf()
exit(0)
logp = prop['logp']
mw = prop['MW']
SA = prop['SA']
sns.distplot(qed)
plt.xlabel('Drug-likeness (QED)', fontsize=16)
plt.ylabel('Density', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig('qm9_qed.png')
plt.clf()
sns.distplot(logp)
plt.xlabel('logp', fontsize=16)
plt.ylabel('Density', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig('qm9_logp.png')
plt.clf()
sns.distplot(mw)
plt.xlabel('Molecular weight', fontsize=16)
plt.ylabel('Density', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig('qm9_MW.png')
plt.clf()
sns.distplot(SA)
plt.xlabel('Synthesis accessibility', fontsize=16)
plt.ylabel('Density', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig('qm9_SA.png')
plt.clf()
