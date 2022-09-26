import lmp
import lmp.vars
import random
import lmp.dset
#%%
"""
All dataset list:
Dataset base class
Chinese Poem Dataset
Demo Dataset
Winograd NLI dataset
Wiki-Text-2 Dataset
"""
# Create demo dataset instance.
demo_dataset = lmp.dset.DemoDset()
# Create wiki-text-2 dataset instance.
wiki_dataset = lmp.dset.WikiText2Dset()
print(lmp.dset.DemoDset.vers) # ['test', 'train', 'valid']
print(f'dataset size: {len(demo_dataset)}') # 2500
print('demo dataset sample:')
rind = random.randint(0, len(demo_dataset))
print(demo_dataset[rind])