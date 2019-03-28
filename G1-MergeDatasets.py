#!/usr/bin/env python
# coding: utf-8

# ## G1 merge datasets
# 
# Simulate the Chembl drug information from the initial Chembl dataset:
# - create all the experimental conditions

# In[ ]:


import pandas as pd
import os
import ds_utils


# In[ ]:


WorkingFolder = './datasets/'
ExperimCondList = ['STANDARD_TYPE_UNITSj','ASSAY_CHEMBLID','ASSAY_TYPE','ASSAY_ORGANISM',
                   'ORGANISM','TARGET_CHEMBLID']


# In[ ]:


# details with MAs (ds_G1_raw0.csv was created manually)
df_det = pd.read_csv(os.path.join(WorkingFolder, 'ds_details.csv'))
df_new = pd.read_csv(os.path.join(WorkingFolder, 'ds_G1_raw0.csv'))


# Get PROTEIN_ACCESSION from details file:

# In[ ]:


df_pa =df_det[['PROTEIN_ACCESSION','TARGET_CHEMBLID']].drop_duplicates()
df_pa


# Merge PROTEIN_ACCESSION to G1:

# In[ ]:


df_last = pd.merge(df_new, df_pa, on=['TARGET_CHEMBLID'])
df_last.head()


# In[ ]:


# change name
df_last.rename(columns={'PROTEIN_ACCESSION_x': 'PROTEIN_ACCESSION'}, inplace=True)
df_last.columns


# In[ ]:


# place the real values
df_last['PROTEIN_ACCESSION']=df_last['PROTEIN_ACCESSION_y']
df_last.drop('PROTEIN_ACCESSION_y', axis = 1, inplace=True)
df_last.head()


# Save the final G1 file to be used to add descriptors, MA, and predictions with the best model:

# In[ ]:


# merge first protein descriptor file
df_prot1 = pd.read_csv(os.path.join(WorkingFolder, 'Protein_descriptors.csv'))
df_last2 = pd.merge(df_last, df_prot1, on='PROTEIN_ACCESSION')
df_last2.head()


# In[ ]:


# merge 2nd protein descriptor file
df_prot2 = pd.read_csv(os.path.join(WorkingFolder, 'Protein_descriptors2.csv'))
df_last3 = pd.merge(df_last2, df_prot2, on='PROTEIN_ACCESSION')
df_last3.head()


# In[ ]:


print('-> Saving the final G1 file ...')
df_last3.to_csv(os.path.join(WorkingFolder, 'ds.G1_raw.csv'), index=False)
print('Done!')


# In[ ]:




