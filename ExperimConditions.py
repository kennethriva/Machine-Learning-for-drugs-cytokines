#!/usr/bin/env python
# coding: utf-8

# # Experimental conditions
# 
# Get the list with experimental ontology (set of different experimental conditions) for 'STANDARD_TYPE_UNITSj','ASSAY_CHEMBLID','ASSAY_TYPE','ASSAY_ORGANISM','ORGANISM','TARGET_CHEMBLID'.

# In[ ]:


import pandas as pd
import os
import ds_utils # our set of useful functions


# Define an working folder with the initial files and the new datasets and the name of the file for the final dataframe.
# 
# Define the files that you want to merge into a final dataset as a dataframe:
# - create a list with the files to merge (the order in the list is the merge order)
# - create a list with fields to use for the merge (each field i corresponds to i, i+1 files)
# - create a list with the fieds to remove from the final merged dataframe

# In[ ]:


WorkingFolder     = './datasets/'

ExpFile  = 'ds_details.csv'
ExperimCondList = ['STANDARD_TYPE_UNITSj','ASSAY_CHEMBLID','ASSAY_TYPE','ASSAY_ORGANISM',
                   'ORGANISM','TARGET_CHEMBLID']


# In[ ]:


# set initial df as the first file data in the list
df_det = pd.read_csv(os.path.join(WorkingFolder, ExpFile))


# In[ ]:


df_det.columns


# Get only the experimental condition columns:

# In[ ]:


df_exp = df_det[ExperimCondList]
df_exp.head()


# Save each experimental condition as separate file:

# In[ ]:


for experim in ExperimCondList:
    ddf =df_exp[[experim]].drop_duplicates()
    print('\n-> Saving experimental data:', experim)
    ddf.to_csv(os.path.join(WorkingFolder, 'ListOf_'+str(experim)+'.csv'), index=False)
print('\nDone!')


# Save all the experimental condition ontologies:

# In[ ]:


ddf2 =df_exp.drop_duplicates()
ddf2.shape


# In[ ]:


ddf2.to_csv(os.path.join(WorkingFolder, 'SetOfExperimentalCondition.csv'), index=False)


# In[ ]:




