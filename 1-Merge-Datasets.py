
# coding: utf-8

# # Merge Datasets
# 
# Create a dataset from different sources: initial dataset, protein and drug descriptors, etc.
# 

# In[1]:


import pandas as pd
import os
import ds_utils # our set of useful functions


# Define an working folder with the initial files and the new datasets and the name of the file for the final dataframe.
# 
# Define the files that you want to merge into a final dataset as a dataframe:
# - create a list with the files to merge (the order in the list is the merge order)
# - create a list with fields to use for the merge (each field i corresponds to i, i+1 files)
# - create a list with the fieds to remove from the final merged dataframe

# In[2]:


WorkingFolder     = './datasets/'
FinalDataSetFile  = 'ds_raw.csv'

Files2Merge  = ['Chembl_Cytokines.csv','Drug_descriptors.csv',
               'Protein_descriptors.csv','Protein_descriptors2.csv']

# list of lists = you can merge 2 datasets using more fields! (= number of merge operations)
Fields2Merge= [['CANONICAL_SMILES'], ['PROTEIN_ACCESSION'], ['PROTEIN_ACCESSION']]

# Fields to remove from the final merged dataframe
Fields2Remove = ['No']


# Checking of file data:

# In[4]:


# for each file
for aFile in Files2Merge:
    df  = os.path.join(WorkingFolder, aFile)
    print('\n-> Checking:', aFile)
    
    # read the CSV file as dataframe
    df = pd.read_csv(os.path.join(WorkingFolder, aFile))
    
    # data checkings
    ds_utils.DataCheckings(df)


# Merge all files using fields:

# In[5]:


# set initial df as the first file data in the list
df = pd.read_csv(os.path.join(WorkingFolder, Files2Merge[0]))

# merge all the other files to the initial one
for i in range(1, len(Fields2Merge) + 1):
    aFile = os.path.join(WorkingFolder, Files2Merge[i])
    print('\n-> Merging:', aFile)
    
    # read the CSV file as dataframe
    df2merge = pd.read_csv(aFile)
    
    # Merge
    print('--> Fields to merge:', Fields2Merge[i-1])
    df = pd.merge(df, df2merge, on=Fields2Merge[i-1])

print('\n===> Merged dataset columns\n', df.columns)


# if you need, remove any column from the merged dataset:

# In[6]:


df = df.drop(Fields2Remove,axis = 1)


# Save the final dataset to disk as CSV file without index column:

# In[7]:


print('\n-> Saving final merged dataset:', FinalDataSetFile)
df.to_csv(os.path.join(WorkingFolder, FinalDataSetFile), index=False)
print('\nDone! Have fun with your dataset!')

