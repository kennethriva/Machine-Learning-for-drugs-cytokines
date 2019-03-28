#!/usr/bin/env python
# coding: utf-8

# # Experimental-centered dataset
# 
# One special type of normalization is centering feature values using averages of the feature in different conditions. Some papers are describing this normalization as Moving Average (MA) calculation. Thus, the experimental normalization could be defined as:
# 
# *Experimental Normalized feature = Feature - Feature Avg using an experimental condition*
# 
# Several files will be created:
# - details including original dataset, experimental-averages for features, experimental-centered features
# - only the experimental-centered features + output variable (non-standardized dataset for ML)
# 
# Different scalers will be used in other scripts using only training set and applying the same transformation to the test / validation sets.

# Let's import pandas for dataframe handling, os for working with files and our *ds_utils.py* with some functions:

# In[ ]:


import pandas as pd
import os
import ds_utils # our set of useful functions


# Choose working folder, dataset file to process, files to save for the details and only the centered features, names for the normalized and standardized dataset files, names of experimental condition columns, number of the columns for the features (starting and ending columns), output variable name. The normalized features will be added to the initial dataset generating a file with details. At the end, only the centered features and the output variable will be save too.

# In[ ]:


# working folder
WorkingFolder = './datasets/'

# dataset file to be pre-processed
dsOriginFile  = 'ds_raw.csv'
ds_G1         = 'ds.G1_raw.csv'

# resulting files
dsDetailsFile  = 'ds_details.csv'        # original dataset + centered features using conditions
dsOnlyMAsFile  = 'ds_MA.csv' # only the centered features using conditions

dsDetailsFile_G1  = 'ds.G1_details.csv'        # original dataset + centered features using conditions
dsOnlyMAsFile_G1  = 'ds.G1_MA.csv' # only the centered features using conditions

# experimental condition columns to use to center features ("MAs")
ExperimCondList = ['STANDARD_TYPE_UNITSj','ASSAY_CHEMBLID',
                   'ASSAY_TYPE','ASSAY_ORGANISM',
                   'ORGANISM','TARGET_CHEMBLID']

# output variable name
outputVar = 'Lij'

# starting and ending columns for features in the original dataset
startColFeatures = 19
endColFeatures   = 271


# Read the original file to process:

# In[ ]:


print('-> Reading original dataset ...')
df_raw = pd.read_csv(os.path.join(WorkingFolder, dsOriginFile))
print('Done')


# In[ ]:


print('-> Reading G1 raw dataset ...')
df_G1_raw = pd.read_csv(os.path.join(WorkingFolder, ds_G1))
print('Done')


# Check the original dataset:

# Center each feature using the experimental condition columns:
# - create a temporal dataframe
# - copy the experimental columns and the features to center
# - for each feature, center the values as difference between the feature value and the average of this feature for a specific value of a specific experimental condition

# In[ ]:


# copy only data for the experimental condition columns into a temporal dataframe
print('-> Center features using experimental conditions ...')
print('--> Reading the experimental data ...')
newdf = df_raw[ExperimCondList].copy()

# get the experimental condition names from the new dataframe
exper_conds = newdf.columns
print(exper_conds)


# In[ ]:


# get list of descriptor names
print('--> Reading the feature names ...')
descriptors = df_raw[df_raw.columns[startColFeatures-1:endColFeatures]].columns
print(descriptors)


# In[ ]:


# get list of descriptor names G1
print('--> Reading the G1 feature names ...')
descriptors_G1 = df_G1_raw[df_G1_raw.columns[startColFeatures-1:endColFeatures]].columns
print(descriptors_G1)


# Calculate the centered features using the averages of theses features for experimental condition:

# In[ ]:


# create a list only for the MA names (centered values using experimental conditions)
onlyMAs = []

# FOR each pair Exp cond - feature calculate MA and add it to the original dataset
print('--> Centering features using', len(descriptors)*len(exper_conds),
      'pairs of experiment - feature ...')

for experim in exper_conds:    # for each experimental condition colunm
    for descr in descriptors:  # for each feature
        
        # calculate the Avg for a pair of experimental conditions and a descriptor
        avgs = df_raw.groupby(experim, as_index = False).agg({descr:"mean"})
        
        # rename the avg column name
        avgs = avgs.rename(columns={descr: 'avg-'+ descr + '-' + experim})
        
        # merge an Avg to datasets
        df_raw = pd.merge(df_raw, avgs, on=[experim])
        df_G1_raw = pd.merge(df_G1_raw, avgs, on=[experim])
        
        
        # add MA to the datasets for pair Exp cond - descr
        df_raw['MA-'+descr+'-'+experim] = df_raw[descr] - df_raw['avg-'+descr+'-'+experim]
        df_G1_raw['MA-'+descr+'-'+experim] = df_G1_raw[descr] - df_G1_raw['avg-'+descr+'-'+experim]
        
        # add the name of the MA to the list
        onlyMAs.append('MA-'+descr+'-'+experim)
        
print("Done!")
# print the new column names
print('Columns of the dataset:')
df_raw.columns


# Save all the details as file: ds_raw with Avgs, MAs, etc.

# In[ ]:


print('--> Saving the dataset with all details ...')
df_raw.to_csv(os.path.join(WorkingFolder, dsDetailsFile), index=False)
# print only the names of columns for MAs
print('No of centered features using experiments:', len(onlyMAs))
print('Done!')

print('--> Saving the G1 dataset with all details ...')
df_G1_raw.to_csv(os.path.join(WorkingFolder, dsDetailsFile_G1), index=False)
# print only the names of columns for MAs
print('No of centered features using experiments:', len(onlyMAs))
print('Done!')


# Therefore, the file with the original dataset is containing extra columns with the averages and centered values.
# 
# In the next step, you will save only the centered features using experiments as dataset for future Machine Learning calculations.

# In[ ]:


# get only the MAs + output variable and save it as the final dataset for ML
# add ouput name to the list of MAs for the final ds
onlyMAs.append(outputVar)

# get only the MAs + output var as a new dataframe
df_MA =df_raw[onlyMAs].copy()
df_MA_G1 =df_G1_raw[onlyMAs].copy()

# no of rows before removing duplications
dsIniLength = len(df_MA)
dsIniLength_G1 = len(df_MA_G1)

# remove duplicated rows!
df_MA.drop_duplicates(inplace=True)
# df_MA_G1.drop_duplicates(inplace=True)

# check the number of removed cases!
print('No of removed rows due duplication:', len(df_MA) - dsIniLength)
#print('No of removed rows due duplication G1:', len(df_MA_G1) - dsIniLength_G1)

# save ds with only MAs + output variable for ML
print('-> Saving non-standardized dataset with MAs ...')
df_MA.to_csv(os.path.join(WorkingFolder, dsOnlyMAsFile), index=False)
print('Done!')

print('-> Saving non-standardized dataset with MAs G1 ...')
df_MA_G1.to_csv(os.path.join(WorkingFolder, dsOnlyMAsFile_G1), index=False)
print('Done!')


# In[ ]:


# print the dimension of the final MA dataset
print('Experimental-centered features dataset: rows =', len(df_MA), 'columns =', len(df_MA.columns))
print('Experimental-centered features dataset G1: rows =', len(df_MA_G1), 'columns =', len(df_MA_G1.columns))


# In[ ]:


# check repeated column names
l = list(df_MA.columns)
set([x for x in l if l.count(x) > 1])


# In[ ]:


# check repeated column names
l = list(df_MA_G1.columns)
set([x for x in l if l.count(x) > 1])


# Other transformation such as scalling will be based only on the training subset and apply to the test/validation subsets.
