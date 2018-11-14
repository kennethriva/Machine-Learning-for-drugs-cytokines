
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

# In[1]:


import pandas as pd
import os
import ds_utils # our set of useful functions


# Choose working folder, dataset file to process, files to save for the details and only the centered features, names for the normalized and standardized dataset files, names of experimental condition columns, number of the columns for the features (starting and ending columns), output variable name. The normalized features will be added to the initial dataset generating a file with details. At the end, only the centered features and the output variable will be save too.

# In[26]:


# working folder
WorkingFolder = './datasets/'

# dataset file to be pre-processed
dsOriginFile  = 'ds_raw.csv'

# resulting files
dsDetailsFile  = 'ds_details.csv' # original dataset + centered features using conditions
dsOnlyMAsFile  = 'ds_MA.csv'      # only the centered features using conditions

# experimental condition columns to use to center features ("MAs")
ExperimCondList = ['ASSAY_CHEMBLID','ASSAY_TYPE','ASSAY_ORGANISM',
                   'TARGET_TYPE','ORGANISM','TARGET_MAPPING']

# output variable name
outputVar = 'Lij'

# starting and ending columns for features in the original dataset
startColFeatures = 19
endColFeatures   = 271


# Read the original file to process:

# In[3]:


print('-> Reading original dataset ...')
df_raw = pd.read_csv(os.path.join(WorkingFolder, dsOriginFile))
print('Done')


# Check the original dataset:

# In[4]:


ds_utils.DataCheckings(df_raw)


# Center each feature using the experimental condition columns:
# - create a temporal dataframe
# - copy the experimental columns and the features to center
# - for each feature, center the values as difference between the feature value and the average of this feature for a specific value of a specific experimental condition

# In[14]:


# copy only data for the experimental condition columns into a temporal dataframe
print('-> Center features using experimental conditions ...')
print('--> Reading the experimental data ...')
newdf = df_raw[ExperimCondList].copy()

# get the experimental condition names from the new dataframe
exper_conds = newdf.columns
print(exper_conds)


# In[16]:


# get list of descriptor names
print('--> Reading the feature names ...')
descriptors = df_raw[df_raw.columns[startColFeatures-1:endColFeatures]].columns
print(descriptors)


# Calculate the centered features using the averages of theses features for experimental condition:

# In[17]:


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
        
        # merge an Avg to dataset
        df_raw = pd.merge(df_raw, avgs, on=[experim])
        
        # add MA to the dataset for pair Exp cond - descr
        df_raw['MA-'+descr+'-'+experim] = df_raw[descr] - df_raw['avg-'+descr+'-'+experim]
        
        # add the name of the MA to the list
        onlyMAs.append('MA-'+descr+'-'+experim)
        
print("Done!")
# print the new column names
print('Columns of the dataset:')
df_raw.columns


# Save all the details as file: ds_raw with Avgs, MAs, etc.

# In[20]:


print('--> Saving the dataset with all details ...')
df_raw.to_csv(os.path.join(WorkingFolder, dsDetailsFile), index=False)
# print only the names of columns for MAs
print('No of centered features using experiments:', len(onlyMAs))
print('Done!')


# Therefore, the file with the original dataset is containing extra columns with the averages and centered values.
# 
# In the next step, you will save only the centered features using experiments as dataset for future Machine Learning calculations.

# In[22]:


# get only the MAs + output variable and save it as the final dataset for ML
# add ouput name to the list of MAs for the final ds
onlyMAs.append(outputVar)

# get only the MAs + output var as a new dataframe
df_MA =df_raw[onlyMAs].copy()

# no of rows before removing duplications
dsIniLength = len(df_MA)

# remove duplicated rows!
df_MA.drop_duplicates(inplace=True)

# check the number of removed cases!
print('No of removed rows due duplication:', len(df_MA) - dsIniLength)

# save ds with only MAs + output variable for ML
print('-> Saving non-standardized dataset with MAs ...')
df_MA.to_csv(os.path.join(WorkingFolder, dsOnlyMAsFile), index=False)
print('Done!')


# In[23]:


# print the dimension of the final MA dataset
print('Experimental-centered features dataset: rows =', len(df_MA), 'columns =', len(df_MA.columns))


# Other transformation such as scalling will be based only on the training subset and apply to the test/validation subsets.
