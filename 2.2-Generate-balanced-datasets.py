
# coding: utf-8

# # Generate ballanced datasets using sampling
# 
# This step is used only if the dataset is not ballanced (each class has a different number of examples). Two sampling methods will be used:
# - down-sampling: randomly cut the examples from the majority class until both classes will have the same number of examples (ballanced dataset);
# - up-sampling: create new examples to complete the minority up to the majority examples.
# 
# We will read the entire dataset and ballance it. After that, with other scripts, we will apply the same spliting steps before ML step.

# In[1]:


import numpy as np
import pandas as pd
import os


# In[20]:


# A CSV files will be create for each type of ballancing
WorkingFolder = './datasets/'
sOrigDataSet  = 'ds_MA.csv' # unballanced dataset

seed = 0          # for reproductibility
outVar = 'Lij'    # output variable

ballancePrefix = ['downsampl.','upsampl.']


# Start by reading the unballenced dataset:

# In[3]:


print('-> Reading source dataset:',sOrigDataSet,'...')
df = pd.read_csv(os.path.join(WorkingFolder, sOrigDataSet))
print('Columns:',len(df.columns),'| Rows:',len(df))
print('Done')

X = df.drop(outVar, axis=1).values # get values of features
y = df[outVar].values              # get output values


# Let's create functions for each type of sampling:

# In[4]:


# Undersampling
def UnderSamplingRnd(X,Y):
    from imblearn.under_sampling import RandomUnderSampler
    
    rus = RandomUnderSampler(return_indices=True, random_state=seed)
    X_rus, y_rus, id_rus = rus.fit_sample(X, Y)
    return (X_rus, y_rus, id_rus) # return new X, Y and removed indexes


# In[5]:


# Upsampling
def UpSamplingSMOTE(X,Y):
    from imblearn.over_sampling import SMOTE

    smote = SMOTE(ratio='minority', random_state=seed)
    X_sm, y_sm = smote.fit_sample(X, Y)
    return (X_sm, y_sm) # return new X, Y for the ballanced dataset


# In[24]:


# sampling transformation of the full dataset
# ballancePrefix = ['upsampl.', 'downsampl.']

for pref in ballancePrefix:
    if 'downsampl.' in pref:
        # apply random undersampling
        print('--> Random Undersampling ...')
        newX, newY, remIndexes = UnderSamplingRnd(X,y)
        print('Removed indexes:', remIndexes)

    if 'upsampl.' in pref:
        ## do SMOTE up-smapling
        print('--> SMOTE updampling ...')
        newX, newY = UpSamplingSMOTE(X,y)
        

    # Class counts checks
    print('Checking: Class 1 = ', sum(newY), 'Class 0 = ', len(newY) - sum(newY))
    print('New shape of inputs = ', newX.shape)
        
    # create a dataframe to save the new file
    df_ballanced = pd.DataFrame(data = newX, columns = df.drop(outVar, axis=1).columns) # add inputs
    df_ballanced[outVar] = newY # add output var

    # Save transformed file
    
    newFile = os.path.join(WorkingFolder, pref+sOrigDataSet)
    print('-->> Saving undersampled dataset',newFile,'...')
    df_ballanced.to_csv(newFile,index=False)
    print('Done!\n')


# The ballanced datasets could be used as input for Machine Learning scripts.
