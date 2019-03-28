#!/usr/bin/env python
# coding: utf-8

# # Create datasets for G1
# ## Scalling and Feature Selection

# The original dataset and/or the ballanced ones will be first splitted into separated files as training and test subsets using a **seed**. All the scalling and feature selection will be apply **only on training set**:
# - *Dataset split*: train, test sets; the train set will be divided into train and validation in future Machine Learning hyperparameter search for the best model with a ML method;
# - *Scalling* of train set using centering, standardization, etc.;
# - *Reduction* of train set dimension (after scalling): decrease the number of features using less dimensions/derived features;
# - *Feature selection* using train set (after scalling): decrease the number of features by keeping only the most important for the classification.
# 
# Two CSV files will be create for each type of scalling, reduction or feature selection: *tr* - trainin and *ts* - test.

# In[ ]:


import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split # for dataset split

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer


# Let's define the name of the original dataset, the folder and the prefix characters for each scalling, dimension reduction or feature selection. Each transformation will add a prefix to the previous name of the file.
# 
# **You can used the original dataset that could be unballanced or the ballanced datasets obtained with previous scripts (one file only)!**

# In[ ]:


# Create scalled datasets using normalized MA dataset
# Two CSV files will be create for each type of scalling, reduction or feature selection
WorkingFolder = './datasets/'

# change this with ballanced datasets such as upsampl.ds_MA.csv or downsampl.ds_MA.csv
# if you want to run all files, you should modify the entire script by looping all
# transformation using a list of input files [original, undersampled, upsampled]
sOrigDataSet  = 'ds_MA.csv'
sOrigDataSet_G1  = 'ds.G1_MA.csv'
sOrigDataSet_G1_det  = 'ds.G1_details.csv'

# Split details
seed = 44          # for reproductibility

test_size = 0.25  # train size = 1 - test_size
outVar = 'Lij'    # output variable

# Scalers: the files as prefix + original name
# =================================================
# Original (no scaling!), StandardScaler, MinMaxScaler, RobustScaler,
# QuantileTransformer (normal), QuantileTransformer(uniform)

# scaler prefix for file name
#scalerPrefix = ['o', 's', 'm', 'r', 'pyj', 'qn', 'qu']
# scalerPrefix = ['o', 's', 'm', 'r']
scalerPrefix = ['s']

# sklearn scalers
#scalerList   = [None, StandardScaler(), MinMaxScaler(),
#                RobustScaler(quantile_range=(25, 75)),
#                PowerTransformer(method='yeo-johnson'),
#                QuantileTransformer(output_distribution='normal'),
#                QuantileTransformer(output_distribution='uniform')]

# sklearn scalers
# scalerList   = [None, StandardScaler(), MinMaxScaler(), RobustScaler()]
scalerList   = [StandardScaler()]

# Dimension Reductions
# ===================
# PCA
reductionPrefix = 'pca'

# Feature selection
# =================
# RF feature selection, Univariate feature selection using chi-squared test,
# Univariate feature selection with mutual information

# prefix to add to the processed files for each FS method
#FSprefix = ['fs.rf.',
#            'fs.univchi.',
#            'fs.univmi.']

FSprefix = ['fs-rf.']

# number of total features for reduction and selection if we are not limited by experiment
noSelFeatures = 50


# Start by reading the original dataset:

# In[ ]:


print('-> Reading source dataset:',sOrigDataSet,'...')
df = pd.read_csv(os.path.join(WorkingFolder, sOrigDataSet))
print('Columns:',len(df.columns),'Rows:',len(df))
print('Done')

print('-> Reading source dataset G1:',sOrigDataSet_G1,'...')
df_G1 = pd.read_csv(os.path.join(WorkingFolder, sOrigDataSet_G1))
print('Columns:',len(df_G1.columns),'Rows:',len(df_G1))
print('Done')

print('-> Reading source dataset G1 details:',sOrigDataSet_G1_det,'...')
df_G1_det = pd.read_csv(os.path.join(WorkingFolder, sOrigDataSet_G1_det))
print('Columns:',len(df_G1_det.columns),'Rows:',len(df_G1_det))
print('Done')


# ## Dataset split
# 
# First, split the dataset using stratification for non-ballanced datasets: the ratio between the classes is the same in training and test sets.

# In[ ]:


# Get features and ouput as dataframes
print('--> Split of dataset in training and test ...')
X = df.drop(outVar, axis = 1) # remove output variable from input features
y = df[outVar]                # get only the output variable

# get only the values for features and output (as arrays)
Xdata = X.values # get values of features
Ydata = y.values # get output values

# split data in training and test sets (X = input features, y = output variable)
# using a seed, test size (defined above) and stratification for un-ballanced classes
X_train, X_test, y_train, y_test = train_test_split(Xdata, Ydata,
                                                    test_size=test_size,
                                                    random_state=seed,
                                                    stratify=Ydata)
print('X_train:', X_train.shape)
print('y_train:', y_train.shape)
print('X_test:', X_test.shape)
print('y_test:', y_test.shape)

print('Done!')


# In[ ]:


MAs = [col for col in df_G1.columns if ('MA-' in col)]
len(MAs)


# In[ ]:


# Get features and ouput as dataframes for G1
print('--> Split of dataset in training and test ...')

X_G1 = df_G1[MAs] # remove output variable from input features
#X_G1 = df_G1.drop(outVar, axis = 1) # remove output variable from input features
y_G1 = df_G1[outVar]                # get only the output variable

# get only the values for features and output (as arrays)
Xdata_G1 = X_G1.values # get values of features
Ydata_G1 = y_G1.values # get output values

print('Xdata_G1:', Xdata_G1.shape)
print('Ydata_G1:', Ydata_G1.shape)


# ## Dataset scaling
# 
# Two files will be saved for training and test sets for each scaling including non-scalling dataset.

# In[ ]:


# Scale dataset
print('-> Scaling dataset train and test:')

for scaler in scalerList: # or scalerPrefix
    
    # new file name; we will add tr and ts + csv
    newFile = scalerPrefix[scalerList.index(scaler)]+'.'+sOrigDataSet[:-4]
    
    # decide to scale or not
    if scaler == None: # if it is the original dataset, do not scale!
        print('--> Original (no scaler!) ...')
        X_train_transf = X_train # do not modify train set
        X_test_transf  = X_test  # do not modify test set
        
    else:              # if it is not the original dataset, apply scalers
        print('--> Scaler:', str(scaler), '...')
        X_train_transf = scaler.fit_transform(X_train) # use a scaler to modify only train set
        X_test_transf  = scaler.transform(X_test)      # use the same transformation for test set
        X_G1_transf    = scaler.transform(Xdata_G1)    # use the same transformation for G1

    # Save the training scaled dataset
    df_tr_scaler = pd.DataFrame(X_train_transf, columns=X.columns)
    df_tr_scaler[outVar]= y_train
    newFile_tr = newFile +'_tr.csv'

    print('---> Saving training:', newFile_tr, ' ...')
    df_tr_scaler.to_csv(os.path.join(WorkingFolder, newFile_tr), index=False)

    # Save the test scaled dataset
    df_ts_scaler = pd.DataFrame(X_test_transf, columns=X.columns)
    df_ts_scaler[outVar]= y_test
    newFile_ts = newFile +'_ts.csv'

    print('---> Saving test:', newFile_ts, ' ...')
    df_ts_scaler.to_csv(os.path.join(WorkingFolder, newFile_ts), index=False)
    
    # Save G1 scaled dataset for future predictions
    df_G1_scaler = pd.DataFrame(X_G1_transf, columns=X.columns)
    df_G1_scaler[outVar]= Ydata_G1
    newFile_tr = newFile +'_G1.csv'

    print('---> Saving G1 scaled:', newFile_tr, ' ...')
    df_G1_scaler.to_csv(os.path.join(WorkingFolder, newFile_tr), index=False)


print('Done!')


# In[ ]:


# save scaler as file
from sklearn.externals import joblib
scaler_filename = os.path.join(WorkingFolder, "scaler.save") 
joblib.dump(scaler, scaler_filename)


# In[ ]:


# means of the scaling
scaler.mean_


# In[ ]:


# means of the scaling
scaler.mean_


# In[ ]:


# variances of the scaling
scaler.var_


# In[ ]:


# s of the scaling
scaler.scale_


# Save to files means, vars and s for StandarScaller (we need these value for the G1 prediction!):

# In[ ]:


np.savetxt(os.path.join(WorkingFolder, 'StandardScaler_mean.csv'), scaler.mean_.reshape((-1, 1)).T, delimiter=',')
np.savetxt(os.path.join(WorkingFolder, 'StandardScaler_var.csv'), scaler.var_.reshape((-1, 1)).T, delimiter=',')
np.savetxt(os.path.join(WorkingFolder, 'StandardScaler_scale.csv'), scaler.scale_.reshape((-1, 1)).T, delimiter=',')


# In[ ]:





# ### G1 scaling

# In[ ]:


from sklearn.externals import joblib
scaler_filename = os.path.join(WorkingFolder, "scaler.save") 

# load the scaler
scaler = joblib.load(scaler_filename)


# In[ ]:


WorkingFolder = './datasets/'
fG1_MAs = "ds.G1_MA.csv"
print('-> Reading source dataset G1:',fG1_MAs,'...')
df_G1 = pd.read_csv(os.path.join(WorkingFolder, fG1_MAs))
print('Columns:',len(df_G1.columns),'Rows:',len(df_G1))
print('Done')


# In[ ]:


X_G1 = df_G1.drop(outVar, axis = 1) # remove output variable from input features
y_G1 = df_G1['Lij']                # get only the output variable

# get only the values for features and output (as arrays)
Xdata_G1 = X_G1.values # get values of features
Ydata_G1 = y_G1.values # get output values

print('Xdata_G1:', Xdata_G1.shape)
print('Ydata_G1:', Ydata_G1.shape)


# In[ ]:


X_G1_transf = scaler.transform(Xdata_G1)    # use the same transformation for G1


# In[ ]:


# Save G1 scaled dataset for future predictions
df_G1_scaler = pd.DataFrame(X_G1_transf, columns=X_G1.columns)
df_G1_scaler['Lij']= Ydata_G1
newFile_tr = 's.ds_MA_G1.csv'


# In[ ]:


print('---> Saving G1 scaled:',newFile_tr, ' ...')
df_G1_scaler.to_csv(os.path.join(WorkingFolder, newFile_tr), index=False)
print('Done!')


# ### Selection of the same features for standardized G1 
# 
# Choose for G1 only the selected features from the best model to use later for predictions:

# In[ ]:


# read G1 MAs
print('-> Reading source dataset:','s.ds_MA_G1.csv','...')
df_G1 = pd.read_csv(os.path.join(WorkingFolder, 's.ds_MA_G1.csv'))
print('Columns:',len(df_G1.columns),'Rows:',len(df_G1))
print('Done')
print(list(df_G1.columns))


# In[ ]:


# get seleted feature names from fs-rf.s.ds_MA_ts.csv
print('-> Reading:','fs-rf.s.ds_MA_ts.csv','...')
df_sel = pd.read_csv(os.path.join(WorkingFolder, 'fs-rf.s.ds_MA_ts.csv'))
print('Columns:',len(df_sel.columns),'Rows:',len(df_sel))
print('Done')
print(list(df_sel.columns))


# In[ ]:


# check repeated column names
l = list(df_sel.columns)
set([x for x in l if l.count(x) > 1])


# In[ ]:


df_G1_sel = df_G1[df_sel.columns]
print('Sel Columns:',len(df_G1_sel.columns),'Sel Rows:',len(df_G1_sel))


# In[ ]:


df_G1_sel.head()


# In[ ]:


# save the fs-rf.s.ds_MA_G1.csv
print('-> Saving:','fs-rf.s.ds_MA_G1.csv','...')
df_G1_sel.to_csv(os.path.join(WorkingFolder, 'fs-rf.s.ds_MA_G1.csv'), index=False)
print('Done!')


# In[ ]:




