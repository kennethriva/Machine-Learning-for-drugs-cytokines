
# coding: utf-8

# # Create datasets
# ## Scalling, Reduction and Feature Selection

# The original dataset and/or the ballanced ones will be first splitted into separated files as training and test subsets using a **seed**. All the scalling and feature selection will be apply **only on training set**:
# - *Dataset split*: train, test sets; the train set will be divided into train and validation in future Machine Learning hyperparameter search for the best model with a ML method;
# - *Scalling* of train set using centering, standardization, etc.;
# - *Reduction* of train set dimension (after scalling): decrease the number of features using less dimensions/derived features;
# - *Feature selection* using train set (after scalling): decrease the number of features by keeping only the most important for the classification.
# 
# Two CSV files will be create for each type of scalling, reduction or feature selection: *tr* - trainin and *ts* - test.

# In[1]:


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

# In[12]:


# Create scalled datasets using normalized MA dataset
# Two CSV files will be create for each type of scalling, reduction or feature selection
WorkingFolder = './datasets/'

# change this with ballanced datasets such as upsampl.ds_MA.csv or downsampl.ds_MA.csv
# if you want to run all files, you should modify the entire script by looping all
# transformation using a list of input files [original, undersampled, upsampled]
sOrigDataSet  = 'ds_MA.csv'

# Split details
seed = 0          # for reproductibility

test_size = 0.25  # train size = 1 - test_size
outVar = 'Lij'    # output variable

# Scalers: the files as prefix + original name
# =================================================
# Original (no scaling!), StandardScaler, MinMaxScaler, RobustScaler,
# QuantileTransformer (normal), QuantileTransformer(uniform)

# scaler prefix for file name
#scalerPrefix = ['o', 's', 'm', 'r', 'pyj', 'qn', 'qu']
scalerPrefix = ['o', 's', 'm']

# sklearn scalers
#scalerList   = [None, StandardScaler(), MinMaxScaler(),
#                RobustScaler(quantile_range=(25, 75)),
#                PowerTransformer(method='yeo-johnson'),
#                QuantileTransformer(output_distribution='normal'),
#                QuantileTransformer(output_distribution='uniform')]

# sklearn scalers
scalerList   = [None, StandardScaler(), MinMaxScaler()]

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

FSprefix = ['fs.rf.']

# number of total features for reduction and selection
noSelFeatures = 50


# Start by reading the original dataset:

# In[3]:


print('-> Reading source dataset:',sOrigDataSet,'...')
df = pd.read_csv(os.path.join(WorkingFolder, sOrigDataSet))
print('Columns:',len(df.columns),'Rows:',len(df))
print('Done')


# ## Dataset split
# 
# First, split the dataset using stratification for non-ballanced datasets: the ratio between the classes is the same in training and test sets.

# In[4]:


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


# ## Dataset scaling
# 
# Two files will be saved for training and test sets for each scaling including non-scalling dataset.

# In[5]:


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

print('Done!')


# ## Dimension reduction
# 
# PCA will be applied to all the previous scaled/non-scaled datasets because each scaler has a different sensibility to the outliers. The name of the transformed files could contain additional information such as the PCA explained variance. You could obtain different PCA transformed datasets using different variance values (ex: 0.99, 0.98, etc.).

# In[10]:


from sklearn.decomposition import PCA

# define the PCA variance you need
PCA_vars = [0.99]
#PCA_vars = [0.99, 0.98, 0.97]
# PCA_comps = 50 # no of PCA components

# Reduce dimension of scaled datasets + the non-scaled one
print('-> PCA reduction:')

for prefix in scalerPrefix:   # for each prefix of file (all scaled and non-scaled files)
    
    # new file name; we will add tr and ts + csv
    newFile = prefix+'.'+sOrigDataSet[:-4]
    print('*', newFile)
        
    # read data train and test
    newFile_tr = newFile +'_tr.csv'
    df_tr = pd.read_csv(os.path.join(WorkingFolder, newFile_tr))
    newFile_ts = newFile +'_ts.csv'
    df_ts = pd.read_csv(os.path.join(WorkingFolder, newFile_ts))
        
    X_tr = df_tr.drop(outVar, axis = 1) # remove output variable from input features
    y_tr = df_tr[outVar]                # get only the output variable
    X_ts = df_ts.drop(outVar, axis = 1) # remove output variable from input features
    y_ts = df_ts[outVar]                # get only the output variable

    # get only the values for features and output (as arrays)
    Xdata_tr = X_tr.values # get values of features
    Ydata_tr = y_tr.values # get output values
    Xdata_ts = X_ts.values # get values of features
    Ydata_ts = y_ts.values # get output values

    # for each PCA variance
    for PCA_var in PCA_vars:    
    
        # apply reduction transform to training
        #pca = PCA(n_components=noSelFeatures) # use PCA a number of new dimension
        pca = PCA(PCA_var) # use PCA a number of new dimension
        pca.fit(Xdata_tr)  # get transformation using training dataset
        
        #print("List of variance for each PCA component:")
        #print(pca.explained_variance_ratio_) # list of PCA component variance
        print('PCA variance  :', sum(pca.explained_variance_ratio_))
        print('PCA components:', pca.n_components_)

        # Transform training data for training and test
        X_tr_transf = pca.transform(Xdata_tr)
        X_ts_transf = pca.transform(Xdata_ts)

        # create a dataframe to save as traning and test
        df_tr_transf = pd.DataFrame(data = X_tr_transf,
                                    columns = [reductionPrefix+str(i) for i in range(1,pca.n_components_+1)])
        df_ts_transf = pd.DataFrame(data = X_ts_transf,
                                    columns = [reductionPrefix+str(i) for i in range(1,pca.n_components_+1)])

        # add output feature values
        df_tr_transf = pd.concat([df_tr_transf, y_tr], axis = 1)
        df_ts_transf = pd.concat([df_ts_transf, y_ts], axis = 1)

        # Save transformed training file
        print('----->> Saving transformed dataset ...')
        df_tr_transf.to_csv(os.path.join(WorkingFolder, reductionPrefix+str(PCA_var)+'.'+newFile_tr),
                            index=False)
        df_ts_transf.to_csv(os.path.join(WorkingFolder, reductionPrefix+str(PCA_var)+'.'+newFile_ts),
                            index=False)

print('Done!')


# Please check if there is the same variance for some PCA! In the case with only 2 PCA dimension, there is no possible to decrease the variance from 0.99! (the files for all the variance are the same)

# ## Model Based Ranking
# 
# We are applying only the following types of feature selection (FS):
# 1. **RF feature selection** - We can fit a classfier to each feature and rank the predictive power. This method selects the most powerful features individually but ignores the predictive power when features are combined. Random Forest Classifier is used in this case because it is robust, nonlinear, and doesn't require scaling. 
# 2. **Univariate feature selection using chi-squared test**.
# 3. **Univariate feature selection with mutual information**.
# 
# Each dataset will be processed with all the FS methods and training and test files will be generated.
# 
# *Note: We will not use FS for PCA! Only training set will be used for FS! Univariate methods need positive values as inputs!*

# In[17]:


# Model Based Ranking
# https://www.kaggle.com/dkim1992/feature-selection-ranking
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif

# set no of features for drugs or for proteins (equal no for both types of descriptors)
nFeats = int(noSelFeatures/2)

# set description of each FS method
#FSnames = ['Random Forest feature selection',
#           'Univariate Feature Selection',
#           'Univariate feature selection with mutual information']

FSnames = ['Random Forest feature selection']

# set sklearn methods with parameters
#FSmethods = [RandomForestClassifier(n_estimators = 50, max_depth = 4, n_jobs = -1),
#             SelectKBest(score_func=chi2, k=2),
#             SelectKBest(score_func = mutual_info_classif, k=2)]

FSmethods = [RandomForestClassifier(n_estimators = 50, max_depth = 4, n_jobs = -1)]

print('-> FEATURE SELECTION')
print('* Drug or Protein features to keep with FS = ',nFeats)

# get the list of files to be processed with FS RF 
# all training and test files except from PCA and FS!

# listFiles_tr  = ['test_tr.csv','test2_tr.csv'] # manually set of training files
listFiles_tr = [col for col in os.listdir(WorkingFolder) if ('tr' in col)
                and ('pca' not in col) and ('fs' not in col) and ('o.ds_' not in col)]

# listFiles_ts  = ['test_ts.csv','test2_ts.csv'] # manually set of test files
listFiles_ts = [col for col in os.listdir(WorkingFolder) if ('ts' in col)
                and ('pca' not in col) and ('fs' not in col) and ('o.ds_' not in col)]

print('* No of FS methods =', len(FSmethods))
print('* No of datasets   =', len(listFiles_tr))
print('* Files to process:', listFiles_tr)


# In[20]:


# for each file apply each method of FS
# no of resulting FS files = no input files * FS methods
for f in range(len(listFiles_tr)):
    newFile_tr = listFiles_tr[f]
    newFile_ts = listFiles_ts[f]
    
    # for each file and method
    for i in range(len(FSmethods)):
        iFSnames   = FSnames[i]
        iFSmethods = FSmethods[i]
        iFSprefix  = FSprefix[i]
        
        print('* Processing: ', newFile_tr, '| FS method: ', iFSnames)
    
        # define the output files for FS RF as fsrf. + input file
        newFile_FS_tr = iFSprefix + newFile_tr
        newFile_FS_ts = iFSprefix + newFile_ts
        
        #### ADD checking if files exists, do not calculate again ???
        #### ADD time for each transformation and print it ???
        
        # TRY - EXCEPT for possible errors
        ### UNIV works only with positive X values!!!!!!!!!!!!!!!!
        #try:
        # read training set
        print('---> Reading data:', newFile_tr, '...')
        df_tr = pd.read_csv(os.path.join(WorkingFolder, newFile_tr))
        X_tr = df_tr.drop(outVar, axis = 1) # remove output variable from input features
        y_tr = df_tr[outVar]                # get only the output variable

        # for each method apply a specific function defined in iFSmethods
        print('---> Calculate the scores for each feature ...')

        # get the the scores using 10-fold CV
        scores = []
        FS = iFSmethods

        # there are different evaluations for each FS method!
        #if iFSnames==FSnames[1] or iFSnames==FSnames[2]: # for univariate methods
        if iFSnames==FSnames[1]: # for univariate methods
            FS.fit(X_tr, y_tr)

        num_features = len(X_tr.columns)
        for i in range(num_features):
            # there are different evaluations for each FS method!
            if iFSnames==FSnames[0]: # for RF
                col = X_tr.columns[i]
                score = np.mean(cross_val_score(FS, X_tr[col].values.reshape(-1,1), y_tr, cv=10))
            #if iFSnames==FSnames[1] or iFSnames==FSnames[2]: # for univariate methods
            if iFSnames==FSnames[1]: # for univariate methods
                score = FS.scores_[i]

            # for all FS methods, append the scores
            scores.append((int(score*100), col))

        # create a dataframe with RF scores for each feature
        df_scores = pd.DataFrame(sorted(scores, reverse = True), columns=['FS_score','FeatureName'])

        # PROCESS PROTEIN DESCRIPTORS
        # ----------------------------------
        # get only the list for protein descriptors or define them manually!
        protein_descriptors = [col for col in X_tr.columns if ('CHOC' in col) or ('BIGC' in col)
                                or ('CHAM' in col) or ('comp_' in col) ]

        # create a dataframe with these names for proteins only
        df_prot_descr = pd.DataFrame(protein_descriptors, columns=['FeatureName'])

        print('---> Filter the scores for proteins ...')
        # Get score only for proteins: merge feature names for protein with the score
        df_protein_scores = pd.merge(df_prot_descr, df_scores, on=['FeatureName'])
        df_protein_scores_sorted = df_protein_scores.sort_values('FS_score', ascending=False)

        # get the best nFeats prot descriptors
        BestProteinFeatures = list(df_protein_scores_sorted.FeatureName[:nFeats])

        # PROCESS DRUG DESCRIPTORS
        # ---------------------------------
        # get the list of drug descriptors or create it manually!
        drug_descriptors = [col for col in X_tr.columns if col not in protein_descriptors]

        # create a dataframe with these names for proteins only
        df_drug_descr = pd.DataFrame(drug_descriptors, columns=['FeatureName'])

        print('---> Filter the scores for drugs ...')
        # Get score for proteins: merge feature names for protein with the score
        df_drug_scores = pd.merge(df_drug_descr, df_scores, on=['FeatureName'])
        df_drug_scores_sorted = df_drug_scores.sort_values('FS_score', ascending=False)

        # get the best nFeats drug descriptors
        BestDrugFeatures =list(df_drug_scores_sorted.FeatureName[:nFeats])

        # Get the list with drug and protein descriptors for the RF FS dataset
        BestRFDescriptors = [y for x in [BestDrugFeatures, BestProteinFeatures] for y in x]

        # Add output feature Lij
        BestRFDescriptors.append('Lij')

        # create feature selection dataframe
        nds_fs = df_tr[BestRFDescriptors]

        # Save feature selected training dataset
        print('---> Saving FS train set', newFile_FS_tr,'...')
        nds_fs.to_csv(os.path.join(WorkingFolder, newFile_FS_tr), index=False)

        # get the same columns from the TS set and write the file!
        # read test set
        df_ts = pd.read_csv(os.path.join(WorkingFolder, newFile_ts))

        # limit to only the selected features + output variable
        df_ts = df_ts[BestRFDescriptors]

        # Save feature selected test set
        print('---> Saving FS test set', newFile_FS_ts,'...')
        df_ts.to_csv(os.path.join(WorkingFolder, newFile_FS_ts), index=False)
        #except:
            #print('!!! Error:', newFile_FS_tr)

print('Done!')


# The univariate method produced stange output, so we eliminated these methods of FS from the script.
# 
# You could different FS methods!
# https://www.kaggle.com/dkim1992/feature-selection-ranking
# 
# In the next step, all the datasets will be used with baseline Machine Learning models.
