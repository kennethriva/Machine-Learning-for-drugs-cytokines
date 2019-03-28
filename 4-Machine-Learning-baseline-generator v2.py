#!/usr/bin/env python
# coding: utf-8

# Modify with linear classifiers and increasing SVC linear time. Use of ds_utils_v2.py!

# # Machine Learning Baseline Generator
# 
# This is the first script to use a set of ML method with default parameters in order to obtain baseline results. We will use the following *sklearn* classifiers:
# 
# 1. KNeighborsClassifier - Nearest Neighbors
# 2. GaussianNB - Gaussian Naive Bayes
# 3. LinearSVC - Linear Support vector machine (SVM)
# 4. SVC - Support vector machine (SVM) with Radial Basis Functions (RBF)
# 5. LogisticRegression - Logistic regression
# 6. MLPClassifier - Multi-Layer Perceptron (MLP) / Neural Networks
# 7. AdaBoostClassifier - AdaBoost
# 8. DecisionTreeClassifier - Decision Trees
# 9. RandomForestClassifier - Random Forest
# 10. GradientBoostingClassifier - Gradient Boosting
# 11. BaggingClassifier - ensemble meta-estimator Bagging
# 12. XGBClassifier - XGBoost
# 
# *Note: More advanced hyperparameter search will be done in future scripts!*

# In[ ]:


# import scripts
from ds_utils_v2 import *
# import warnings
# warnings.filterwarnings("ignore")


# In this notebook we will show the chosen procedure for the Machine Learning baseline generator.
# The first step is to create a list of train and test datasets which will be used to generate and estimae a set of performances of more common used algorithms. In order to have a wide approximation several metrics will be used for every model.
# 
# ### Step 1 - List of datasets and classifiers
# So as a first step lets define a list o datasets.

# In[ ]:


# dataset folder
WorkingFolder  = './datasets/'
# BasicMLResults = 'ML_basic.csv' # a file with all the statistis for ML models

# Split details
seed = 44          # for reproductibility

# output variable
outVar = 'Lij'    

# parameter for ballanced (equal number of examples in all classes) and non-ballanced dataset 
class_weight = "balanced" # use None for ballanced datasets!


# set list of train and test files

listFiles_tr = ['s.ds_MA_tr.csv','fs-rf.s.ds_MA_tr.csv','pca0.99.s.ds_MA_tr.csv']
listFiles_ts = ['s.ds_MA_ts.csv','fs-rf.s.ds_MA_ts.csv','pca0.99.s.ds_MA_ts.csv']

#listFiles_tr = [col for col in os.listdir(WorkingFolder) if ('tr' in col)
#                and (col[:5] != 'o.ds_') ]

#listFiles_ts = [col for col in os.listdir(WorkingFolder) if ('ts' in col)
#                and (col[:5] != 'o.ds_') ]

# Check the list of files to process
print('* Files to use for ML baseline generator:')
print('Training sets:\n', listFiles_tr)
print('Test sets:\n',listFiles_ts)


# Once defined our list of datasets, let's call baseline_generator function which will generate a dataframe with all performances for every combination of dataset and algorithm. Remenber we are using a set of algorithms as a baseline where are included basic, complex and ensemble methods. For more information you can call baseline_classifiers method fror the ds_utils.py script to see which algorithms and parameters are being used for the baseline. Another aspect to point out is that weights for algorithms that used this parameter are calculated using set_weights method based on the class_weight.compute_class_weight sklearn method. More information can be found at: https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html.
# 
# Let's verify the ballance of the classes and create the classifier definitions:

# In[ ]:


# calling baseline_classifiers to see which algorithms and parameters are being used. Remember that
# baseline_classifiers() need a y_tr_data argument to be executed, it can be any y_tr_data if you just want
# to see the output

y_tr_data = datasets_parser(listFiles_tr[0], listFiles_ts[0],outVar=outVar, WorkingFolder=WorkingFolder)[1]
ML_methods_used = baseline_classifiers(y_tr_data)
ML_methods_used


# ### Step 2 - Baseline generator for all datasets and ML methods
# Once settle this. The next step is to call the baseline_generator method which takes as arguments a list of train and test sets we define previously. This function will calculate some metrics for each combination of train-test sets and will create a dataframe with all the performances. The final dataframe is sorted by AUC value, so the first row will correspond to the algorithm and dataset which achieve better perforamce. 
# 
# For each dataset and method we will print here only the test **AUC** and **Accuracy**. The rest of statistics will be save on a local CSV file:

# In[ ]:


baseline = baseline_generator(listFiles_tr, listFiles_ts, outVar, WorkingFolder,
                              out_name = 'ML_baseline_generator_linear.csv')
baseline


# According to the previous result it seems that the minitrain.csv dataset tend to get better performances that s.ds_MA.tr.csv. On the other hand Gradient Boosting Classifier is the method that achieves better performance, so is probably a good candidate for the minitrain.csv dataset. We could try some combination of parameters on that dataset and algorithm in the gridsearch strategy. But before we go any further we can plot the ROC curves for this baseline so that we can have a graphic comparaison across the methods used for the baseline.

# In another notebook we will analyze how to look for a good combination of parameters for a set of chosen algorithms.

# In[ ]:




