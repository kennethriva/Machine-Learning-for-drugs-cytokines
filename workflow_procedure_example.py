#!/usr/bin/env python
# coding: utf-8

# # WORKFLOW PROCEDURE

# In[ ]:


# import utilities
from ds_utils import *
# to plot results
get_ipython().run_line_magic('matplotlib', 'inline')


# ## How to use this code:
# 
# ### Step 1 
# 
#  From a list of train and test datasets run the baseline_generator function and check the results
#  in the output file. This file is sorted by AUC value in each dataset and algorithm. You should probably need to
#  run the ROC_baseline_plot as well to get a visualization of the previous baseline results. This will give us
#  an idea of the general performances. So the next step should be optimized the best model(s) using the best dataset
#  according to the previous results. If you want to optimized more than one model they can be stored into a list to use a grid search
#  in all models by using the nestedCV function
# 
# 
# ### Step 2 
# 
# Pick the dataset and the algorithm to optimized and pass them to the nestedCV function. This function will find the best combination of 
# parameters and train a model based on it. As an output the fitted model will be returned, so there is no need to fit the model again. This
# output could be used in the next step testing these models on a unseen test set which was not used in the nestedCV phase.
# 
# 
# ### Step 3 
# 
# From a list of optimized model by the nesteCV funtion, predict classes using an unseen test set using the check_predictions_unseen_test_set.
# This function will return a file which is sorted by AUC value as well as a roc curve plot. This file will tell us the model which achieves better performance in the 
# test set.
# 
# 
# ### Step 4 
# 
# Further analysis plotting some graphs such as ROC curve, PR, etc..

# In[ ]:


# set list of train and test files
listFiles_tr = ['minitrain.csv', 's.ds_MA_tr.csv']
listFiles_ts = ['minitest.csv', 's.ds_MA_ts.csv']
# run a baseline with datasets from above
baseline_generator(listFiles_tr, listFiles_ts)


# In[ ]:


# plot the ROC curves for the dataset which achieves the best performance
# as we can see 'minitrain.csv' is the dataset which seems to get better performances
# so let's plot roc curves on it.
newFile_tr = 'minitrain.csv' # new training data 
newFile_ts = 'minitest.csv' # new testing data
ROC_baseline_plot(newFile_tr, newFile_ts)


# According to this baseline results it seems  that GradientBoostingClassifier is a good candidate as is one of the model with higher AUC, so we can try to optimize its parameters on the minitrain  dataset since is the one the suits better GradientBoostingClassifier. For simplicity we will look for parameters on an algorithm which is faster to train, let's say Logistics Regression and  another one more complex such as Random Forest.
# 
# So we should proceed as follows:

# Once we have decided to use a dataset we can extract its values only once. By doing this we can use some
# useful functions like the ones described below

# In[ ]:


# Since now we were using just one dataset. So we keep newFile_tr and newFile_ts from above
# Get data from that datasets

values = datasets_parser(newFile_tr, newFile_ts, outVar=outVar)
X_tr_data = values[0] # X_train data
y_tr_data = values[1] # y_train data
X_ts_data = values[2] # X_test data
y_ts_data = values[3] # y_test data


# In[ ]:


def gridsearchCV_strategy(X_tr_data, y_tr_data, list_estimators, list_params):
    
    """
    
    len of list_estimators and list_params should be the same. For any
    estimator you need a list of parameters to optimize. Eg
    list_estimators = [RandomForestClassifier(),
                        LogisticRegression()]
    list_params = [{'n_estimators': [500,1000],
    'max_features': [8,10],
    'max_depth' : [4,6,8],
    'criterion' :['gini', 'entropy']},'C': [100, 1000], 'solver' : ['lbfgs'],
                                        'max_iter' : [1000, 2000], 'n_jobs' : [-1]
                                        }]                    
    """
    # First check if both lists has the same length
    
    if len(list_estimators) != len(list_params):
        
        raise ValueError("list_estimators and list_params must have the same length")
    
    
    
    # Estimate weights in the data used to look for parameters

    class_weights = set_weights(y_tr_data)
    
    
    # iterate through the list of estimators to see if any of them has some parameters such as random_state or
    # class_weight or n_jobs, if so we will set them to the chosen seed for the running task and the weights estimated
    # into this function which will be the ones obtained from the training data used.
    
    
    for est in list_estimators:
        est_params = est.get_params()
        if 'class_weight' in est_params:
            est.set_params(class_weight = class_weights)
        if 'n_jobs' in est_params:
            est.set_params(n_jobs = -1)
        if 'random_state' in est_params:
            est.set_params(random_state = seed)

    
    dict_estimators_to_optimize = {}
    
    for estimator, parameters in zip(list_estimators, list_params):
        dict_estimators_to_optimize[estimator] = parameters
    
    
    list_optimized_models = [nestedCV(estimator, X_tr_data, y_tr_data, param_grid=parameters) 
                                    for estimator, parameters  in dict_estimators_to_optimize.items()]
    
    #check which params were used in the list_optimized_models
    #for op_model in list_optimized_models:
    #    print(op_model.get_params())
    
    return list_optimized_models

    


# In[ ]:


# Example of execution
list_estimators = [RandomForestClassifier(),LogisticRegression()]
list_params = [{'n_estimators': [500], 
    'max_features': [8],
    'max_depth' : [8],
    'criterion' :['entropy']}, {'C': [1000], 'solver' : ['lbfgs'],
                                        'max_iter' : [200]
                                        }]

list_optimized_models = gridsearchCV_strategy(X_tr_data, y_tr_data, list_estimators, list_params)
# Converge warning are due to the scale of the dataset. It would be converge faster using standar_scaler
# transformation from scikit-learn


# In[ ]:


# Make predictions on unseen dataset
check_predictions_unseen_test_set(list_optimized_models, X_ts_data, y_ts_data, newFile_ts)


# In[ ]:




