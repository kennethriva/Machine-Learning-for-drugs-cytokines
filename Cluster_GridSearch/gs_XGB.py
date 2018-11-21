from ds_utils import *
from sklearn.utils import class_weight
import numpy

# set list of train and test files
listFiles_tr = ['fs.rf.s.ds_MA_tr.csv']
listFiles_ts = ['fs.rf.s.ds_MA_ts.csv']


# dataset folder
WorkingFolder  = './datasets/'
# BasicMLResults = 'ML_basic.csv' # a file with all the statistis for ML models

# Split details
seed = 44          # for reproductibility

# output variable
outVar = 'Lij'    

# Check the list of files to process
print('* Datafiles:')
print('Training sets:\n', listFiles_tr)
print('Test sets:\n',listFiles_ts)

# Since now we were using just one dataset. So we keep newFile_tr and newFile_ts from above
# Get data from that datasets

values = datasets_parser(listFiles_tr[0], listFiles_ts[0],outVar=outVar, WorkingFolder=WorkingFolder)
X_tr_data = values[0] # X_train data
y_tr_data = values[1] # y_train data
X_ts_data = values[2] # X_test data
y_ts_data = values[3] # y_test data

print('X_tr_data=', X_tr_data.shape)
print('y_tr_data=', y_tr_data.shape)
print('X_ts_data=', X_ts_data.shape)
print('y_ts_data=', y_ts_data.shape)

# In order to calculate the class weight do the following
class_weights = class_weight.compute_class_weight('balanced',
                                                 numpy.unique(y_tr_data),
                                                 y_tr_data)
# Example of execution
list_estimators = [XGBClassifier()]
list_params = [{'nthread':[20], #when use hyperthread, xgboost may become slower
              'objective':['binary:logistic'],
              'gamma': [0.05,0.1,0.3,0.5,1.0], # Minimum loss reduction required to make a further partition on a leaf node of the tree.
              'learning_rate': [0.01,0.015,0.025,0.05,0.1], #so called learning rate value
              'max_depth': [6,7,8,12,15],
              'min_child_weight': [1,3,5,7],
              'silent': [0],
              'subsample': [0.2,0.4,0.6,0.8,1.0],
              'colsample_bytree': [0.6,0.7,0.8,1.0],
              'n_estimators': [100, 500, 1000], #number of trees
              'scale_pos_weight': [class_weights[0]/class_weights[1]], # sum(negative cases) / sum(positive cases) = 2.7
              'seed': [seed]}
              ]

# Run Search grid:

# In[ ]:


list_optimized_models = best_fitted_gridsearchCV(X_tr_data, y_tr_data, list_estimators, list_params)


# Make predictions on unseen dataset:

# In[ ]:


check_predictions_unseen_test_set(list_optimized_models, X_ts_data, y_ts_data, listFiles_ts[0])
