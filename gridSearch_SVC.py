from ds_utils import *


# set list of train and test files
listFiles_tr = ['fs-rf.s.ds_MA_tr.csv']
listFiles_ts = ['fs-rf.s.ds_MA_ts.csv']


# dataset folder
WorkingFolder  = './datasets/'
# BasicMLResults = 'ML_basic.csv' # a file with all the statistis for ML models

# Split details
seed = 0          # for reproductibility

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
class_weights = set_weights(y_tr_data)

# Example of execution

##Baseline:
##SVC(C=1.0, cache_size=200,
##   class_weight={0: 0.667943692088382, 1: 1.9885941644562335}, coef0=0.0,
##   decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
##   max_iter=-1, probability=True, random_state=0, shrinking=True, tol=0.001,
##   verbose=False)
    
list_estimators = [SVC(random_state=seed, class_weight=class_weights, max_iter=-1)]

list_params = [{'C':[1,10],
                'gamma':['scale',0.1,0.001,0.0001],
                'kernel':['linear','rbf']}
              ]

# Run Search grid:

# In[ ]:


list_optimized_models = best_fitted_gridsearchCV(X_tr_data, y_tr_data, list_estimators, list_params)

# Make predictions on unseen dataset:

# In[ ]:


check_predictions_unseen_test_set(list_optimized_models, X_ts_data, y_ts_data, listFiles_ts[0],out_name='gs_SVC_results.csv')
