from ds_utils import *


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


# Define the list of classifiers and the correspondent parameters for Grid Search:

# In[ ]:


# Example of execution
list_estimators = [MLPClassifier()]
list_params = [

            {'learning_rate': ["constant", "invscaling", "adaptive"],
			'hidden_layer_sizes': [(10,), (20,), (30,), (50,), (100,),(200,), (50,50), (100,100)], # remember we try 20 neurons in pnly one hidden layer in the baseline
			'alpha': list(10.0 ** -np.arange(1, 7)), # [10.0 ** -np.arange(1, 7)],
			'activation': ["relu"] # we are only using relu activation since seems to be the most used nowadays
            }
              ]

# LogisticRegression(random_state=seed, class_weight=class_weights, solver='lbfgs', max_iter=500)

# Once we have decided to use a dataset we can extract its values only once. By doing this we can use some
# useful functions like the ones described below

# In[ ]:



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


# Run Search grid:

# In[ ]:


list_optimized_models = best_fitted_gridsearchCV(X_tr_data, y_tr_data, list_estimators, list_params)


# Make predictions on unseen dataset:

# In[ ]:


check_predictions_unseen_test_set(list_optimized_models, X_ts_data, y_ts_data, listFiles_ts[0])
