from ds_utils import *


# set list of train and test files
listFiles_tr = ['pca0.99.s.ds_MA_tr.csv']
listFiles_ts = ['pca0.99.s.ds_MA_ts.csv']


# dataset folder
WorkingFolder  = './datasets/'
# BasicMLResults = 'ML_basic.csv' # a file with all the statistis for ML models

# Split details
seed = 44          # for reproductibility

# output variable
outVar = 'Lij'    

# Check the list of files to process
print('* Files to use for ML baseline generator:')
print('Training sets:\n', listFiles_tr)
print('Test sets:\n',listFiles_ts)


# Define the list of classifiers and the correspondent parameters for Grid Search:

# In[ ]:


# Example of execution
list_estimators = [RandomForestClassifier(), XGBClassifier(), MLPClassifier(), SCV()]
list_params = [{ 
           "n_estimators" : [100,500,1000],
           "max_depth" : [3, 5, 7, 9, 12, 20],
           "min_samples_split": [5,10,20],
           "min_samples_leaf" : [1, 2, 4, 6, 8, 10],
           'max_features': ['auto'],
           "class_weight" : [class_weights]
           "random_state" : [seed]},

           {'nthread':[20], #when use hyperthread, xgboost may become slower
              'objective':['binary:logistic'],
              'gamma': [0.05,0.1,0.3,0.5,1.0], # Minimum loss reduction required to make a further partition on a leaf node of the tree.
              'eta': [0.01,0.015,0.025,0.05,0.1], #so called learning rate value
              'max_depth': [6,7,8,12,15],
              'min_child_weight': [1,3,5,7],
              'silent': [0],
              'subsample': [0.2,0.4,0.6,0.8,1.0],
              'colsample_bytree': [0.6,0.7,0.8,1.0],
              'n_estimators': [100, 500, 1000], #number of trees
              'scale_pos_weight': [class_weights[0]/class_weights[1]], # sum(negative cases) / sum(positive cases) = 2.7
              'seed': [seed]},

            {'learning_rate': ["constant", "invscaling", "adaptive"],
			'hidden_layer_sizes': [(10,), (50,), (100,),(200,), (50,50), (100,100)], # remember we try 20 neurons in pnly one hidden layer in the baseline
			'alpha': [10.0 ** -np.arange(1, 7)],
			'activation': ["relu"] # we are only using relu activation since seems to be the most used nowadays

            },


            {"C":[1,10,100,1000],"gamma":[1,0.1,0.001,0.0001], "kernel":['rbf'],
            "probability" : [True], # just for radial kernel 
            "class_weight" : [class_weights],
            "random_state" : [seed]


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
