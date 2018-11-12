import pandas as pd
from sklearn.utils import class_weight
import os
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score, roc_auc_score, f1_score, \
recall_score, precision_score,classification_report, roc_curve, auc, matthews_corrcoef
import matplotlib.pyplot as plt

path = os.getcwd()
seed = 0
#outVar = ''

def DataCheckings(df):
    # CHECKINGS ***************************
    # Check the number of data points in the data set
    print("\nData points =", len(df))
    
    # Check the number of columns in the data set
    print("\nColumns (output + features)=",len(df.columns))
    
    # Check the data types
    print("\nData types =", df.dtypes.unique())
    
    # Dataset statistics
    print('\n')
    df.describe()
    
    # print names of columns
    print('Column Names:\n', df.columns)
    
    # see if there are categorical data
    print("\nCategorical features:", df.select_dtypes(include=['O']).columns.tolist())
    
    # Check NA values
    # Check any number of columns with NaN
    print("\nColumns with NaN: ", df.isnull().any().sum(), ' / ', len(df.columns))

    # Check any number of data points with NaN
    print("\nNo of data points with NaN:", df.isnull().any(axis=1).sum(), ' / ', len(df))

def  set_weights(y_data, option='balanced'):
    """Estimate class weights for umbalanced dataset
       If ‘balanced’, class weights will be given by n_samples / (n_classes * np.bincount(y)). 
       If a dictionary is given, keys are classes and values are corresponding class weights. 
       If None is given, the class weights will be uniform """
    cw = class_weight.compute_class_weight(option,
                                                 np.unique(y_data),
                                                 y_data)
    return cw

def datasets_parser(tr_File, ts_File, outVar='label', WorkingFolder=path):


    """Given a training and a testing dataset parse
    them to extract data. It is supposed to be csv data"""

    # read training set as dataframe
    #print('---> Reading data:', tr_File, '...')
    df_tr = pd.read_csv(os.path.join(WorkingFolder, tr_File))
    X_tr = df_tr.drop(outVar, axis = 1) # remove output variable from input features
    y_tr = df_tr[outVar]  

    # read test set as dataframe
    #print('---> Reading data:', ts_File, '...')
    df_ts = pd.read_csv(os.path.join(WorkingFolder, ts_File))
    X_ts = df_ts.drop(outVar, axis = 1) # remove output variable from input features
    y_ts = df_ts[outVar]                # get only the output variable
            
    # get only array data for train
    X_tr_data = X_tr.values # get values of features
    y_tr_data = y_tr.values # get output values
    # get only array data for test
    X_ts_data = X_ts.values # get values of features
    y_ts_data = y_ts.values # get output values

    return (X_tr_data, y_tr_data, X_ts_data, y_ts_data)



def baseline_classifiers(y_tr_data, seed=0):
   
    """return a list of the classifiers used for the baseline
    models with default values"""


# compute the class_weight for the umbalanced dataset
    class_weights = set_weights(y_tr_data)
    w0 = class_weights[0]
    w1 = class_weights[1]
    print('class weights', class_weights)

    print('Ratio 0/1', w0, ':', w1)
    
    print('**************************************')


    MLA_lassifiers = [
            
            KNeighborsClassifier(5),
            LinearSVC(class_weight={0: w0, 1: w1}, random_state=seed, max_iter=5000),
            LogisticRegression(random_state=seed, class_weight={0: w0, 1: w1}, solver='lbfgs'),
            # playing with these parameters for this small dataset, may be change for the large dataset
            # https://stats.stackexchange.com/questions/125353/output-of-scikit-svm-in-multiclass-classification-always-gives-same-label
            # parameters which get some predicte values are gamma = 1e-2, C=10
            SVC(kernel = 'rbf', gamma= 'scale', probability=True,  class_weight={0: w0, 1: w1}, random_state=seed),
            AdaBoostClassifier(random_state = seed),
            GaussianNB(),
            # MLP for small datasets use lbfgs solver which converge faster and get better perfomances. We can
            # predict some minority labels by using this solver.
            MLPClassifier(hidden_layer_sizes= (100,), random_state = seed,max_iter=1500),
            DecisionTreeClassifier(random_state = seed, class_weight={0: w0, 1: w1}),
            RandomForestClassifier(n_estimators = 100, random_state = seed, class_weight={0: w0, 1: w1}, n_jobs=-1),
            GradientBoostingClassifier(random_state=seed),
            BaggingClassifier(random_state=seed),
            # For the moment XGB is not predicting anything with this dataset, maybe with the real one suits better.
            XGBClassifier(objective='binary:logistic', gamma=0.1, learning_rate=0.1, max_depth=6,subsample=0.6,
                            colsample_bytree=0.6, n_estimators=100,  n_jobs=-1, 
                          scale_pos_weight= int(w0/w1), # ratio weights negative / positive class
                          seed=seed)

                    ]

    return MLA_lassifiers


def baseline_generator(listFiles_tr, listFiles_ts, outVar='label',
                        WorkingFolder=path, out_name = 'ML_baseline_generator.csv'):

    """"Return and create a file which contains main metrics and
    algorithms performances for a given set of train and test 
    datasets. Make sure outputVar is the name of the label class"""

    print('-> Generating Basic Machine Learning baseline...')

    # to save all dataframes
    dataframes = []

    for tr, ts in zip(listFiles_tr, listFiles_ts):
        
        # Get data from training and test files
        data = datasets_parser(tr, ts, outVar=outVar)

        X_tr_data = data[0] # X training
        y_tr_data = data[1] # y training
        X_ts_data = data[2] # X test
        y_ts_data = data[3] # y test
               
        # we are using scale_pos_weight for unballanced dataset!
        # for the baseline is better not to change algorithm parameters and set them as default whenever possible
        # and just change the ones that depends on our datasets as weights and probability for SVC
        
        MLA_compare = pd.DataFrame() 

        row_index= 0
        
        for alg in baseline_classifiers(y_tr_data):
            alg.fit(X_tr_data, y_tr_data)
            y_pred = alg.predict(X_ts_data)
            MLA_compare.loc[row_index, 'MLA_dataset'] = tr 
            MLA_name = alg.__class__.__name__
            MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
            MLA_compare.loc[row_index, 'MLA Train Accuracy'] = round(alg.score(X_tr_data, y_tr_data), 4)
            MLA_compare.loc[row_index, 'MLA Test Accuracy'] = round(alg.score(X_ts_data, y_ts_data), 4)
            MLA_compare.loc[row_index, 'MLA Precission'] = str(precision_score(y_ts_data, y_pred, average=None)) 
            MLA_compare.loc[row_index, 'MLA Recall'] = str(recall_score(y_ts_data, y_pred, average=None))
            MLA_compare.loc[row_index, 'MLA F1_score']  = f1_score(y_ts_data, y_pred, pos_label=1, average="binary",
                             sample_weight=None, labels=None)
            MLA_compare.loc[row_index, 'MLA AUC'] = roc_auc_score(y_ts_data, y_pred, average='weighted', sample_weight=None)
            MLA_compare.loc[row_index, 'MLA Matthews Coefficient'] = matthews_corrcoef(y_ts_data, y_pred)


            row_index += 1
        
        dataframes.append(MLA_compare)
    
    result = pd.concat(dataframes)
    result.sort_values(by=['MLA AUC'], ascending = False, inplace = True)
    print('---> Saving results ...')
    result.to_csv(out_name, index=False)
    print('! Please find your ML results in:', out_name)
    print('Done!')
    return result

def ROC_baseline_plot(newFile_tr, newFile_ts, outVar='label',
                        WorkingFolder=path, plot_name = 'ROC_baseline_plot.png'):


    """This function is thought to be used with one dataset after having run the baseline
    generator. The idea is to plot ROC curves for all baseline algorithms
    on the dataset with better performances"""
    
    # Get data from training and test files
    data = datasets_parser(newFile_tr, newFile_ts, outVar=outVar)

    X_tr_data = data[0] # X training
    y_tr_data = data[1] # y training
    X_ts_data = data[2] # X test
    y_ts_data = data[3] # y test

    row_index= 0
        
    for alg in baseline_classifiers(y_tr_data):
        alg.fit(X_tr_data, y_tr_data)
        y_pred = alg.predict(X_ts_data)

        fp, tp, th = roc_curve(y_ts_data, y_pred)
        roc_auc_mla = auc(fp, tp)

        MLA_name = alg.__class__.__name__
        plt.plot(fp, tp, lw=2, alpha=0.3, label='ROC {0} (AUC = {1:.2f})'.format(MLA_name, roc_auc_mla))

        row_index += 1

    plt.title('ROC Curve comparison for {0} dataset'.format(newFile_tr.strip('_.csv')))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate') 
    plt.savefig(plot_name, bbox_inches="tight")
    
  def nestedCV(estimator, training_set, param_grid, scoring='roc_auc',
                i_cv=2, o_cv=5, k_fold=StratifiedKFold):
    
    """Given an estimator (model) and a set of parameter to be tuned
    nestedCV will output the best combination after the total number
    of trials as well as a trained model based on that parameters.
    Just the training data will be used to avoid overrealistic perfromances.
    This output could be used to test on a real test set
    Useful to figure out whether, say, a random forest or 
    a SVM is better suited for our problem."""
    
    training_data = datasets_parser(training_set, outVar=outVar)

    X_tr_data = test_data[0]
    y_tr_data = test_data[1]

    trials = i_cv * o_cv

    for t in range(trials): # total number of trials 
        # To be used within GridSearch 
        inner_cv = k_fold(i_cv, shuffle=True, random_state=t)
        # To be used in outer CV 
        outer_cv = k_fold(o_cv, shuffle=True, random_state=t)
        
        # Non_nested parameter search and scoring
        gs = GridSearchCV(estimator=estimator, param_grid=param_grid,
            cv=inner_cv)
        gs.fit(X_tr_data, y_tr_data)
        non_nested_score = gs.best_score_

        # Pass the gridSearch estimator to cross_val_score
        # This will be your required outer_cv x inner_cv
        nested_score = cross_val_score(gs,X_tr_data, y_tr_data,
            cv=outer_cv, scoring=scoring, n_jobs=-1)
    
    print('AUC mean score for {0} {1:.3f} +/- {2:.3f}'.format(estimator.__class__.__name__, 
        np.mean(nested_score), 
        np.std(nested_score)))

    # the grid_search.best_estimator_ contains the corss-validated fitted model
    # with best_params_ parameters, so there is no need to refit again.
    # Simply calling grid_search.score() or grid_search.predict() will have the same 
    # effect. Because it will internally access best_estimator_ automatically.
    print('Best parameters {0}'.format(gs.best_params_))
    return gs.best_estimator_  

def check_predictions_unseen_test_set(model, test_set, outVar='label',n_metrics=7, out_name='unseen_results.csv',
                                     target_names = ['class 0', 'class 1']):
    
    """Check performances of gridsearch evaluation on test set.
    The output will write a file which contains main metrics evaluated on the
    test set. This funciton is supposed to be used with output generated by
    nestedCV() from above"""

    
    
    test_data = datasets_parser(test_set, outVar=outVar)

    X_ts_data = test_data[2]
    y_ts_data = test_data[3]



    y_pred = model.predict(X_ts_data) 
    target_names = target_names
    report = classification_report(y_ts_data, y_pred, target_names=target_names)
    # print the classification report
    print(report)
    ML_test_performances = pd.DataFrame() 
    n_metrics = n_metrics
    
    row_index= 0
        
    for i in range(n_metrics):
            
        ML_test_performances.loc[row_index, 'ML_test_set'] = test_set.strip('_.csv') 
        ML_test_performances.loc[row_index, 'ML Test Accuracy'] = round(alg.score(X_ts_data, y_ts_data), 4)
        ML_test_performances.loc[row_index, 'ML Precission'] = str(precision_score(y_ts_data, y_pred, average=None)) 
        ML_test_performances.loc[row_index, 'ML Recall'] = str(recall_score(y_ts_data, y_pred, average=None))
        ML_test_performances.loc[row_index, 'ML F1_score']  = f1_score(y_ts_data, y_pred, pos_label=1, average="binary",
                         sample_weight=None, labels=None) 

        ML_test_performances.loc[row_index, 'MLA AUC'] = roc_auc_score(y_ts_data, y_pred, average='weighted', sample_weight=None)
        ML_test_performances.loc[row_index, 'MLA Matthews Coefficient'] = matthews_corrcoef(y_ts_data, y_pred)


        row_index += 1

    ML_test_performances.sort_values(by=['MLA AUC'], ascending = False, inplace = True)
    print('---> Saving results ...')
    result.to_csv(out_name, index=False)
    print('! Please find your unseen results in:', out_name)
    print('Done!')
    return ML_test_performances

if __name__ == "__main__":
    
