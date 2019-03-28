# Search for the best DL model topology and hiperparameters using keras & sklearn
# @muntisa

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from sklearn.metrics import roc_auc_score
import numpy as np
import time as tm
from sklearn.utils import class_weight

from ds_utils import *

#########################################################################
# PARAMETERS
#########################################################################

# Dataset params
# =====================

listFiles_tr = 'fs-rf.s.ds_MA_tr.csv'
listFiles_ts = 'fs-rf.s.ds_MA_ts.csv'

no_cols     = 121 # no of columns in the dataset (features + class)
no_features = 120 # no of features

# Results file
resFileCSV = 'gs_DL2.csv'

# dataset folder
WorkingFolder  = './datasets/'

# output variable
outVar = 'Lij'

seed = 0
np.random.seed(seed)


# Grid search parameters
# ======================

neuronH1L  = [120,240] # no of neurons in H1

##topologyL = ['n','n-n','n-n-n',
##             'n-n*2','n-n*3','n-n*4','n-n*2-n','n-n*3-n','n-n*4-n','n-n*2-n*3-n*2-n',
##             'n*2','n*3','n*4','n*2-n','n*3-n*2-n','n*4-n*3-n*2',
##             'n*2-n-n*2','n*3-n*2-n*3','n*3-n*2-n-n*2-n*3','n*4-n-n*4',
##             'n-n:2',
topologyL = ['n-n*2-n*3-n*2-n']

# topologyL = ['n-n','n-n-n','n-n*2','n-n*3','n-n*2-n','n*2','n*3','n*2-n-n*2','n-n:2','n-n:3','n:2','n:3']

activationL = ['relu','tanh', 'selu','elu','linear']
dropL      = [0.4,0.5,0.6]
optimizerL = ['Adam']# ['RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'] # types of optimization algorithms
initL      =  ['glorot_normal']#['uniform', 'lecun_uniform', 'normal', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'] # how initilize the weights in the neurons
batchsL    = [1024]
epochL     = [100]
# cvNo       = 3 # 10 no of k-cross validation (default is 3)
#########################################################################

# Function to create model DL classifier, required for KerasClassifier
def create_model(topology='n', activation='relu', neurons=0, drop=0.1, optimizer='adam', init='normal'):
    # create model
    model = Sequential()

    noL = 0    # no of layer
    for L in topology.split('-'): # translate each layer from topology string
        # SET NEURONS
        # ===========
        # Check if n, n*?, n:?
        # If 'n' (default)
        if len(L)==1:
            nneurons = neurons
        if len(L)==3: # any other lenght are skipped!
            if L[1]=='*': # if n*?
                nneurons = int(neurons*int(L[2])) # if n*?, get ? and multiply with n
            else:
                if L[1]==':': # if n:?
                    nneurons = int(neurons/int(L[2])) # if n:?, get ? and divide n by ?

        # ADD LAYERS
        # ==========
        noL+=1 # increase no of layer
        if noL == 1: #if this is the first layer
            model.add(Dense(nneurons, input_dim=no_features, kernel_initializer=init, activation='relu'))
        else:
            model.add(Dense(nneurons, kernel_initializer=init, activation='relu'))

        # Add BatchNormalization after each FC layer (always)
        # ==================================================
        model.add(BatchNormalization())
          
        # Add Dropouts
        # ============
        model.add(Dropout(drop))

    # Add output layer for 2 classes model
    # ====================================
    model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
    
    # Compile model
    # =============
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Print model summary
    #print(model.summary())
    return model

#################################################################
## MAIN
#################################################################

start_time = tm.time() # get starting time


print ("-> Finding optimal DL topology ...")

# loading datasets
print ("\n--> Reading dataset ...")

values = datasets_parser(listFiles_tr, listFiles_ts,outVar=outVar, WorkingFolder=WorkingFolder)
X_train = values[0] # X_train data
y_train = values[1] # y_train data
X_test  = values[2] # X_test data
y_test  = values[3] # y_test data

print('X_tr_data=', X_train.shape)
print('y_tr_data=', y_train.shape)
print('X_ts_data=', X_test.shape)
print('y_ts_data=', y_test.shape)

# result file
fout = open(resFileCSV, 'a')
fout.write('no,topology,activation,neurons,drop,optimizer,initialize,batchs,epochs,AUROCtr,ACCtr,AUROCts,ACCts\n')
fout.close()

# In order to calculate the class weight do the following
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)
print('* DL configurations = ', len(topologyL)*len(activationL)*len(neuronH1L)*len(dropL)*len(optimizerL)
      *len(initL)*len(batchsL)*len(epochL))

n = 0 # counting models
# DL fiting
for topology in topologyL:
    for activation in activationL:
        for neurons in neuronH1L:
            for drop in dropL:
                for optimizer in optimizerL:
                    for init in initL:
                        for batchs in batchsL:
                            for epochs in epochL:
                                model = create_model(topology, activation, neurons, drop, optimizer, init)
                                model.fit(X_train, y_train,
                                          batch_size=batchs, epochs=epochs, verbose=0,
                                          shuffle=False,
                                          class_weight=class_weights)

                                score_train = model.evaluate(X_train, y_train, verbose=0)
                                y_pred_train = model.predict(X_train)
                                roc_train = roc_auc_score(y_train, y_pred_train)

                                score_test = model.evaluate(X_test, y_test, verbose=0)
                                y_pred_test = model.predict(X_test)
                                roc_test = roc_auc_score(y_test, y_pred_test)
                                
                                n+=1
                                csvData = str(n)+','+str(topology)+','+str(activation)+','+str(neurons)
                                csvData = csvData +',' + str(drop)+','+str(optimizer)+','+str(init)
                                csvData = csvData +','+str(batchs)+','+str(epochs)+','+str(roc_train)
                                csvData = csvData +','+str(score_train[1])+','+str(roc_test)+','+str(score_test[1])
                                print(csvData)
                                
                                fout = open(resFileCSV, 'a')
                                fout.write(csvData+'\n')
                                fout.close()
                                
                                del model, score_train, y_pred_train, roc_train
                                del score_test, y_pred_test, roc_test


time_min = (tm.time() - start_time)/60
time_h   = time_min/60

print("\nExecution time: %0.1f min (%0.1f sec)" % (time_h,time_min))
