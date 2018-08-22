# code for vgg-19 extraction followed by XGB
import numpy as np
np.random.seed(123)
import cPickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit
import random
from keras.utils import np_utils
from sklearn.metrics import accuracy_score
from collections import OrderedDict
import math
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import h5py
random.seed(12345)

# generated from file deepyeasttf_vgg19.npy
train_feature_dir = '/space/data/deepyeastmodel/transferlearning/deepyeast_bottleneck_features_train_vgg19.npy'
validation_feature_dir = '/space/data/deepyeastmodel/transferlearning/deepyeast_bottleneck_features_val_vgg19.npy'
test_feature_dir = '/space/data/deepyeastmodel/transferlearning/deepyeast_bottleneck_features_test_vgg19.npy'


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    accuracy = np.mean(predictions==test_labels)
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy

nb_train_samples = 65000
nb_val_samples = 12500
NumCombind = nb_train_samples+nb_val_samples
nb_test_samples = 12500


# load training, val, test data
train_data=np.load(train_feature_dir)
validation_data = np.load(validation_feature_dir)
test_data = np.load(test_feature_dir)


# preprocess data labels
# the labels are generated from "data_overview.rmd"
trlabelpath = "/space/data/deepyeastlabels/TrainLabel.txt"
vlabelpath = "/space/data/deepyeastlabels/ValidationLabel.txt"
tlabelpath = "/space/data/deepyeastlabels/TestLabel.txt"

trainlabel=np.loadtxt(trlabelpath, comments="#",delimiter=",", skiprows=1)
train_label = np_utils.to_categorical(trainlabel, 12)


validationlabel=np.loadtxt(vlabelpath, comments="#",delimiter=",", skiprows=1)
validation_label = np_utils.to_categorical(validationlabel, 12)


testlabel=np.loadtxt(tlabelpath, comments="#",delimiter=",", skiprows=1)
test_label = np_utils.to_categorical(testlabel, 12)



# combine train and validation for sklearn
trainvalX = np.concatenate((train_data,validation_data),axis=0)
del train_data,validation_data
trainvalY = np.concatenate((trainlabel,validationlabel),axis=0)


# manually define the validation data to use in GridSearchCV
valFold=[-1]*nb_train_samples+[0]*nb_val_samples
ps = PredefinedSplit(valFold)

               
               

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()

trainvalFlatX=np.reshape(trainvalX, [NumCombind,-1])
del trainvalX

# search across 100 different combinations, and use all available cores
param_grid = {
    'n_estimators': [100, 300, 800, 1000]
}
# search the paramaters and find the best RF model
rf_grid = GridSearchCV(estimator = rf, param_grid = param_grid, cv = ps, verbose=0, n_jobs=-1)
rf_grid.fit(trainvalFlatX,trainvalY)
rf_best_model = rf_grid.best_estimator_


# evaluate the test accuracy
# flatten the data first
testFlatX=np.reshape(test_data, [nb_test_samples,-1])
testAcc = evaluate(rf_best_model, testFlatX, testlabel)

with open('/home/menglix/result/deepyeast/randomForest/vgg19Grid_rf_best_params.pkl', 'wb') as f:
        cPickle.dump(rf_grid.best_params_, f)

