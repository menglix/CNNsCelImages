# code for DeepYeast extraction followed by XGB


import numpy as np

np.random.seed(123)
import cPickle
from generateData import loadData
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras import regularizers
from keras import backend as K
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import tensorflow as tf
import random
import math
from sklearn.metrics import classification_report, confusion_matrix
import itertools
random.seed(12345)


K.set_image_dim_ordering('tf')
tf.set_random_seed(1234)


def predict_classes(self, x, batch_size=100, verbose=1):
	'''Generate class predictions for the input samples
	batch by batch.
	# Arguments
	x: input data, as a Numpy array or list of Numpy arrays
	(if the model has multiple inputs).
	batch_size: integer.
	verbose: verbosity mode, 0 or 1.
	# Returns
	A numpy array of class predictions.
	'''
	proba = self.predict(x, batch_size=batch_size, verbose=verbose)
	if proba.shape[-1] > 1:
		return proba.argmax(axis=-1)
	else:
		return (proba > 0.5).astype('int32')

        
        
def convertCNNfeature(data):
    cnnFeatures = intermediate_lyr_model.predict(data,batch_size=100, verbose=1)
    npCNNFeatures = np.array(cnnFeatures)
    return npCNNFeatures
    
    
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    accuracy = np.mean(predictions==test_labels)
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy

img_width, img_height = 64, 64

train_data_dir = '/space/data/deepyeast/training'
validation_data_dir = '/space/data/deepyeast/validation'
train_data_dir = '/space/data/deepyeast/test'

filepath="/space/data/deepyeastmodel/loaddata/correctone/weights.{epoch:02d}-{val_acc:.2f}.hdf5"
nb_train_samples = 65000
nb_test_samples = 12500
epochs = 300
batch_size = 100


# load the data
X_train, X_val, X_test = loadData()


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


if K.image_data_format() == 'channels_first':
	input_shape = (3, img_width, img_height)
else:
	input_shape = (img_width, img_height, 3)

	
model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=(64, 64, 3),kernel_regularizer=regularizers.l2(0.0005),kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3),kernel_regularizer=regularizers.l2(0.0005),kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3),kernel_regularizer=regularizers.l2(0.0005),kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))


model.add(Conv2D(128, (3, 3),kernel_regularizer=regularizers.l2(0.0005),kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3),kernel_regularizer=regularizers.l2(0.0005),kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(256, (3, 3),kernel_regularizer=regularizers.l2(0.0005),kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(256, (3, 3),kernel_regularizer=regularizers.l2(0.0005),kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(256, (3, 3),kernel_regularizer=regularizers.l2(0.0005),kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(512,kernel_regularizer=regularizers.l2(0.0005),kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(512,kernel_regularizer=regularizers.l2(0.0005),kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(12,kernel_regularizer=regularizers.l2(0.0005),kernel_initializer='glorot_normal'))
model.add(Activation('softmax'))

# load pre-trained model
model.load_weights("/space/data/deepyeastmodel/DeepYeastWeight.hdf5")

# extract the features from the second to the last fc layer
intermediate_lyr_model = Model(inputs=model.input,outputs=model.get_layer('dropout_2').output)




# combine train and validation for sklearn
trainvalX = np.concatenate((X_train,X_val),axis=0)
del X_train, X_val
trainvalY = np.concatenate((trainlabel,validationlabel),axis=0)

# extract features by CNN
CNNFeatures=convertCNNfeature(trainvalX)

# manually define the validation data to use in GridSearchCV
valFold=[-1]*nb_train_samples+[0]*nb_test_samples
ps = PredefinedSplit(valFold)

# define the classifier and the searching grid
XGB = XGBClassifier()
param_grid = {
    'n_estimators': [100, 200, 300, 1000]
}
# search the paramaters and find the best RF model
XGB_grid = GridSearchCV(estimator = XGB, param_grid = param_grid, cv = ps, verbose=0, n_jobs=-1)
XGB_grid.fit(CNNFeatures,trainvalY)
XGB_best_model = XGB_grid.best_estimator_


# evaluate the test accuracy
testCNNFeatures=convertCNNfeature(X_test)
testAcc = evaluate(XGB_best_model, testCNNFeatures, testlabel)

with open('/home/menglix/result/deepyeast/transferlearning/XGB_best_params.pkl', 'wb') as f:
        cPickle.dump(XGB_grid.best_params_, f)



# Clean up Keras session by clearing memory. 
if K.backend()== 'tensorflow':
    K.clear_session()
