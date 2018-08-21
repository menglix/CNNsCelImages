### code for test accuracy in VGG-type 11-layer model
import numpy as np

np.random.seed(123)
import Pickle
from generateData import loadData
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras import regularizers
from keras import backend as K
import tensorflow as tf
import random
random.seed(12345)


K.set_image_dim_ordering('tf')
tf.set_random_seed(1234)





img_width, img_height = 64, 64


nb_train_samples = 65000
nb_test_samples = 12500
epochs = 300
batch_size = 100


# load the data
X_train, X_val, X_test = loadData()


# preprocess data labels
# the labels are generated from "data_overview.rmd"
tlabelpath = "/space/data/deepyeastlabels/TestLabel.txt"



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

model.load_weights("/space/data/deepyeastmodel/weights.99-1.20.hdf5")


sgd = SGD(lr=0.0, decay=0.04, momentum=0.9, nesterov=False)
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])





scores = model.evaluate(
    X_test,test_label,
    batch_size=100,verbose=1)

# print model accuracy and loss
print("%s: %.3f%%" % (model.metrics_names[0], scores[0]*100))
print("%s: %.3f%%" % (model.metrics_names[1], scores[1]*100))
# # or store it in a pickle (.pkl) file
'''
with open('~/dystModelPerformance.pkl','wb') as file: pickle.dump(model.metrics_names, file)

'''
# Clean up Keras session by clearing memory. 
if K.backend()== 'tensorflow':
    K.clear_session()
