### code for VGG-type Residual 18 model
# Author: Mengli Xiao

import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""


np.random.seed(123)
import cPickle
from generateData import loadData
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense, Input, Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.merge import add
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from keras import backend as K
import tensorflow as tf
import random
import math
from sklearn.metrics import classification_report, confusion_matrix
import itertools
random.seed(12345)


K.set_image_dim_ordering('tf')
tf.set_random_seed(1234)




img_width, img_height = 64, 64

train_data_dir = '/space/data/deepyeast/training'
validation_data_dir = '/space/data/deepyeast/validation'
train_data_dir = '/space/data/deepyeast/test'
train_feature_dir = '/space/data/tempfile/deepyeast_bottleneck_features_train.npy'
validation_feature_dir = '/space/data/tempfile/deepyeast_bottleneck_features_val.npy'
filepath="/space/data/deepyeastmodel/loaddata/correctone/weights.{epoch:02d}-{val_loss:.2f}.hdf5"
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

 # copy and paste the model specification in the train scripts--------------------------------------------------------
def relu(x):
    return Activation('relu')(x)

def neck1(nip, nop, stride):
    def unit(x):
        ident = x
        # in this case, the default stride will be 1
        x = Conv2D(nop,(3,3),
        padding='same',kernel_regularizer=regularizers.l2(0.0005),kernel_initializer='glorot_normal')(x)

        x = BatchNormalization(axis=-1)(x)
        x = relu(x)
        x = Conv2D(nop,(3,3),padding='same',kernel_regularizer=regularizers.l2(0.0005),kernel_initializer='glorot_normal')(x)

        out = add([ident,x])
        return out
    return unit
    

def neck(nip,nop,stride):
    def unit(x):

        if nip==nop:
            ident = x

            x = BatchNormalization(axis=-1)(x)
            x = relu(x)
            x = Conv2D(nop,(3,3),
            strides=(stride,stride),padding='same',kernel_regularizer=regularizers.l2(0.0005),kernel_initializer='glorot_normal')(x)

            x = BatchNormalization(axis=-1)(x)
            x = relu(x)
            x = Conv2D(nop,(3,3),padding='same',kernel_regularizer=regularizers.l2(0.0005),kernel_initializer='glorot_normal')(x)

            out = add([ident,x])
        else:
            x = BatchNormalization(axis=-1)(x)
            x = relu(x)
            ident = x

            x = Conv2D(nbp,(3,3),
            strides=(stride,stride),padding='same',kernel_regularizer=regularizers.l2(0.0005),kernel_initializer='glorot_normal')(x)

            x = BatchNormalization(axis=-1)(x)
            x = relu(x)
            x = Conv2D(nbp,(3,3),padding='same',kernel_regularizer=regularizers.l2(0.0005),kernel_initializer='glorot_normal')(x)


            ident = Conv2D(nop,(1,1),
            strides=(stride,stride),kernel_regularizer=regularizers.l2(0.0005),kernel_initializer='glorot_normal')(ident)

            out = add([ident,x])

        return out
    return unit

def cake(nip,nop,layers,std):
    def unit(x):
        for i in range(layers):
            if i==0:
                x = neck(nip,nop,std)(x)
            else:
                x = neck(nop,nop,1)(x)
        return x
    return unit

def conv_block(nip,nop,layers,std):
    def unit(x):
        for i in range(layers):
            if i==0:
                x = Conv2D(nop, (3,3), strides=(std,std),padding='same',kernel_regularizer=regularizers.l2(0.0005),kernel_initializer='glorot_normal')(x)
                x = BatchNormalization(axis=-1)(x)
                x = relu(x)
                x = Conv2D(nop, (3,3), padding='same',kernel_regularizer=regularizers.l2(0.0005),kernel_initializer='glorot_normal')(x)
                x = BatchNormalization(axis=-1)(x)
                x = relu(x)
            else:
                x = Conv2D(nop, (3,3), strides=(1,1),padding='same',kernel_regularizer=regularizers.l2(0.0005),kernel_initializer='glorot_normal')(x)
                x = BatchNormalization(axis=-1)(x)
                x = relu(x)
                x = Conv2D(nop, (3,3), padding='same',kernel_regularizer=regularizers.l2(0.0005),kernel_initializer='glorot_normal')(x)
                x = BatchNormalization(axis=-1)(x)
                x = relu(x)                
        return x
    return unit


inp = Input(shape=(64,64,3))
i = inp

i = Conv2D(64,(7,7),strides=(2,2),padding='same',kernel_regularizer=regularizers.l2(0.0005),kernel_initializer='glorot_normal')(i) # 32*32
i = BatchNormalization(axis=-1)(i)
i = relu(i)
i = MaxPooling2D(pool_size=(2, 2))(i) # 16*16

# conv2_x
i = conv_block(64,64,2,1)(i) # 16*16


# conv3_x
i = conv_block(64,64,2,2)(i) # 8*8


# conv4_x
i = conv_block(64,64,2,2)(i) # 4*4

# conv5_x
i = conv_block(64,64,2,2)(i) # 2*2



i = AveragePooling2D(pool_size=(2,2),padding='same')(i)
i = Flatten()(i)

i = Dense(12)(i)
i = Activation('softmax')(i)

model = Model(outputs=i,inputs=inp)


# # # # # # # # # # # # # # load the model weights file .hdf5 here 
model.load_weights("/space/data/deepyeastmodel/Res18plainweights.299-1.17.hdf5")
sgd = SGD(lr=0.0, decay=0.04, momentum=0.9, nesterov=False)
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])


# copy and paste the model specification in the train scripts--------------------------------------------------------




scores = model.evaluate(
    X_test,test_label,
    batch_size=100,verbose=1)


print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))




# Clean up Keras session by clearing memory. 
if K.backend()== 'tensorflow':
    K.clear_session()
