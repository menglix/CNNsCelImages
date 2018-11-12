# residual neural network implementation code
# it corresponds to wide residual network with widening factor 2 
# The current version is under TensorFlow backend. By changing the K.set_image_dim_ordering('tf') to K.set_image_dim_ordering('th'), the backend will change to theano
# Author: Mengli Xiao

# some code were adapted from: https://ctmakro.github.io/site/on_learning/resnet_keras.html


import numpy as np
import pickle
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
K.set_image_dim_ordering('tf')


img_width, img_height = 64, 64

train_data_dir = '/space/data/deepyeast/training'
validation_data_dir = '/space/data/deepyeast/validation'
test_data_dir = '/space/data/deepyeast/test'
train_feature_dir = '/space/data/tempfile/deepyeast_bottleneck_features_train.npy'
validation_feature_dir = '/space/data/tempfile/deepyeast_bottleneck_features_val.npy'
filepath="/space/data/deepyeastmodel/ResWide/ResWide40weights.{epoch:02d}-{val_acc:.4f}.hdf5"
output_dir = '/home/menglix/result/deepyeast/ResNet/ResWide'
nb_train_samples = 65000
nb_validation_samples = 12500
nb_test_samples = 12500
epochs = 300
batch_size = 100

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

# # # # # # model specification scripts------------------------------------------------------------
def relu(x):
    return Activation('relu')(x)

def neck1(nip, nop, stride):
    def unit(x):
        nBottleneckPlane = int(nop/4)
        nbp = nBottleneckPlane
        if nip==nop:
            nbp = nBottleneckPlane
            ident = x
            # in this case, the default stride will be 1
            x = Conv2D(nbp,(1,1),padding='same',kernel_regularizer=regularizers.l2(0.0005),kernel_initializer='glorot_normal')(x)
            x = BatchNormalization(axis=-1)(x)
            x = relu(x)
            x = Conv2D(nbp,(3,3),
            padding='same',kernel_regularizer=regularizers.l2(0.0005),kernel_initializer='glorot_normal')(x)
            x = BatchNormalization(axis=-1)(x)
            x = relu(x)
            x = Conv2D(nop,(1,1),padding='same',kernel_regularizer=regularizers.l2(0.0005),kernel_initializer='glorot_normal')(x)
            out = add([ident,x])
        else:
            ident = x
            x = Conv2D(nbp,(1,1),
            strides=(stride,stride),padding='same',kernel_regularizer=regularizers.l2(0.0005),kernel_initializer='glorot_normal')(x)
            x = BatchNormalization(axis=-1)(x)
            x = relu(x)
            x = Conv2D(nbp,(3,3),padding='same',kernel_regularizer=regularizers.l2(0.0005),kernel_initializer='glorot_normal')(x)
            x = BatchNormalization(axis=-1)(x)
            x = relu(x)
            x = Conv2D(nop,(1,1),padding='same',kernel_regularizer=regularizers.l2(0.0005),kernel_initializer='glorot_normal')(x)
            ident = Conv2D(nop,(1,1),
            strides=(stride,stride),kernel_regularizer=regularizers.l2(0.0005),kernel_initializer='glorot_normal')(ident)
            out = add([ident,x])            
        return out
    return unit
    

def neck(nip,nop,stride):
    def unit(x):

        if nip==nop:
            ident = x
            x = BatchNormalization(axis=-1)(x)
            x = relu(x)
            x = Conv2D(nop,(3,3),padding='same',kernel_regularizer=regularizers.l2(0.0005),kernel_initializer='glorot_normal')(x)
            x = BatchNormalization(axis=-1)(x)
            x = relu(x)
            x = Conv2D(nop,(3,3),padding='same',kernel_regularizer=regularizers.l2(0.0005),kernel_initializer='glorot_normal')(x)            
            out = add([ident,x])
        else:
            x = BatchNormalization(axis=-1)(x)
            x = relu(x)
            ident = x
            x = Conv2D(nop,(3,3),padding='same',kernel_regularizer=regularizers.l2(0.0005),kernel_initializer='glorot_normal')(x)
            x = BatchNormalization(axis=-1)(x)
            x = relu(x)
            x = Conv2D(nop,(3,3),padding='same',kernel_regularizer=regularizers.l2(0.0005),kernel_initializer='glorot_normal')(x)
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

    
k=2
n_stages=[16, 16*k, 32*k, 64*k]

inp = Input(shape=(64,64,3))
i = inp

i = Conv2D(n_stages[0],(3,3),strides=(1,1),padding='same',kernel_regularizer=regularizers.l2(0.0005),kernel_initializer='glorot_normal')(i) # 64*64


# conv2_x
i = cake(n_stages[0],n_stages[1],6,1)(i) # 64*64


# conv3_x
i = cake(n_stages[1],n_stages[2],6,1)(i)
i = MaxPooling2D(pool_size=(2,2),padding='same')(i) # 32*32
# conv4_x
i = cake(n_stages[2],n_stages[3],6,1)(i)
i = MaxPooling2D(pool_size=(2,2),padding='same')(i) # 16*16


i = BatchNormalization(axis=-1)(i)
i = relu(i)

i = AveragePooling2D(pool_size=(16,16),padding='same')(i)
i = Flatten()(i)

i = Dense(12)(i)
i = Activation('softmax')(i)

model = Model(outputs=i,inputs=inp)

sgd = SGD(lr=0.1, decay=0.04, momentum=0.9, nesterov=False)
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
# # # # # # model specification scripts------------------------------------------------------------

# import the image data from the directory which stores each image under its category folder
# if you want more details about what the data should be arranged for imagedatagenerator to successfully import into the system, see "https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html"
 
train_datagen = ImageDataGenerator(
    featurewise_center=True,featurewise_std_normalization=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
validation_generator = train_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')


# save the model
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, mode='auto',period=50)
callbacks_list = [checkpoint]

# model fit
history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size, callbacks=callbacks_list)
with open(output_dir+'/#3_ResWide40'+'.pkl', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
	
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt 
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(output_dir+'/#3_ResWide40accuracy.pdf')



# Clean up Keras session by clearing memory. 
if K.backend()== 'tensorflow':
    K.clear_session()