import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from keras import backend as K
K.set_image_dim_ordering('tf')


img_width, img_height = 64, 64
# specify where the image data is stored
train_data_dir = '/space/data/deepyeast/training'
validation_data_dir = '/space/data/deepyeast/validation'
# store the trained model as .hdf5 file
filepath="/space/data/deepyeastmodel/weights.{epoch:02d}-{val_acc:.3f}.hdf5"
nb_train_samples = 65000
nb_validation_samples = 12500
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

# define model structure as illustrated in Figure 2 of our paper
# model specification follows the method-implementation part in the paper
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

sgd = SGD(lr=0.1, decay=0.04, momentum=0.9, nesterov=False)
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])


			  
train_datagen = ImageDataGenerator(
    featurewise_center=True)
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


# the off-the-shelf method for monitoring and storing intermediate keras models
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, mode='auto',period=50)
callbacks_list = [checkpoint]

history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
	validation_steps=nb_validation_samples // batch_size, callbacks=callbacks_list)
	

# generate accuracy and loss plot     
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt 
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('/home/menglix/result/deepyeast/dyscratchaccuracy.pdf')
plt.clf()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('/home/menglix/result/deepyeast/dyscratchloss.pdf')