# generate vgg 19 features

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation, BatchNormalization
from keras import applications
from keras.utils import np_utils
from keras.models import model_from_json
from keras import backend as K
K.set_image_dim_ordering('tf')
import pickle

img_width, img_height = 64, 64

train_data_dir = '/space/data/deepyeast/training'
validation_data_dir = '/space/data/deepyeast/validation'
train_feature_dir = '/space/data/deepyeastmodel/transferlearning/deepyeast_bottleneck_features_train_vgg19.npy'
validation_feature_dir = '/space/data/deepyeastmodel/transferlearning/deepyeast_bottleneck_features_val_vgg19.npy'
output_dir = "/home/menglix/result/deepyeast/transferlearning"
nb_train_samples = 65000
nb_validation_samples = 12500
epochs = 100
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



def save_bottlebeck_features():
    datagen = ImageDataGenerator(featurewise_center=True)

    # build the VGG16 network
    model = applications.VGG19(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    np.save(open(train_feature_dir, 'wb'),
            bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save(open(validation_feature_dir, 'wb'),
            bottleneck_features_validation)







def train_top_model():
	train_data=np.load(train_feature_dir)
	validation_data = np.load(validation_feature_dir)
	model = Sequential()
	model.add(Flatten(input_shape=train_data.shape[1:]))  # this converts our 3D feature maps to 1D feature vectors
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(512))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(12))
	model.add(BatchNormalization())
	model.add(Activation('softmax'))
	model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
	history = model.fit(train_data, train_label,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_label))
	with open(output_dir+'/bn23fc_hist'+'.pkl', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
	# import matplotlib
	# matplotlib.use('agg')
	# from matplotlib import pyplot as plt 
	# plt.plot(history.history['acc'])
	# plt.plot(history.history['val_acc'])
	# plt.title('model accuracy')
	# plt.ylabel('accuracy')
	# plt.xlabel('epoch')
	# plt.legend(['train', 'validation'], loc='upper left')
	# plt.savefig('/home/merganser/menglix/Desktop/result/dyaccuracytfvgg19.pdf')
	# plt.clf()
	# plt.plot(history.history['loss'])
	# plt.plot(history.history['val_loss'])
	# plt.title('model loss')
	# plt.ylabel('loss')
	# plt.xlabel('epoch')
	# plt.legend(['train', 'validation'], loc='upper left')
	# plt.savefig('/home/merganser/menglix/Desktop/result/dylosstfvgg19.pdf')
	


save_bottlebeck_features()
train_top_model()
    
