import os
import numpy as np
# import cPickle as pickle
# from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array, load_img
import pdb

def loadData():
	'''get the image data transformed into numpy array without much preprocessing
	'''
	# get the class name as a list
	a=sorted(os.listdir('/space/data/deepyeast/test'))
	imgType = ".png"

	train_img_path=['/space/data/deepyeast/training/'+i+'/'+img for i in a for img in os.listdir('/space/data/deepyeast/training/'+i) if img.endswith(".png")]
	validation_img_path=['/space/data/deepyeast/validation/'+i+'/'+img for i in a for img in os.listdir('/space/data/deepyeast/validation/'+i) if img.endswith(".png")]
	test_img_path=['/space/data/deepyeast/test/'+i+'/'+img for i in a for img in os.listdir('/space/data/deepyeast/test/'+i) if img.endswith(".png")]

	def loadImgFromFile(dirPath):
		""" Reads an image from file and returns a numpy matrix with shape (1, 3, width, height)
		@param {string} dirPath: directory that patch file is located
		@param {string} patch: file of image patch
		"""
		img = load_img(dirPath) # this is a PIL image
		x = img_to_array(img) # this is a Numpy array with shape (3, 300, 300)
		return x 

	# Load image matrices for each file
	X_train = np.array([loadImgFromFile(path) for path in train_img_path]) # train
	X_val = np.array([loadImgFromFile(path) for path in validation_img_path]) # validation
	X_test = np.array([loadImgFromFile(path) for path in test_img_path]) # test

	# X = np.vstack((X_ROI, X_Normal, X_White))
	# y = [1]*len(X_ROI) + [0]*len(X_Normal) + [2]*len(X_White)
	# X_train, X_test, y_train, y_test= train_test_split(X, y, train_size=0.8, random_state=42, stratify=y) # Split into train and test
	# X_train, X_val, y_train, y_val= train_test_split(X_train, y_train, train_size=0.9, random_state=42, stratify=y_train) # Split into train and validation
	print('Train shape: ', X_train.shape)
	print('Val shape: ', X_val.shape)
	print('Test shape: ', X_test.shape)

	return X_train, X_val, X_test

# pklFilename = 'patchesData.pkl'
# X_train, X_val, X_test, y_train, y_val, y_test = loadData()
# pickle.dump((X_train, X_val, X_test, y_train, y_val, y_test), open(pklFilename, 'wb'))
# print("Pickle " + pklFilename + " saved.")

