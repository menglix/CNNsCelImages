import numpy as np
import matplotlib.pyplot as plt
import preprocess_images as procIm
import pickle
import h5py
from sklearn.metrics import confusion_matrix
import itertools


dataBaseDir='~/DeepLoc_full_datasets/datasets/'
testSetPath = dataBaseDir+'Schuldiner_test_set.hdf5'
testH5 = h5py.File(testSetPath,'r')
testdata=testH5['Index1']
testdata=np.array(testdata)
testdata.astype(np.float32)


sumpred=0
data=0
label=[]
with open ('number_training_per_class_'+str(100)+'.pkl','rb') as f: data=pickle.load(f)
pred=data['testAllPred']
for key in pred: sumpred+=pred[key]
avgpred=sumpred/5


ind = [4,1,2,3,0,5,6,7,8,9,10]
ind2=np.argsort(ind)

testdata=testdata[:,ind2]
#testdata=np.delete(testdata,(-2),axis=1)
avgpred=avgpred[:,ind2]
#avgpred=np.delete(avgpred,(-2),axis=1)


label=np.argmax(testdata,1)
y_pred=np.argmax(avgpred,1)
cnf_matrix = confusion_matrix(label, y_pred)

np.set_printoptions(precision=2)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

class_names = ['bud','bud_neck', 'cell_periphery', 'cytosol', 'ER', 'mitochondria', 'nuclear_periphery', 'nucleus', 'punctate','vacuole','vacuole_membrane']



plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

#for i in range(19): print((label[:,i]==1).sum())