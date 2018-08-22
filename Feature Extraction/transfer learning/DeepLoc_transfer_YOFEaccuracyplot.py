import numpy as np
import matplotlib.pyplot as plt

accuracysch=[]
accuracytf = []
import pickle
for i in [0,1,3,5,10,25,50,100,250,500]:
	sum=0
	# accLs=[]
	data=0
	avg=0
	acc={}
	with open ('directory to pkl folder/'+'number_training_per_class_'+str(i)+'.pkl','rb') as f: data=pickle.load(f)
	acc=data['testAllAcc']
	for key in acc: sum+=acc[key]
	# for key in acc: accLs.append(acc[key])
	avg=sum/5
	# median=np.median(np.array(accLs))
	accuracytf.append(avg)

for i in [0,1,3,5,10,25,50,100,250,500]:
	sum=0
	# accLs=[]
	data=0
	avg=0
	median=0
	acc={}
	with open ('directory to pkl folder/'+'number_training_per_class_'+str(i)+'.pkl','rb') as f: data=pickle.load(f)
	acc=data['testAllAcc']
	for key in acc: sum+=acc[key]
	# for key in acc: accLs.append(acc[key])
	avg=sum/5
	# median=np.median(np.array(accLs))
	accuracysch.append(avg)

accuracytf=np.array(accuracytf)
accuracytf.astype(float)
accuracysch=np.array(accuracysch)
accuracysch.astype(float)
steps = np.array([0,1,3.5,4.5,6,8,9.5,11,13,14.5])
steps.astype(int)


plt.plot(steps, accuracytf, 'ro-')
plt.plot(steps, accuracysch, 'bo-')
plt.xticks(steps,np.array([0,1,3,5,10,25,50,100,250,500]),fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.xlabel('Number of training samples per class',fontsize=20)
plt.legend(['transfer learning','training from scratch'], loc='lower right',fontsize=20)
plt.show()
