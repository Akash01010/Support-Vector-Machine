import numpy as np
from PIL import Image
import os
import random
import math
from operator import itemgetter
import BoVW
from sklearn import svm
import matplotlib.pyplot as plt
from matplotlib import style
from mlxtend.plotting import plot_decision_regions
from sklearn.mixture import GaussianMixture

style.use("ggplot")

dataTrain=[]
dataTest=[]
clf=[]


def classify(dataPoint):
	freq=[]
	for i in range(len(clf)):
		freq.append(clf[i].predict(np.array(dataPoint).reshape(1,-1))[0])
	return max(freq,key=freq.count)



dataset = int(input("Which dataset you want to use: \n1)Linearly Separable\n2)Non-linearly Separable\n3)Image dataset\nEnter your choice: "))
kernel = int(input("Which kernel you want to use:\n1)Linear\n2)Polynomial\n3)Gaussian\nEnter your choice: "))

if dataset==1:
	folder ="D:\\Sem 5\\CS 669\\SVM\\Linearly_separable"

	for filename in os.listdir(os.path.join(folder,"Train")):
		data_oneclass=[]
		with open(os.path.join(os.path.join(folder,"Train"),filename)) as infile:
			for line in infile:
				number_strings=line.split()
				numbers=[float(n) for n in number_strings]
				data_oneclass.append(numbers)
		dataTrain.append(data_oneclass)

	for filename in os.listdir(os.path.join(folder,"Test")):
		data_oneclass=[]
		with open(os.path.join(os.path.join(folder,"Test"),filename)) as infile:
			for line in infile:
				number_strings=line.split()
				numbers=[float(n) for n in number_strings]
				data_oneclass.append(numbers)
		dataTest.append(data_oneclass)

	data_oneclass=[]
	numbers=[]

elif dataset==2:
	folder ="D:\\Sem 5\\CS 669\\SVM\\Non-linearly_separable"

	for filename in os.listdir(os.path.join(folder,"Train")):
		data_oneclass=[]
		with open(os.path.join(os.path.join(folder,"Train"),filename)) as infile:
			for line in infile:
				number_strings=line.split()
				numbers=[float(n) for n in number_strings]
				data_oneclass.append(numbers)
		dataTrain.append(data_oneclass)

	for filename in os.listdir(os.path.join(folder,"Test")):
		data_oneclass=[]
		with open(os.path.join(os.path.join(folder,"Test"),filename)) as infile:
			for line in infile:
				number_strings=line.split()
				numbers=[float(n) for n in number_strings]
				data_oneclass.append(numbers)
		dataTest.append(data_oneclass)

	data_oneclass=[]
	numbers=[]

elif dataset==3:
	folder ="D:\\Sem 5\\CS 669\\SVM\\image\\Train"

	#for foldername in os.listdir(folder):
	#6(TrainClass1, ..., TestClass1, ...)*50(no of examples assumng same)* 64
	BoVWFeatureVectors_allData = BoVW.generateFeatureVectors(folder)

	BoVWFeatureVectors_TrainData=[]
	for i in range(int(len(BoVWFeatureVectors_allData)/2)):
	    BoVWFeatureVectors_TrainData.append(BoVWFeatureVectors_allData[i])

	BoVWFeatureVectors_TestData=[]
	for i in range(int(len(BoVWFeatureVectors_allData)/2),len(BoVWFeatureVectors_allData)):
	    BoVWFeatureVectors_TestData.append(BoVWFeatureVectors_allData[i])

	BoVWFeatureVectors_allData=[]
	BoVWFeatureVectors_oneClass=[]
	dataTrain=BoVWFeatureVectors_TrainData
	dataTest=BoVWFeatureVectors_TestData

dataTrain_allClassPairs=[]
label=[]
for i in range(len(dataTrain)):
	for j in range(i+1,len(dataTrain)):
		dataTrain_twoClass=[]
		labelPair=[]
		for k in range(len(dataTrain[i])):
			dataTrain_twoClass.append(dataTrain[i][k])
			labelPair.append(i)
		for k in range(len(dataTrain[j])):
			dataTrain_twoClass.append(dataTrain[j][k])
			labelPair.append(j)
		dataTrain_allClassPairs.append(dataTrain_twoClass)
		label.append(labelPair)


if kernel==1:
	for i in range(len(dataTrain_allClassPairs)):
		clfTemp=svm.SVC(kernel='linear', C = 1.0)
		clfTemp.fit(dataTrain_allClassPairs[i],label[i])
		clf.append(clfTemp)
		# plot_decision_regions(np.asarray(dataTrain_allClassPairs[i]),np.asarray(label[i]), clf=clf[i],
  #                     res=0.02, legend=2)
		# plt.show()

elif kernel==2:
	for i in range(len(dataTrain_allClassPairs)):
		clfTemp=svm.SVC(kernel='poly', degree=3, C = 1.0)
		clfTemp.fit(dataTrain_allClassPairs[i],label[i])
		clf.append(clfTemp)
		# plot_decision_regions(np.asarray(dataTrain_allClassPairs[i]),np.asarray(label[i]), clf=clf[i],
  #                     res=0.02, legend=2)
		# plt.show()

elif kernel==3:
	for i in range(len(dataTrain_allClassPairs)):
		clfTemp=svm.SVC(kernel='rbf', gamma=1.0, C = 1.0)
		clfTemp.fit(dataTrain_allClassPairs[i],label[i])
		clf.append(clfTemp)
		# plot_decision_regions(np.asarray(dataTrain_allClassPairs[i]),np.asarray(label[i]), clf=clf[i],
  #                     res=0.02, legend=2)
		# plt.show()

allData=[]
allLabel=[]
for i in range(len(dataTrain)):
	for j in range(len(dataTrain[i])):
		allData.append(dataTrain[i][j])
		allLabel.append(i)


# classifier=GaussianMixture(n_components=3, covariance_type='full')
# classifier.fit(np.asarray(allData),np.asarray(allLabel))
# plot_decision_regions(np.asarray(allData),np.asarray(allLabel), clf=classifier,
#                       res=0.02, legend=2)
# plt.title("Decision Region Plot using Bayes Classifier")
# plt.xlabel("X-coordinate")
# plt.ylabel("Y-coordinate")
# plt.show()

# Uncomment this section if you want decision region plots for the selected dataset
classifier=svm.SVC(kernel='linear', C = 1.0)
classifier.fit(np.asarray(allData),np.asarray(allLabel))
plot_decision_regions(np.asarray(allData),np.asarray(allLabel), clf=classifier,
                      res=0.02, legend=2)
plt.title("Decision Region Plot using Linear Kernel")
plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")
plt.show()


classifier=svm.SVC(kernel='poly', degree=3, C = 1.0)
classifier.fit(np.asarray(allData),np.asarray(allLabel))
plot_decision_regions(np.asarray(allData),np.asarray(allLabel), clf=classifier,
                      res=0.02, legend=2)
plt.title("Decision Region Plot using Polynomial Kernel")
plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")
plt.show()


classifier=svm.SVC(kernel='rbf', gamma=1.0, C = 1.0)
classifier.fit(np.asarray(allData),np.asarray(allLabel))
plot_decision_regions(np.asarray(allData),np.asarray(allLabel), clf=classifier,
                      res=0.02, legend=2)
plt.title("Decision Region Plot using Gaussian Kernel")
plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")
plt.show()



# if dataset==1 or dataset==2:
# 	xmin=np.inf,xmax=-np.inf,ymin=np.inf,ymax=-np.inf
# 	for i in range(len(dataTrain)):
# 		for j in range(len(dataTrain[i])):
# 			if(dataTrain[i][j][0] < xmin):
# 				xmin=dataTrain[i][j][0]
# 			elif(dataTrain[i][j][0] > xmax):
# 				xmax=dataTrain[i][j][0]
# 			elif(dataTrain[i][j][1] < ymin):
# 				ymin=dataTrain[i][j][1]
# 			elif(dataTrain[i][j][1] > ymax):
# 				ymax=dataTrain[i][j][1]

# 	xmin-=1;ymin-=1;xmax+=1;ymax+=1

# plot_decision_regions(X, y, clf=clf[0],
#                       res=0.02, legend=2)
# plot_decision_regions(X, y, clf=clf[1],
#                       res=0.02, legend=2)







confusionMatrix = [[0 for j in range(len(dataTest))] for i in range(len(dataTest))]

for i in range(len(dataTest)):
	for j in range(len(dataTest[i])):
		confusionMatrix[classify(dataTest[i][j])][i]+=1
		# confusionMatrix[int(classifier.predict(np.array(dataTest[i][j]).reshape(1,-1)))][i]+=1

print(confusionMatrix)


Nr=Dr=0
for i in range(len(confusionMatrix)):
	for j in range(len(confusionMatrix[0])):
		if i==j:
			Nr+=confusionMatrix[i][j]
		Dr+=confusionMatrix[i][j]
print("Accuracy: ",Nr/Dr)

meanPrecision=0
meanRecall=0
meanFmeasure=0
x=y=z=0
for i in range(len(confusionMatrix)):
	Dr=0
	for j in range(len(confusionMatrix[i])):
		Dr+=confusionMatrix[i][j]
	if Dr!=0:
		x=confusionMatrix[i][i]/Dr
		meanPrecision+=x
		print("Precision for class ",i,": ",x)
	Dr=0
	for j in range(len(confusionMatrix)):
		Dr+=confusionMatrix[j][i]
	if Dr!=0:
		y=confusionMatrix[i][i]/Dr
		meanRecall+=y
		print("Recall for class ",i,": ",y)
	if (x+y)!=0:
		z=2*x*y/(x+y)
		meanFmeasure+=z
		print("F-measure for class ",i,"; ",z)

print("Mean Precision: ",meanPrecision/len(confusionMatrix))
print("Mean Recall: ",meanRecall/len(confusionMatrix))
print("Mean F-measure: ",meanFmeasure/len(confusionMatrix))

