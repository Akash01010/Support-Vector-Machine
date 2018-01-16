import numpy as np
from PIL import Image
import os
import random
import math
from operator import itemgetter
import BoVW
import FDA
from sklearn.mixture import GaussianMixture
import scipy.stats
import matplotlib.pyplot as plt

dataTrain=[]
dataTest=[]
GMMparameters=[]
eigenVector=[]
mixtures=3
classLabel=[]
#calculates the prob of a point in the gaussian mixture of a particular class
def calcProb(dataPoint,gmmParam):
	prob=0
	for i in range(len(gmmParam[0])):
		prob+=gmmParam[0][i]*scipy.stats.norm(gmmParam[1][i], gmmParam[2][i]).pdf(dataPoint)
		#prob+=gmmParam[0][i]*(1/(math.pow(2*math.pi,dimension/2)*gmmParam[2][i]))*np.exp(-0.5*np.dot())
		#(1/(math.pow(2*math.pi,dimension/2)*param[2][k]))*np.exp(-0.5*np.dot(np.dot(np.transpose(np.subtract(data[n],param[1][k])),np.linalg.inv(param[2][k])),np.subtract(data[n],param[1][k]))))/(pow(2*math.pi,dimension)*math.fabs(np.linalg.det(param[2][k]))
	return prob

def classify(dataPoint):
	freq=[]
	for i in range(int(len(GMMparameters)/2)):
		x=np.dot(dataPoint,eigenVector[int(i)])
		a=calcProb(x,GMMparameters[2*i])
		b=calcProb(x,GMMparameters[2*i+1])
		if(a>=b):
			freq.append(classLabel[2*i])
		else:
			freq.append(classLabel[2*i+1])
		i+=1
	return max(freq,key=freq.count)


dataset = int(input("Which dataset you want to use: \n1)Linearly Separable\n2)Non-linearly Separable\n3)Image dataset\nEnter your choice: "))
mixtures = int(input("Enter the number of mixtures in GMM: "))

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

for i in range(len(dataTrain)):
	for j in range(i+1,len(dataTrain)):
		eigenVector.append(FDA.maxSeparationVec(dataTrain[i],dataTrain[j],dataset))


# uncomment this if you want to see the maximum sepatation line
# if dataset==1:
# 	xx=[dataTrain[0][i][0] for i in range(len(dataTrain[0]))]
# 	yy=[dataTrain[0][i][1] for i in range(len(dataTrain[0]))]
# 	plt.scatter(xx,yy,color='r')
# 	xx=[dataTrain[1][i][0] for i in range(len(dataTrain[1]))]
# 	yy=[dataTrain[1][i][1] for i in range(len(dataTrain[1]))]
# 	plt.scatter(xx,yy,color='b')
# 	xcoor = np.array(range(-25,35))
# 	a_ = eigenVector[0][1]/eigenVector[0][0]
# 	ycoor = eval('a_*xcoor')
# 	plt.plot(xcoor,ycoor)
# 	plt.plot(xcoor,ycoor,color='g')
# 	plt.title("Maximum Separation Line")
# 	plt.xlabel("X-Coordinate")
# 	plt.ylabel("Y-Coordinate")
# 	plt.xlim(-25,35)
# 	plt.ylim(-15,20)
# 	plt.show()

# 	xx=[dataTrain[0][i][0] for i in range(len(dataTrain[0]))]
# 	yy=[dataTrain[0][i][1] for i in range(len(dataTrain[0]))]
# 	plt.scatter(xx,yy,color='r')
# 	xx=[dataTrain[2][i][0] for i in range(len(dataTrain[2]))]
# 	yy=[dataTrain[2][i][1] for i in range(len(dataTrain[2]))]
# 	plt.scatter(xx,yy,color='g')
# 	xcoor = np.array(range(-25,35))
# 	a_ = eigenVector[1][1]/eigenVector[1][0]
# 	ycoor = eval('a_*xcoor')
# 	plt.plot(xcoor,ycoor)
# 	plt.plot(xcoor,ycoor,color='b')
# 	plt.title("Maximum Separation Line")
# 	plt.xlabel("X-Coordinate")
# 	plt.ylabel("Y-Coordinate")
# 	plt.xlim(-25,35)
# 	plt.ylim(-15,20)
# 	plt.show()

# 	xx=[dataTrain[1][i][0] for i in range(len(dataTrain[1]))]
# 	yy=[dataTrain[1][i][1] for i in range(len(dataTrain[1]))]
# 	plt.scatter(xx,yy,color='b')
# 	xx=[dataTrain[2][i][0] for i in range(len(dataTrain[2]))]
# 	yy=[dataTrain[2][i][1] for i in range(len(dataTrain[2]))]
# 	plt.scatter(xx,yy,color='g')
# 	xcoor = np.array(range(-25,35))
# 	a_ = eigenVector[2][1]/eigenVector[2][0]
# 	ycoor = eval('a_*xcoor')
# 	plt.plot(xcoor,ycoor)
# 	plt.plot(xcoor,ycoor,color='b')
# 	plt.title("Maximum Separation Line")
# 	plt.xlabel("X-Coordinate")
# 	plt.ylabel("Y-Coordinate")
# 	plt.xlim(-25,35)
# 	plt.ylim(-15,20)
# 	plt.show()

# if dataset==2:
# 	xx=[dataTrain[0][i][0] for i in range(len(dataTrain[0]))]
# 	yy=[dataTrain[0][i][1] for i in range(len(dataTrain[0]))]
# 	plt.scatter(xx,yy,color='r')
# 	xx=[dataTrain[1][i][0] for i in range(len(dataTrain[1]))]
# 	yy=[dataTrain[1][i][1] for i in range(len(dataTrain[1]))]
# 	plt.scatter(xx,yy,color='b')
# 	xcoor = np.array(range(-3,5))
# 	a_ = eigenVector[0][1]/eigenVector[0][0]
# 	ycoor = eval('a_*xcoor')
# 	plt.plot(xcoor,ycoor)
# 	plt.plot(xcoor,ycoor,color='g')
# 	plt.title("Maximum Separation Line")
# 	plt.xlabel("X-Coordinate")
# 	plt.ylabel("Y-Coordinate")
# 	plt.xlim(-3,5)
# 	plt.ylim(-2,2)
# 	plt.show()

# 	xx=[dataTrain[0][i][0] for i in range(len(dataTrain[0]))]
# 	yy=[dataTrain[0][i][1] for i in range(len(dataTrain[0]))]
# 	plt.scatter(xx,yy,color='r')
# 	xx=[dataTrain[2][i][0] for i in range(len(dataTrain[2]))]
# 	yy=[dataTrain[2][i][1] for i in range(len(dataTrain[2]))]
# 	plt.scatter(xx,yy,color='g')
# 	xcoor = np.array(range(-3,5))
# 	a_ = eigenVector[1][1]/eigenVector[1][0]
# 	ycoor = eval('a_*xcoor')
# 	plt.plot(xcoor,ycoor)
# 	plt.plot(xcoor,ycoor,color='b')
# 	plt.title("Maximum Separation Line")
# 	plt.xlabel("X-Coordinate")
# 	plt.ylabel("Y-Coordinate")
# 	plt.xlim(-3,5)
# 	plt.ylim(-2,2)
# 	plt.show()

# 	xx=[dataTrain[1][i][0] for i in range(len(dataTrain[1]))]
# 	yy=[dataTrain[1][i][1] for i in range(len(dataTrain[1]))]
# 	plt.scatter(xx,yy,color='b')
# 	xx=[dataTrain[2][i][0] for i in range(len(dataTrain[2]))]
# 	yy=[dataTrain[2][i][1] for i in range(len(dataTrain[2]))]
# 	plt.scatter(xx,yy,color='g')
# 	xcoor = np.array(range(-3,5))
# 	a_ = eigenVector[2][1]/eigenVector[2][0]
# 	ycoor = eval('a_*xcoor')
# 	plt.plot(xcoor,ycoor)
# 	plt.plot(xcoor,ycoor,color='b')
# 	plt.title("Maximum Separation Line")
# 	plt.xlabel("X-Coordinate")
# 	plt.ylabel("Y-Coordinate")
# 	plt.xlim(-3,5)
# 	plt.ylim(-2,2)
# 	plt.show()

ReducedFeatureVectors_TrainData=[]
count=0
for i in range(len(dataTrain)):
	for j in range(i+1,len(dataTrain)):
		data_ = [0 for k in range(len(dataTrain[i]))]
		for k in range(len(dataTrain[i])):
			data_[k] = np.dot(dataTrain[i][k],eigenVector[count])
		ReducedFeatureVectors_TrainData.append(data_)
		data_=[0 for k in range(len(dataTrain[j]))]
		for k in range(len(dataTrain[j])):
			data_[k] = np.dot(dataTrain[j][k],eigenVector[count])
		ReducedFeatureVectors_TrainData.append(data_)
		count+=1
dataTrain=[]


# uncomment this if you want to see the projected data on maximum separation line
# if dataset==1:
# 	yy=[0 for i in range(len(ReducedFeatureVectors_TrainData[0]))]
# 	plt.scatter(ReducedFeatureVectors_TrainData[0],yy,color='r')
# 	yy=[0 for i in range(len(ReducedFeatureVectors_TrainData[1]))]
# 	plt.scatter(ReducedFeatureVectors_TrainData[1],yy,color='b')
# 	plt.title("Data projected on direction of maximum sepatation")
# 	plt.xlabel("Max-Separating Line --->")
# 	# plt.ylabel("")
# 	plt.xlim(-25,15)
# 	plt.ylim(-15,20)
# 	plt.show()

# 	yy=[0 for i in range(len(ReducedFeatureVectors_TrainData[0]))]
# 	plt.scatter(ReducedFeatureVectors_TrainData[0],yy,color='r')
# 	yy=[0 for i in range(len(ReducedFeatureVectors_TrainData[2]))]
# 	plt.scatter(ReducedFeatureVectors_TrainData[1],yy,color='g')
# 	plt.title("Data projected on direction of maximum sepatation")
# 	plt.xlabel("Max-Separating Line --->")
# 	# plt.ylabel("")
# 	plt.xlim(-25,15)
# 	plt.ylim(-15,20)
# 	plt.show()

# 	yy=[0 for i in range(len(ReducedFeatureVectors_TrainData[1]))]
# 	plt.scatter(ReducedFeatureVectors_TrainData[0],yy,color='b')
# 	yy=[0 for i in range(len(ReducedFeatureVectors_TrainData[2]))]
# 	plt.scatter(ReducedFeatureVectors_TrainData[1],yy,color='g')
# 	plt.title("Data projected on direction of maximum sepatation")
# 	plt.xlabel("Max-Separating Line --->")
# 	# plt.ylabel("")
# 	plt.xlim(-25,15)
# 	plt.ylim(-15,20)
# 	plt.show()


# if dataset==2:
# 	yy=[0 for i in range(len(ReducedFeatureVectors_TrainData[0]))]
# 	plt.scatter(ReducedFeatureVectors_TrainData[0],yy,color='r')
# 	yy=[0 for i in range(len(ReducedFeatureVectors_TrainData[1]))]
# 	plt.scatter(ReducedFeatureVectors_TrainData[1],yy,color='b')
# 	plt.title("Data projected on direction of maximum sepatation")
# 	plt.xlabel("Max-Separating Line --->")
# 	# plt.ylabel("")
# 	plt.xlim(-8,5)
# 	plt.ylim(-2,2)
# 	plt.show()

# 	yy=[0 for i in range(len(ReducedFeatureVectors_TrainData[2]))]
# 	plt.scatter(ReducedFeatureVectors_TrainData[2],yy,color='r')
# 	yy=[0 for i in range(len(ReducedFeatureVectors_TrainData[3]))]
# 	plt.scatter(ReducedFeatureVectors_TrainData[3],yy,color='g')
# 	plt.title("Data projected on direction of maximum sepatation")
# 	plt.xlabel("Max-Separating Line --->")
# 	# plt.ylabel("")
# 	plt.xlim(-8,5)
# 	plt.ylim(-2,2)
# 	plt.show()

# 	yy=[0 for i in range(len(ReducedFeatureVectors_TrainData[4]))]
# 	plt.scatter(ReducedFeatureVectors_TrainData[4],yy,color='b')
# 	yy=[0 for i in range(len(ReducedFeatureVectors_TrainData[5]))]
# 	plt.scatter(ReducedFeatureVectors_TrainData[5],yy,color='g')
# 	plt.title("Data projected on direction of maximum sepatation")
# 	plt.xlabel("Max-Separating Line --->")
# 	# plt.ylabel("")
# 	plt.xlim(-8,5)
# 	plt.ylim(-2,2)
# 	plt.show()



for i in range(len(dataTest)):
	for j in range(i+1,len(dataTest)):
		classLabel.append(i)
		classLabel.append(j)

for i in range(len(ReducedFeatureVectors_TrainData)):
	gmix=GaussianMixture(n_components=mixtures, covariance_type='full')
	gmix.fit(np.asmatrix(ReducedFeatureVectors_TrainData[i]).T)

	GMMparameters.append([gmix.weights_,gmix.means_,gmix.covariances_])


confusionMatrix = [[0 for j in range(len(dataTest))] for i in range(len(dataTest))]

for i in range(len(dataTest)):
	for j in range(len(dataTest[i])):
		confusionMatrix[classify(dataTest[i][j])][i]+=1

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
