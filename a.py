import numpy as np
from PIL import Image
import os
import random
import math
from operator import itemgetter
import BoVW
#import GMM
import PCA
from sklearn.mixture import GaussianMixture
import scipy.stats

dataTrain=[]
dataTest=[]
GMMparameters=[]
mixtures=3
ReducedDimension=7

#calculates the prob of a point in the gaussian mixture of a particular class
def calcProb(dataPoint,gmmParam):
	prob=0
	dimension=len(dataPoint)
	for i in range(len(gmmParam[0])):
		# print(dataPoint)
		# print(gmmParam[1][i])
		# print(np.dot(np.dot(np.subtract(dataPoint,gmmParam[1][i]),np.linalg.inv(gmmParam[2][i])),np.subtract(dataPoint,gmmParam[1][i])))
		# prob+=gmmParam[0][i]*scipy.stats.multivariate_normal(gmmParam[1][i],gmmParam[2][i]).pdf(dataPoint)
		prob+=gmmParam[0][i]*(1/(math.pow(2*math.pi,
			dimension/2)*np.linalg.det(gmmParam[2][i])))*np.exp(-0.5*np.dot(np.dot(np.subtract(np.asarray(dataPoint),
				gmmParam[1][i]),
			np.linalg.inv(gmmParam[2][i])),
			np.subtract(dataPoint,
				np.asarray(gmmParam[1][i]))))
		#(1/(math.pow(2*math.pi,dimension/2)*param[2][k]))*np.exp(-0.5*np.dot(np.dot(np.transpose(np.subtract(data[n],param[1][k])),np.linalg.inv(param[2][k])),np.subtract(data[n],param[1][k]))))/(pow(2*math.pi,dimension)*math.fabs(np.linalg.det(param[2][k]))
	return prob

def classify(dataPoint):
	a=calcProb(dataPoint,GMMparameters[0])
	b=calcProb(dataPoint,GMMparameters[1])
	c=calcProb(dataPoint,GMMparameters[2])

	if((a>=b) and (a>=c)):
		return 0
	elif(b>=c):
		return 1
	else:
		return 2

#pdb.settrace()
mixtures = int(input("Enter the number of mixtures in GMM: "))
ReducedDimension = int(input("Enter the dimension of the reduced vector(less than 64): "))

folder ="D:\\Sem 5\\CS 669\\SVM\\image\\Train"

#for foldername in os.listdir(folder):
#6(TrainClass1, ..., TestClass1, ...)*50(no of examples assumng same)* 64
BoVWFeatureVectors_allData = BoVW.generateFeatureVectors(folder)

BoVWFeatureVectors_TrainData=[]
for i in range(int(len(BoVWFeatureVectors_allData)/2)):
    BoVWFeatureVectors_oneClass=[]
    for j in range(len(BoVWFeatureVectors_allData[i])):
        BoVWFeatureVectors_oneClass.append(BoVWFeatureVectors_allData[i][j])
    BoVWFeatureVectors_TrainData.append(BoVWFeatureVectors_oneClass)

BoVWFeatureVectors_TestData=[]
for i in range(int(len(BoVWFeatureVectors_allData)/2),len(BoVWFeatureVectors_allData)):
    BoVWFeatureVectors_oneClass=[]
    for j in range(len(BoVWFeatureVectors_allData[i])):
        BoVWFeatureVectors_oneClass.append(BoVWFeatureVectors_allData[i][j])

    BoVWFeatureVectors_TestData.append(BoVWFeatureVectors_oneClass)

BoVWFeatureVectors_allData=[]
BoVWFeatureVectors_oneClass=[]

eigenVectors = PCA.generateEigenVec(BoVWFeatureVectors_TrainData,ReducedDimension)

#reducing Train data
ReducedFeatureVectors_TrainData=[]
for i in range(len(BoVWFeatureVectors_TrainData)):
	ReducedFeatureVectors_oneClass=[]
	for j in range(len(BoVWFeatureVectors_TrainData[i])):
		reducedVec=[0 for k in range(ReducedDimension)]
		for k in range(ReducedDimension):
			reducedVec[k]=np.dot(BoVWFeatureVectors_TrainData[i][j],eigenVectors[k])
		ReducedFeatureVectors_oneClass.append(reducedVec)
	ReducedFeatureVectors_TrainData.append(ReducedFeatureVectors_oneClass)

BoVWFeatureVectors_TrainData=[]

#reducing Test data
ReducedFeatureVectors_TestData=[]
for i in range(len(BoVWFeatureVectors_TestData)):
	ReducedFeatureVectors_oneClass=[]
	for j in range(len(BoVWFeatureVectors_TestData[i])):
		reducedVec=[0 for k in range(ReducedDimension)]
		for k in range(ReducedDimension):
			reducedVec[k]=np.dot(BoVWFeatureVectors_TestData[i][j],eigenVectors[k])
		ReducedFeatureVectors_oneClass.append(reducedVec)
	ReducedFeatureVectors_TestData.append(ReducedFeatureVectors_oneClass)

BoVWFeatureVectors_TestData=[]


for i in range(len(ReducedFeatureVectors_TrainData)):
	gmix=GaussianMixture(n_components=mixtures, covariance_type='full')
	gmix.fit(ReducedFeatureVectors_TrainData[i])

	GMMparameters.append([gmix.weights_,gmix.means_,gmix.covariances_])

dataTest=ReducedFeatureVectors_TestData
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
