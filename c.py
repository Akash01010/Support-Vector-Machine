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
import Perceptron as Perceptron

dataTrain=[]
dataTest=[]
GMMparameters=[]
eigenVector=[]
mixtures=3
lineParamaters=[]
classLabel=[]

def classify(dataPoint):
	freq=[]
	for i in range(len(lineParamaters)):
		x=np.dot(lineParamaters[i][1],dataPoint)
		x+=lineParamaters[i][0]
		if(x>0):
			freq.append(classLabel[i][0])
		else:
			freq.append(classLabel[i][1])
	return max(freq,key=freq.count)


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

# for i in range(len(dataTrain[2])):
# 	plt.scatter(dataTrain[2][i][0],dataTrain[2][i][1])

plt.show()
for i in range(len(dataTrain)):
	for j in range(i+1,len(dataTrain)):
		classLabel.append([i,j])
		line=Perceptron.getline(dataTrain[i],dataTrain[j])
		lineParamaters.append(line)



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

