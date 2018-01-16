import numpy as np
from PIL import Image
import os
import random
import math
from operator import itemgetter
import BoVW

def responsibilityMatrix(data,param,mixtures):
	N=len(data)
	dimension=len(data[0])
	responsibility=[[0 for j in range(mixtures)] for i in range(n)]
	for n in range(N):
		for k in range(mixtures):
			responsibility[n][k] = (param[0][k]*(1/(math.pow(2*math.pi,dimension/2)*param[2][k]))*np.exp(-0.5*np.dot(np.dot(np.transpose(np.subtract(data[n],param[1][k])),np.linalg.inv(param[2][k])),np.subtract(data[n],param[1][k]))))/(pow(2*math.pi,dimension)*math.fabs(np.linalg.det(param[2][k])))
		temp=0
		for k in range(mixtures):
			temp+=responsibility[n][k]
		for k in range(mixtures):
			responsibility[n][k]/=temp
	return responsibility

def logLikelihood(data,param):
	N=len(data)
	K=len(param[0])
	dimension=len(data[0])
	prob=0
	for n in range(N):
		temp=0
		for k in range(K):
			temp+=(param[0][k]*np.exp(-0.5*(1/(math.pow(2*math.pi,dimension/2)*param[2][k]))*np.dot(np.dot(np.transpose(np.subtract(data[n]-param[1][k])),np.linalg.inv(param[2][k])),np.subtract(data[n]-param[1][k]))))/(pow(2*math.pi,dimension)*math.fabs(np.linalg.det(param[2][k])))
		prob+=math.log10(temp)
	return prob


def calcDist(x,y):
	distance=0
	z=len(x)
	for i in range(z):
		distance+=((x[i]-y[i])**2)
	return math.sqrt(distance)

#assuming that number of datapoints we have are greater than number of clusters we want to form
#dataMatrix is a 2D matrix
def K_MeansClustering(dataMatrix,K):
	
	dimension = len(dataMatrix[0])
	#	Assigning random means to the K clusters...
	tempClusterMean=[[0 for i in range(dimension)] for i in range(K)]
	#randomKMeans=random.sample(range(0,N),K%N)
	for i in range(K):
		for j in range(dimension):
			tempClusterMean[i][j]=dataMatrix[i%len(dataMatrix)][j]

	#	Dividing the data of this class to K clusters...
	#symbol=[[[-1 for k in range(len(dataMatrix[i][j]))] for j in range(len(dataMatrix[i]))] for i in range(len(dataMatrix))]
	tempClusters=[[] for i in range(K)]
	totDistance=0
	for i in range(len(dataMatrix)):
		minDist=np.inf
		minDistInd=0
		for l in range(K):
			Dist=calcDist(dataMatrix[i],tempClusterMean[l])
			if Dist<minDist:
				minDist=Dist
				minDistInd=l
		tempClusters[minDistInd].append(dataMatrix[i])
		totDistance+=minDist

	#	Re-evaluating centres until the energy of changes becomes insignificant...
	energy=100
	while energy>60:
		tempClusterMean=[[0 for i in range(dimension)] for i in range(K)]
		for i in range(K):
			for j in range(len(tempClusters[i])):
				for k in range(dimension):
					tempClusterMean[i][k]+=tempClusters[i][j][k]
			for k in range(dimension):
				if len(tempClusters[i])==0:
					#you can do something else here too
					for l in range(dimension):
						tempClusterMean[i][l]=(tempClusterMean[0][l]+tempClusterMean[K][l])/2
					break;
				else:
					tempClusterMean[i][k]/=len(tempClusters[i])
		tempClusters=[[] for i in range(K)]
		newTotDistance=0

		for i in range(len(dataMatrix)):
			minDist=np.inf
			minDistInd=0
			for l in range(K):
				Dist=calcDist(dataMatrix[i],tempClusterMean[l])
				if Dist<minDist:
					minDist=Dist
					minDistInd=l
			tempClusters[minDistInd].append(dataMatrix[i])
			newTotDistance+=minDist

		energy=math.fabs(totDistance-newTotDistance)
		totDistance=newTotDistance
		#print("KNN",energy,len(tempClusters[0]),len(tempClusters[1]),len(tempClusters[2]))
	
	#calculate variance,pie and then return tuple
	tempClusterPie = [0 for i in range(K)]
	for i in range(K):
		tempClusterPie[i]=(len(tempClusters[i])/len(data))

	tempClusterVariance=[]
	for i in range(K):
		temp=[[0 for j in range(dimension)] for k in range(dimension)]
		for j in range(len(tempClusters[i])):
			np.add(temp,np.dot(np.subtract(tempClusters[i][j],tempClusterMean[i]),np.transpose(np.subtract(tempClusters[i][j],tempClusterMean[i]))))
		np.divide(temp,len(tempClusters[i]))
		tempClusterVariance.append(temp)
	return [tempClusterPie, tempClusterMean, tempClusterVariance]


#data is a 2D matrix
def parameters(data,mixtures):
	N=len(data)
	dimension=len(data[0])
	param_old = K_MeansClustering(data,mixtures)
	param_new=[]
	p_new=0
	p_old=logLikelihood(data,param_old)

	error=100
	while (error>1):
		param_new = []
		responsibility = responsibilityMatrix(data,param_old,mixtures)
		
		Dr=[0 for x in range(mixtures)]
		for i in range(mixtures):
			for j in range(N):
				Dr[i]+=responsibility[i][j]

		tempPie=[0 for x in range(mixtures)]
		tempMean=[]
		tempVariance=[]

		#re-estimating pie
		for i in range(mixtures):
			tempPie[i]=Dr[i]/N

		#re-estimating mean
		for i in range(mixtures):
			temp=[0 for j in range(dimension)]
			for n in range(N):
				np.add(temp,responsibility[n][i]*np.array(data[n]))
			temp=np.array(temp)/Dr[i]
			tempMean.append(temp)
		
		#re-estimating covarianceMatrix
		for i in range(mixtures):
			temp=[[0 for j in range(dimension)] for k in range(dimension)]
			for j in range(N):
				np.add(temp,responsibility[j][i]*np.dot(np.subtract(data[j]-param_old[i]),np.transpose(np.subtract(data[j]-param_old[i]))))
			temp=np.array(temp)/Dr[i]
			tempVariance.append(temp)

		param_new=[tempPie, tempMean,tempVariance]
		p_new=logLikelihood(data,param_new)
		error=fabs(p_new- p_old)
		p_old=p_new

	return param_new

