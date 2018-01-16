import numpy as np
from PIL import Image
import os
import random
import math



patchHeight = 64
patchWidth = 64
bins = 8
BoVW_VectorLen = 64

def calcVectorforPatch(img,xPnt,yPnt):
	vector_ = [0 for i in range(3*bins)]
	for i in range(patchHeight):
		for j in range(patchWidth):
			vector_[img[xPnt+i][yPnt+j][0]/32]+=1;
			vector_[8+img[xPnt+i][yPnt+j][1]/32]+=1;
			vector_[16+img[xPnt+i][yPnt+j][2]/32]+=1;
	return vector_


#create histograms of all the patches and store them into a file in folder output
def createHistograms(folder,filename,img, patchHeight, patchWidth, bins):
	folder=os.path.join(folder,"output")
	filename = os.path.splitext(filename)[0]
	filename+='.txt'
	x,y,z = img.shape
	xPnt=0
	with open(os.path.join(folder,filename), 'w') as outfile:
		for i in range(x/patchHeight):
			yPnt=0
			for j in range(y/patchWidth):
				vect = calcVectorforPatch(img,xPnt,j*patchWidth)
				outfile.write(vect)
				outfile.write("\n")
			xPnt+=patchHeight
	outfile.close()
		


#it should return the matrix containing mean vectors of all the K-clusters
def K_MeansClustering(filename):
	K=BoVW_VectorLen
	dimension=3*bins
	file=open(filename)
	data=[]
	for line in file:
		number_strings=line.split()
		numbers=[float(n) for n in number_strings]
		data.append(numbers)
	tempClass=np.array(data)
	N=len(tempClass)

	#	Assigning random means to the K clusters...
	tempClusterMean=[[0 for i in range(dimension)] for i in range(K)]
	randomKMeans=random.sample(range(N),k=K)
	for i in range(K):
		for j in range(dimension):
			tempClusterMean[i][j]=tempClass[randomKMeans[i]][j]

	#	Dividing the data of this class to K clusters...
	tempClusters=[[] for i in range(K)]
	totDistance=0
	energy=1
	for i in range(N):
		minDist=np.inf
		minDistInd=0
		for j in range(K):
			Dist=calcDist(tempClass[i],tempClusterMean[j])
			if Dist<minDist:
				minDist=Dist
				minDistInd=j
		tempClusters[minDistInd].append(tempClass[i])
		totDistance+=minDist
	
	#	Re-evaluating centres until the energy of changes becomes insignificant...
	while energy>1:
		tempClusterMean=[[0 for i in range(dimension)] for i in range(K)]
		for i in range(K):
			for j in range(len(tempClusters[i])):
				for k in range(dimension):
					tempClusterMean[i][k]+=tempClusters[i][j][k]
			for k in range(dimension):
				if(len(tempClusters[i])!=0):
					tempClusterMean[i][k]/=len(tempClusters[i])
		tempClusters=[[] for i in range(K)]
		newTotDistance=0
		for i in range(N):
			minDist=np.inf
			minDistInd=0
			for j in range(K):
				Dist=calcDist(tempClass[i],tempClusterMean[j])
				if Dist<minDist:
					minDist=Dist
					minDistInd=j
			tempClusters[minDistInd].append(tempClass[i])
			newTotDistance+=minDist
		energy=math.fabs(totDistance-newTotDistance);
		totDistance=newTotDistance;
	return tempClusterMean

def calcDist(x,y):
	distance=0
	for i in range(3*bins):
		distance+=(x[i]-y[i])**2
	return math.sqrt(distance)



def classify(patchVector,means):
	mini=np.inf
	ind = 0
	for i in range(len(means)):
		dist=calcDist(patchVector,means[i])
		if dist < mini:
			ind = i
			mini = dist
	return ind


#this function will take input a folder and output 2D array having 64 dimensional vector corresponding to each file in folder
def BagofVisualWorlds(folder):
	all_classfeatureVectors = []
	featureVectors=[]
	#concatinating all the files
	outfile=open(os.path.join(folder,"output.txt"), 'w')
	for foldername in os.listdir(folder):
		if os.path.splitext(foldername)[0]!="output" and os.path.splitext(foldername)[0]!="." and os.path.splitext(foldername)[0]!="..":
			for filename in os.listdir(os.path.join(folder,foldername)):
				if os.path.splitext(filename)[1] == ".txt":
					infile=open(os.path.join(os.path.join(folder,foldername),filename))
					for line in (infile):
						outfile.write(line)
					infile.close()
	outfile.close()

	#generating mean vectors from the concatinated file :p
	means = K_MeansClustering(os.path.join(folder,"output.txt"))

	for foldername in os.listdir(folder):
		if os.path.splitext(foldername)[0]!="output" and os.path.splitext(foldername)[0]!="." and os.path.splitext(foldername)[0]!="..":
			for filename in os.listdir(os.path.join(folder,foldername)):
				if os.path.splitext(filename)[1] == ".txt" and os.path.splitext(filename)[0]!="output":
					featureVector=[0 for i in range(BoVW_VectorLen)]
					with open(os.path.join(os.path.join(folder,foldername),filename)) as infile:
						for line in infile:
							number_strings=line.split()
							numbers=[float(n) for n in number_strings]
							clusterNum=classify(numbers,means)
							featureVector[clusterNum]+=1
					infile.close()
					featureVectors.append(featureVector)
			all_classfeatureVectors.append(featureVectors)
			featureVectors=[]
	folder=os.path.join(os.path.join(folder,".."),"Test")
	for foldername in os.listdir(folder):
		if os.path.splitext(foldername)[0]!="." and os.path.splitext(foldername)[0]!="..":
			for filename in os.listdir(os.path.join(folder,foldername)):
				if os.path.splitext(filename)[1] == ".txt":
					featureVector=[0 for i in range(BoVW_VectorLen)]
					with open(os.path.join(os.path.join(folder,foldername),filename)) as infile:
						for line in infile:
							number_strings=line.split()
							numbers=[float(n) for n in number_strings]
							clusterNum=classify(numbers,means)
							featureVector[clusterNum]+=1
					infile.close()
					featureVectors.append(featureVector)
			all_classfeatureVectors.append(featureVectors)
			featureVectors=[]


	return all_classfeatureVectors

	#File = open(os.path.join(folder,filename),"r")


#argument should be a folder name where all the images of a class are present
#functioin should return the matrix containing 64 dimension vector corresponding to each image file in a folder
def generateFeatureVectors(folder):
    # for filename in os.listdir(folder):
    #     img = Image.open(os.path.join(folder,filename))
    #     if img is not None:
    #         createHistograms(folder,filename,np.array(img),patchHeight,patchWidth,bins)
    # Test_folder =os.path.join(os.path.join(folder,".."),"Test")
    # for filename in os.listdir(Test_folder):
    #     img = Image.open(os.path.join(Test_folder,filename))
    #     if img is not None:
    #         createHistograms(Test_folder,filename,np.array(img),patchHeight,patchWidth,bins)
    return BagofVisualWorlds(folder)


#images=[];
#images=generateFeatureVectors("D:\Sem 5\CS 669\GMM\Data\group2\Train\coast")

#print(images[0][0][0])
#print(images[len(images)][len(images[len(images)])][len(images[len(images)][len(images[len(images)])])])
