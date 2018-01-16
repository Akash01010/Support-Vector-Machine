import numpy as np


def maxSeparationVec(dataPositive,dataNegative,d):

	mean1 = np.mean(dataPositive,axis=0,dtype=np.float64)
	mean2 = np.mean(dataNegative,axis=0,dtype=np.float64)

	sigmaPositive=np.cov(np.asmatrix(dataPositive).T)
	sigmaNegative=np.cov(np.asmatrix(dataNegative).T)


	scatterPositive = len(dataPositive)*(sigmaPositive)
	scatterNegative = len(dataNegative)*(sigmaNegative)

	diff = np.subtract(mean1,mean2)
	betweenClassScatterMatrix=[[0 for j in range(len(diff))] for i in range(len(diff))]
	for i in range(len(diff)):
		for j in range(len(diff)):
			betweenClassScatterMatrix[i][j]=diff[i]*diff[j]

	withinClassScatterMatrix = np.add(scatterPositive ,scatterNegative)
	if(d==3):
		eigenValues,eigenVectors = np.linalg.eigh(np.asmatrix(betweenClassScatterMatrix))
	else:
		eigenValues,eigenVectors = np.linalg.eigh(np.matmul(np.linalg.inv(np.asmatrix(withinClassScatterMatrix)),np.asmatrix(betweenClassScatterMatrix)))
	sortedEigenVectors = [x for _,x in sorted(zip(eigenValues,np.array(eigenVectors).T),reverse=True)]

	return 	sortedEigenVectors[0]