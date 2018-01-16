import numpy as np
import matplotlib.pyplot as plt
#input n*64 matrix data
#output n*dimension matrix
def generateEigenVec(data_,reducedDimension):
	data = []
	for i in range(len(data_)):
		for j in range(len(data_[i])):
			data.append(data_[i][j])
	data_=[]
	mean = np.mean(data,axis=0,dtype=np.float64)

	# for i in range(len(data)):
	# 	np.subtract(data[i],mean)

	sigma=np.cov(np.array(data).T)

	#these are column vectors and the values will be rounded off
	eigenValues,eigenVectors = np.linalg.eigh(sigma)

	sortedEigenVectors = [x for _,x in sorted(zip(eigenValues,np.array(eigenVectors).T),reverse=True)]

	# Uncomment this section if you want plot for eigenValues
	# sortedEigenValues = sorted(eigenValues,reverse=True)
	
	# ComponentNo= [i+1 for i in range(len(data[0]))]

	# plt.plot(ComponentNo,sortedEigenValues,'-o')
	# plt.title("Eigen Values for Image Dataset")
	# plt.xlabel("Component No")
	# plt.ylabel("Eigen Value")
	# plt.show()

	return sortedEigenVectors

