import numpy as np

def getline(dataPositive,dataNegative):
	#initial values of parameters for perceptron
	coefficientsLine=[1 for i in range(len(dataPositive[0]))]
	constant=-1
	neta=0.5

	misclassifiedPositive=misclassifiedNegative = 10
	while((misclassifiedPositive+misclassifiedNegative)!=0):
		misclassifiedPositive=misclassifiedNegative=0
		errorTerm=[0 for i in range(len(dataPositive[0]))]
		for i in range(len(dataPositive)):
			x=np.dot(coefficientsLine,dataPositive[i])
			x+=constant
			if x<0:
				errorTerm=np.add(errorTerm,dataPositive[i])
				misclassifiedPositive+=1

		for i in range(len(dataNegative)):
			x=np.dot(coefficientsLine,dataNegative[i])
			x+=constant
			if x>0:
				errorTerm = np.subtract(errorTerm,dataNegative[i])
				misclassifiedNegative+=1

		errorTerm=neta*np.asmatrix(errorTerm)
		coefficientsLine = np.add(coefficientsLine,errorTerm)
		constant+=(neta*(misclassifiedPositive - misclassifiedNegative) )
		# print("misclassifiedPositive",misclassifiedPositive)
		# print("misclassifiedNegative",misclassifiedNegative)

	return [constant,coefficientsLine]

