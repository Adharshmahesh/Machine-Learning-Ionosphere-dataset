import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import sklearn
import sklearn.metrics as metrics


def load_data(path, header):
    df = pd.read_csv(path, header=header)
    return df

class Logistic_Regression:

	def __init__(self, w):
		self.w = w

	def sigmoidal(self, z):
		z = z.astype(float)
		return (1/(1+np.exp(-z)))

	
	def gradient(self, xtrain, ytrain):
		
		z = self.sigmoidal(np.transpose(self.w).dot(np.transpose(xtrain)))
		grad = np.transpose(xtrain).dot(np.transpose(ytrain - z))
		
		return grad
		

	def fit(self, xtrain, ytrain, lr=0.01, iter=50000, eps=0.1, normal = True):

				
		ytrain = ytrain[np.newaxis]
		numiter = 1
		cost = 0
		cost1 =list()
		const = 1e-5 #To prevent zero 
		for i in range(iter):
			
			
			cost = -1*(ytrain.dot(np.log(self.sigmoidal(np.transpose(w).dot(np.transpose(xtrain))).T+const)) + (1-ytrain).dot(np.log(1-self.sigmoidal(np.transpose(self.w).dot(np.transpose(xtrain))).T+const)))
			
			cost1.append(cost)

			g = self.gradient(xtrain,ytrain)
			
			self.w = self.w+(lr * g)
			
			if(np.linalg.norm(g)<eps):
				print("No of iterations is", numiter)
				break
			numiter = numiter + 1
		
		return self.w, numiter, cost1

	def predict(self, xtest, normal = True):
		
		pred = (self.sigmoidal((self.w.T.dot(xtest.T))))
		for i in range(len(pred)):
			for j in range(len(pred[i])):
				if pred[i][j]==0:
					pred[i][j] = 0
				elif pred[i][j] ==1:
					pred[i][j] = 1
				elif pred[i][j]<0.5:
					pred[i][j] = 0
				else:
					pred[i][j] = 1 
		
		
		return pred
		

	def calc_accuracy(self, ytest, pred):
		

		return np.mean(ytest == pred)
	

	def conf_matrix(self, ytest, pred):

		cm = np.zeros(shape = (2,2))

		for i in range(len(pred)):
			for j in range(len(pred[i])):
				if ytest[i] == 0:
					if pred[i][j] == 0:
							cm[1][1] += 1
					else:
						cm[1][0] += 1

				elif ytest[i] == 1:
					if pred[i][j] == 1:
						cm[0][0] += 1

					else:
						cm[0][1] += 1
		positive = cm[0][0] + cm[1][0]
		negative = cm[0][1] + cm[1][1]

		accuracy_cm = (cm[0][0] + cm[1][1]) / (positive + negative)
		#precision = cm[0][0] / (cm[0][0] + cm[0][1])
		#recall = cm[0][0] / positive
		#f_measure = (2*recall*precision)/ (recall + precision)
		
		return accuracy

	def crossvalidation(self, xtrain, ytain, k, alpha = 0.01, iter = 50000, eps = 0.01):

		size = int(len(xtrain)/k)
		cv_accuracy = 0
		cv_loss = []

		for i in range(k):

			valstart = i*size
			valend = valstart + size

			if i!=(k-1):
				valend = size

				xval = xtrain[:valend,:]
				yval = ytrain[:valend]

				kxtrain = xtrain[valend:,:]
				kytrain = ytrain[valend:]

			else:
		
				xval = xtrain[valstart:,:]
				yval = ytrain[valstart:]

				kxtrain = xtrain[:valstart,:]
				kytrain = ytrain[:valstart]

				kxtrain = np.concatenate((xtrain[:valstart,:],xtrain[valend:,:]), axis = 0)
				kytrain = np.concatenate((ytrain[:valstart],ytrain[valend:]))

			w_kfold, numiter_k, cv_loss = self.fit(kxtrain, kytrain, alpha, iter)
			
			predy = self.predict(xval)
			
			cv_accuracy = cv_accuracy+self.calc_accuracy(yval, predy)

		cv_accuracy = cv_accuracy / k

		return cv_accuracy

	

if __name__ == "__main__":
    # load the data from the file
	data = load_data("ionosphere.csv", None)

	for i in range(0,351):
		if data.loc[i,34] == 'g':
			data.loc[i,34] = 1
		else:
			data.loc[i,34] = 0
	a = np.ones((351,1),dtype = int)
	data.insert(0,0,a, True)

	#To check the accuracy after Feature Engineering
	#data = data.drop(columns = [1,3,4,19,21,23,31])
	x=data.iloc[:,:-1]
	y=data.iloc[:,-1]
	
	
	train_data = data.sample(frac = 0.8)

	xtrain = np.array(train_data.iloc[:,:-1])
	ytrain = np.array(train_data.iloc[:,-1])	
	test_data = data.drop(train_data.index)
	xtest = np.array((test_data.iloc[:,:-1]))
	ytest = np.array((test_data.iloc[:,-1]))
	
	
	w = np.array(np.transpose(np.zeros((xtrain.shape[1]))))
	w = w.reshape((len(w),1))
	LR = Logistic_Regression(w)
	w, numiter, cost = LR.fit(xtrain, ytrain)

	#Plotting cost vs no of iterations
	#cost1 = np.asarray(cost, dtype=np.float32)
	#cost1 = cost1.reshape((len(cost),1))
	#plt.title('Loss function')
	#plt.xlabel('iterations')
	#plt.xlabel('loss function')
	#plt.plot(cost1)
	#plt.show()

	pred = LR.predict(xtest)
	accuracy = LR.calc_accuracy(ytest,pred)
	print("Accuracy without using k-fold:", accuracy)
	
	#Function call for k-fold validation
	#accuracy_kfold = LR.crossvalidation(xtrain, ytrain, 5)
	#print("Accuracy using k-fold is:", accuracy_kfold)
	#accuracy_cm= LR.conf_matrix(ytest, pred)
	
	#print("Estimated regression coefficients:", w) 
	#print("No. of iterations:", numiter) 
	#print("Accuracy is:", accuracy)
	#print("Accuracy using confusion matrix is:", accuracy_cm)
'''
	#Alpha and Number of Iterations plot

	alpha_vector = [0.0001, 0.001, 0.01, 0.05, 0.1]
	accuracy_cv = []

	for j in alpha_vector:

		w = np.array(np.transpose(np.zeros((xtrain.shape[1]))[np.newaxis]))
		LR = Logistic_Regression(w)
		accuracy_cv.append(LR.crossvalidation(xtrain, ytrain, 5, j, 500))
	print(accuracy_cv)
	plt.plot(alpha_vector, accuracy_cv, '.-')
	plt.title('Accuracy vs Learning rate alpha for Ionosphere data ')
	plt.xlabel('Learning rate')
	plt.ylabel('Accuracy using cross validation')
	plt.show()


	#No of instances and accuracy plot
	instance_vector = [10, 50, 100, 200, 300]
	accuracy = []

	for k in instance_vector:
		xtr = np.array(train_data.iloc[:k,:-1])
		ytr = np.array(train_data.iloc[:k,-1])
		xte = np.array((test_data.iloc[:k,:-1]))
		yte = np.array((test_data.iloc[:k,-1]))
		w = np.array(np.transpose(np.zeros((xtrain.shape[1]))[np.newaxis]))
		LR = Logistic_Regression(w)
		w = LR.fit(xtr, ytr)
		p = LR.predict(xte)
		a1 = LR.calc_accuracy(yte, p)
		accuracy.append(a1)
		print(accuracy)

	print(accuracy)'''
'''
	#code for ploting accuracy wrt instance_vector
	instance_vector = [10, 50, 100, 200, 300]
	accuracy = []
  
	for j in instance_vector:

		w = np.array(np.transpose(np.zeros((xtrain.shape[1]))[np.newaxis]))
		LR = Logistic_Regression(w)
		accuracy.append(LR.crossvalidation(xtrain, ytrain, 5, 0.01,j))
	print(accuracy)
	plt.plot(instance_vector, accuracy, '.-')
	plt.title('Accuracy vs Learning rate alpha for Ionosphere data ')
	plt.xlabel('Learning rate')
	plt.ylabel('Accuracy using cross validation')
	plt.show()'''